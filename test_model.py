"""

Runs training for deepInsight

"""
# -----------------------------------------------------------------------

from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets, WaveletDataset
from deep_insight.trainer import Trainer
import deep_insight.loss
import deep_insight.networks
import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import imageio

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

takeout = False

#PREPROCESSED_HDF5_PATH = './data/processed_R2478.h5'
PREPROCESSED_HDF5_PATH = 'data/grid_world.h5'
MODEL_PATH = 'models/trained_grid_world_500.pt'

hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode='r')
wavelets = np.array(hdf5_file['inputs/wavelets'])
loss_functions = {'position': 'euclidean_loss',
                  'head_direction': 'cyclical_mae_rad',
                  'direction': 'cyclical_mae_rad',
                  'speed': 'mae'}
# Get loss functions for each output
for key, item in loss_functions.items():
    function_handle = getattr(deep_insight.loss, item)
    loss_functions[key] = function_handle

loss_weights = {'position': 1,
                'head_direction': 25,
                'direction': 25,
                'speed': 2}
# ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(np.array([]), np.array([])))
training_options['loss_functions'] = loss_functions.copy()
training_options['loss_weights'] = loss_weights
training_options['loss_names'] = list(loss_functions.keys())
training_options['shuffle'] = False

exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
cv_splits = np.array_split(exp_indices, training_options['num_cvs'])

training_indices = []
for arr in cv_splits[0:-1]:
    training_indices += list(arr)
training_indices = np.array(training_indices)

test_indeces = np.array(cv_splits[-1])
# opts -> generators -> model
# reset options for this cross validation set
training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(training_indices, test_indeces))
training_options['loss_functions'] = loss_functions.copy()
training_options['loss_weights'] = loss_weights
training_options['loss_names'] = list(loss_functions.keys())
training_options['shuffle'] = False
training_options['random_batches'] = False

train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_file)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=training_options['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=training_options['batch_size'],
    shuffle=False,
    num_workers=0,
    pin_memory=True)

model_function = getattr(deep_insight.networks, train_dataset.model_function)
model = model_function(train_dataset, show_summary=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

plt.ion()
pos_losses = []
hd_losses = []
speed_losses = []
P = 1

fig = plt.figure(figsize=(10., 4.))

position_ax = fig.add_subplot(121, facecolor='#E6E6E6')
position_ax.set_xlim([0, 750])
position_ax.set_ylim([0, 600])

compass_ax = fig.add_subplot(122, polar=True, facecolor='#E6E6E6')
compass_ax.set_ylim(0, 5)
compass_ax.set_yticks(np.arange(0, 5, 1.0))

# arr1 = compass_ax.arrow(0, 0.5, 0, 1, alpha=0.5, width=0.05,
#                  edgecolor='black', facecolor='green', lw=2, zorder=5)
# # arrow at 45 degree
# arr2 = compass_ax.arrow(45 / 180. * np.pi, 0.5, 0, 1, alpha=0.5, width=0.05,
#                  edgecolor='black', facecolor='green', lw=2, zorder=5)

poses = []
for batch, labels in test_loader:
    pos = labels[0]
    poses += pos
plt.scatter([p[0] for p in poses], [p[1] for p in poses])

ALL_TH = []
ALL_SP = []
ALL_THE = []
ALL_SPE = []

with imageio.get_writer('test.gif', mode='I') as writer:
    for batch, labels in test_loader:
        logits = model(batch)

        position_ests = list(logits[0])
        angle_ests = list(logits[2])
        speed_ests = list(logits[3])

        position = list(labels[0])
        angle = list(labels[2])
        speeds = list(labels[3])

        plot_positions = position
        plot_position_ests = position_ests

        plot_angle = angle
        plot_angle_ests = angle_ests

        plot_speed = speeds
        plot_speed_ests = speed_ests

        # radar green, solid grid lines
        plt.rc('grid', color='#316931', linewidth=1, linestyle='-')
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        # force square figure and square axes looks better for polar, IMO
        # width, height = matplotlib.rcParams['figure.figsize']
        # size = min(width, height)
        # make a square figure
        #ax.plot()
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

        for i in range(len(plot_positions)):
            # plt.clf()
            # plt.xlim([0,200])
            # plt.ylim([0, 200])
            # plt.xlim([0,750])
            # plt.ylim([0,600])
            position_ax.clear()
            compass_ax.clear()

            position_ax.set_xlim([0, 750])
            position_ax.set_ylim([0, 600])

            compass_ax.set_ylim(0, 0.02)
            compass_ax.set_yticks(np.arange(0, 0.2, 0.05))

            position_ax.scatter([plot_positions[i][0]], [plot_positions[i][1]], c="green")
            position_ax.scatter([plot_position_ests[i][0].item()], [plot_position_ests[i][1].item()], c="red")
            tr_x = speeds[i].item()*np.cos(angle[i].item()*(2.*np.pi / 360. ))
            tr_y = speeds[i].item() * np.sin(angle[i].item() * (2. * np.pi / 360.))
            tr_y = -1.
            ty_x = -1.
            es_x = speed_ests[i].item() * np.cos(angle_ests[i].item() * (2. * np.pi / 360.))
            es_y = speed_ests[i].item() * np.sin(angle_ests[i].item() * (2. * np.pi / 360.))
            es_x = -1.
            es_y = -1.

            th = (angle[i].item())
            sp = speeds[i].item()

            compass_ax.arrow(0, 0,
                             #-5.,5.,
                             th, sp,
                             alpha=0.5, width=0.1,
                             edgecolor='black', facecolor='green', lw=2, zorder=5)
            the = angle_ests[i].item()
            spe = speed_ests[i].item()

            ALL_TH.append(th)
            ALL_SP.append(sp)
            ALL_THE.append(the)
            ALL_SPE.append(spe)

            compass_ax.arrow(0, 0,
                             the, spe,
                             alpha=0.5, width=0.1,
                             edgecolor='black', facecolor='red', lw=2, zorder=5)
            plt.draw()
            plt.pause(0.0001)
            print(f"{P}")
            #plt.savefig(f"imgs/{P}.png")
            P += 1

            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

    pos_losses.append( training_options['loss_functions']['position'](labels[0], logits[0]).mean().detach().numpy().item() )
    hd_losses.append(
        training_options['loss_functions']['head_direction'](labels[0], logits[0]).mean().detach().numpy().item())
    speed_losses.append(
        training_options['loss_functions']['speed'](labels[0], logits[0]).mean().detach().numpy().item())

    # for i in range(len(position)):
    #     for j in range(len(list(position[i]))):
    #         plt.scatter(list(position[i])[j].detach().numpy()[0], list(position[i])[j].detach().numpy()[1], c="green")
    #         plt.scatter(list(position_ests[i])[j].detach().numpy()[0], list(position_ests[i])[j].detach().numpy()[1], c="red")
    #         plt.draw()
    #         plt.pause(0.0001)
    # print("ok")
pos_losses = torch.tensor(pos_losses)
hd_losses = torch.tensor(hd_losses)
speed_losses = torch.tensor(speed_losses)

print(f"Position loss: {pos_losses.mean()}")
print(f"HD loss: {hd_losses.mean()}")
print(f"Speed loss: {speed_losses.mean()}")

print("DONE!")