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

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


PREPROCESSED_HDF5_PATH = './data/processed_R2478.h5'
hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode='r')
wavelets = np.array(hdf5_file['inputs/wavelets'])
loss_functions = {'position': 'euclidean_loss',
                  'head_direction': 'cyclical_mae_rad',
                  'speed': 'mae'}
# Get loss functions for each output
for key, item in loss_functions.items():
    function_handle = getattr(deep_insight.loss, item)
    loss_functions[key] = function_handle

loss_weights = {'position': 1,
                'head_direction': 25,
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
model.load_state_dict(torch.load('models/trained_0.pt'))
model.eval()

plt.ion()
pos_losses = []
hd_losses = []
speed_losses = []
P = 1
for batch, labels in test_loader:
    logits = model(batch)
    position_ests = list(logits[0])
    position = list(labels[0])
    plot_positions = [p[3] for p in position]
    plot_position_ests = [p[3] for p in position_ests]
    for i in range(len(plot_positions)):
        plt.clf()
        plt.xlim([0,200])
        plt.ylim([0, 200])
        plt.scatter([plot_positions[i][0]], [plot_positions[i][1]], c="green")
        plt.scatter([plot_position_ests[i][0].item()], [plot_position_ests[i][1].item()], c="red")
        plt.draw()
        plt.pause(0.0001)
        print(f"{P}")
        plt.savefig(f"imgs/{P}.png")
        P += 1
    # pos_losses.append( training_options['loss_functions']['position'](labels[0], logits[0]).mean().detach().numpy().item() )
    # hd_losses.append(
    #     training_options['loss_functions']['head_direction'](labels[0], logits[0]).mean().detach().numpy().item())
    # speed_losses.append(
    #     training_options['loss_functions']['speed'](labels[0], logits[0]).mean().detach().numpy().item())
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