"""

Runs training for deepInsight

"""

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

import deep_insight.loss
import deep_insight.networks
from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets
from utils.get_MJ_dataset import get_mj_dataset
from utils.use_DI_model import get_odometry

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------

PREPROCESSED_HDF5_PATH = "data/Elliott_train.h5"
MODEL_PATH = "models/Elliott_0.pt"

# Setup datasets
hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode="r")
wavelets = np.array(hdf5_file["inputs/wavelets"])
loss_functions = {
    "position": "euclidean_loss",
    #'head_direction': 'cyclical_mae_rad',
    "direction": "cyclical_mae_rad",
    "speed": "mae",
}
for key, item in loss_functions.items():
    function_handle = getattr(deep_insight.loss, item)
    loss_functions[key] = function_handle
loss_weights = {
    "position": 1,
    #'head_direction': 25,
    "direction": 25,
    "speed": 2,
}
training_options = get_opts(
    PREPROCESSED_HDF5_PATH, train_test_times=(np.array([]), np.array([]))
)
training_options["loss_functions"] = loss_functions.copy()
training_options["loss_weights"] = loss_weights
training_options["loss_names"] = list(loss_functions.keys())
training_options["shuffle"] = False
exp_indices = np.arange(0, wavelets.shape[0] - training_options["model_timesteps"])
cv_splits = np.array_split(exp_indices, training_options["num_cvs"])
training_indices = []
for arr in cv_splits[0:-1]:
    training_indices += list(arr)
training_indices = np.array(training_indices)
test_indeces = np.array(cv_splits[1])
training_options = get_opts(
    PREPROCESSED_HDF5_PATH, train_test_times=([training_indices], [test_indeces])
)
training_options["loss_functions"] = loss_functions.copy()
training_options["loss_weights"] = loss_weights
training_options["loss_names"] = list(loss_functions.keys())
training_options["shuffle"] = False
training_options["random_batches"] = False
train_dataset, test_dataset = create_train_and_test_datasets(
    training_options, [hdf5_file]
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=training_options["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=training_options["batch_size"],
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)
model_function = getattr(deep_insight.networks, train_dataset.model_function)
model = model_function(train_dataset, show_summary=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Setup Plotting
plt.ion()
pos_losses = []
hd_losses = []
speed_losses = []
P = 1
fig = plt.figure(figsize=(15.0, 6.0))
position_ax = fig.add_subplot(121, facecolor="#E6E6E6")
position_ax.set_xlim([0, 750])
position_ax.set_ylim([0, 600])
compass_ax = fig.add_subplot(122, polar=True, facecolor="#E6E6E6")
compass_ax.set_ylim(0, 5)
compass_ax.set_yticks(np.arange(0, 5, 1.0))
plt.rc("grid", color="#316931", linewidth=1, linestyle="-")
plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=15)

with imageio.get_writer("testt.gif", mode="I") as writer:
    for d in get_mj_dataset():
        speed_est, ang_est, pos_est = get_odometry(d.raw_data[0], True)
        pos, ang, speed = tuple(d.raw_data[1])

        position_ax.clear()
        compass_ax.clear()
        position_ax.set_xlim([0, 750])
        position_ax.set_ylim([0, 600])
        compass_ax.set_ylim(0, 0.02)
        compass_ax.set_yticks(np.arange(0, 0.2, 0.05))
        position_ax.scatter([pos[0]], [pos[1]], c="green")
        position_ax.scatter([pos_est[0]], [pos_est[1]], c="red")
        compass_ax.arrow(
            0,
            0,
            ang.item() * np.pi / 180,
            speed.item(),
            alpha=0.5,
            width=0.04,
            edgecolor="black",
            facecolor="green",
            lw=2,
            zorder=5,
        )
        compass_ax.arrow(
            0,
            0,
            ang_est,
            speed_est,
            alpha=0.5,
            width=0.04,
            edgecolor="black",
            facecolor="red",
            lw=2,
            zorder=5,
            head_length=0,
        )
        plt.draw()
        plt.pause(0.1)
