"""

Runs training for deepInsight

"""

# -----------------------------------------------------------------------

import h5py
import numpy as np
import torch

import deep_insight.loss
import deep_insight.networks
from deep_insight.options import H5_PATH, MODEL_PATH, get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# PREPROCESSED_HDF5_PATH = './data/processed_R2478.h5'
PREPROCESSED_HDF5_PATH = H5_PATH
hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode="r")
wavelets = np.array(hdf5_file["inputs/wavelets"])
loss_functions = {
    "position": "euclidean_loss",
    #'head_direction': 'cyclical_mae_rad',
    "direction": "cyclical_mae_rad",
    #'direction_delta': 'cyclical_mae_rad',
    "speed": "mae",
}
# Get loss functions for each output
for key, item in loss_functions.items():
    function_handle = getattr(deep_insight.loss, item)
    loss_functions[key] = function_handle

loss_weights = {
    "position": 1,
    #'head_direction': 25,
    "direction": 25,
    #'direction_delta': 25,
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

test_indeces = np.array(cv_splits[-1])
# opts -> generators -> model
# reset options for this cross validation set
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
model_function = getattr(deep_insight.networks, train_dataset.model_function)
MODEL = model_function(train_dataset, show_summary=False)
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.eval()


def get_odometry(data, ret_loc=False):
    if not torch.is_tensor(data):
        data = torch.from_numpy(data)
    tansor = data.unsqueeze(0)
    logits = MODEL(tansor)
    position_ests = list(logits[0])[0]
    angle_ests = list(logits[1])[0]
    speed_ests = list(logits[2])[0]
    if ret_loc == False:
        return speed_ests[0].item() * 150, angle_ests.item() * np.pi / 180.0
    else:
        speed = speed_ests[0].item() * 150
        ang = angle_ests[0].item() * np.pi / 180.0
        return speed, ang, [p.item() for p in position_ests]
