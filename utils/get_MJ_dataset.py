# """
#
# Runs training for deepInsight
#
# """
# # -----------------------------------------------------------------------
#
# from deep_insight.options import get_opts, H5_PATH
# from deep_insight.wavelet_dataset import create_train_and_test_datasets
# import deep_insight.loss
# import deep_insight.networks
# import h5py
# import numpy as np
# import torch
# from ratSLAM.input import MattJonesDummyInput, MattJonesInput
#
# # -----------------------------------------------------------------------
#
# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
# else:
#     DEVICE = torch.device("cpu")
#
#
# #PREPROCESSED_HDF5_PATH = './data/processed_R2478.h5'
# PREPROCESSED_HDF5_PATH = H5_PATH
# hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode='r')
# wavelets = np.array(hdf5_file['inputs/wavelets'])
# loss_functions = {'position': 'euclidean_loss',
#                   #'head_direction': 'cyclical_mae_rad',
#                   'direction': 'cyclical_mae_rad',
#                   'speed': 'mae'}
# # Get loss functions for each output
# for key, item in loss_functions.items():
#     function_handle = getattr(deep_insight.loss, item)
#     loss_functions[key] = function_handle
#
# loss_weights = {'position': 1,
#                 #'head_direction': 25,
#                 'direction': 25,
#                 'speed': 2}
# # ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
# training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(np.array([]), np.array([])))
# training_options['loss_functions'] = loss_functions.copy()
# training_options['loss_weights'] = loss_weights
# training_options['loss_names'] = list(loss_functions.keys())
# training_options['shuffle'] = False
#
# exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
# cv_splits = np.array_split(exp_indices, training_options['num_cvs'])
#
# training_indices = []
# for arr in cv_splits[0:-1]:
#     training_indices += list(arr)
# training_indices = np.array(training_indices)
#
# test_indeces = np.array(cv_splits[-1])
# # opts -> generators -> model
# # reset options for this cross validation set
# training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=([training_indices], [test_indeces]))
# training_options['loss_functions'] = loss_functions.copy()
# training_options['loss_weights'] = loss_weights
# training_options['loss_names'] = list(loss_functions.keys())
# training_options['shuffle'] = False
# training_options['random_batches'] = False
#
# train_dataset, test_dataset = create_train_and_test_datasets(training_options, [hdf5_file])
# dataset = []
#
# for t in test_dataset:
#     d = t[0]
#     labels = [t_2 for t_2 in t[1]]
#     #dataset.append( MattJonesDummyInput( (d, labels) ) )
#     dataset.append( MattJonesInput( (d, labels) ) )
#
# def get_mj_dataset():
#     return dataset

# -----------------------------------------------------------------------------------------------------------------

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

import deep_insight.loss
import deep_insight.networks
from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets
from ratSLAM.input import MattJonesDummyInput, MattJonesInput
from utils.use_DI_model import get_odometry

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------

PREPROCESSED_HDF5_PATH = "data/Elliott.h5"
MODEL_PATH = "models/Elliott_0.pt"


def get_mj_dataset():
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
    dataset = []
    for t in test_dataset:
        d = t[0]
        labels = [t_2 for t_2 in t[1]]
        # dataset.append( MattJonesDummyInput( (d, labels) ) )
        dataset.append(MattJonesInput((d, labels)))
    return dataset
