"""

Processes dataset from matt jones lab

"""

# -----------------------------------------------------------------------------------------------------------------

import h5py
import numpy as np

import deep_insight.loss
import deep_insight.networks
from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets
from ratSLAM.input import MattJonesInput

# -----------------------------------------------------------------------


def get_mj_dataset(h5_path):
    # Setup datasets
    hdf5_file = h5py.File(h5_path, mode="r")
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
    training_options = get_opts(h5_path, train_test_times=(np.array([]), np.array([])))
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
        h5_path, train_test_times=([training_indices], [test_indeces])
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
