"""

Runs training for deepInsight

"""

# -----------------------------------------------------------------------

import h5py
import numpy as np
import torch

import deep_insight.loss
import deep_insight.networks
from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets
from utils.logger import logger

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class model_wrapper:
    def __init__(self):
        self.model = None

    def setup_model(self, h5_path, model_path):
        hdf5_file = h5py.File(h5_path, mode="r")
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
            h5_path, train_test_times=(np.array([]), np.array([]))
        )
        training_options["loss_functions"] = loss_functions.copy()
        training_options["loss_weights"] = loss_weights
        training_options["loss_names"] = list(loss_functions.keys())
        training_options["shuffle"] = False

        exp_indices = np.arange(
            0, wavelets.shape[0] - training_options["model_timesteps"]
        )
        cv_splits = np.array_split(exp_indices, training_options["num_cvs"])

        training_indices = []
        for arr in cv_splits[0:-1]:
            training_indices += list(arr)
        training_indices = np.array(training_indices)

        test_indeces = np.array(cv_splits[-1])
        # opts -> generators -> model
        # reset options for this cross validation set
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
        model_function = getattr(deep_insight.networks, train_dataset.model_function)
        self.model = model_function(train_dataset, show_summary=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_odometry(self, data, ret_loc=False):
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)
        if self.model is None:
            logger.error(
                "ERROR: Model has not been instantiated. Call setup_model first"
            )
            raise TypeError(
                "Model is NoneType; it has not been instantiated. Call setup_model first."
            )
        tensor = data.unsqueeze(0)
        logits = self.model(tensor)
        position_ests = list(logits[0])[0]
        angle_ests = list(logits[1])[0]
        speed_ests = list(logits[2])[0]
        if ret_loc == False:
            return speed_ests[0].item() * 150, angle_ests.item() * np.pi / 180.0
        else:
            speed = speed_ests[0].item() * 150
            ang = angle_ests[0].item() * np.pi / 180.0
            return speed, ang, [p.item() for p in position_ests]


MODEL = model_wrapper()
