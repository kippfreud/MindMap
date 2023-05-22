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
import time

# -----------------------------------------------------------------------

DATA_DIR = "/home/Tharg/Projects/DeepInsight/data/"
DATA_ID = "E-2009-06-24"
USE_WANDB = False

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

if __name__ == '__main__':
    rat_name = "Elliott"
    start_time = time.time()
    if USE_WANDB: wandb.init(project=rat_name)
    #PREPROCESSED_HDF5_PATH = './data/processed_R2478.h5'
    PREPROCESSED_HDF5_PATH = f'{DATA_DIR}{DATA_ID}.h5'
    hdf5_file = h5py.File(PREPROCESSED_HDF5_PATH, mode='r')
    wavelets = np.array(hdf5_file['inputs/wavelets'])
    frequencies = np.array(hdf5_file['inputs/fourier_frequencies'])

    loss_functions = {'position': 'euclidean_loss',
                      'head_direction': 'cyclical_mae_rad',
                      #'direction': 'cyclical_mae_rad',
                      #'direction_delta': 'cyclical_mae_rad',
                      'speed': 'mae'}
    # Get loss functions for each output
    for key, item in loss_functions.items():
        function_handle = getattr(deep_insight.loss, item)
        loss_functions[key] = function_handle

    loss_weights = {'position': 1,
                    'head_direction': 200,  #was 10, tweaked for MJ
                    'direction': 200,  # was 10, tweaked for MJ
                    #'direction_delta': 10,  # was 10, tweaked for MJ
                    'speed': 50} #was 2 but tweaked for MJ dataset

    # ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
    training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(np.array([]), np.array([])))

    exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
    cv_splits = np.array_split(exp_indices, training_options['num_cvs'])

    for cv_run, cvs in enumerate(cv_splits):
        # For cv
        training_indices = np.setdiff1d(exp_indices, cvs)  # All except the test indices
        testing_indices = cvs
        # opts -> generators -> model
        # reset options for this cross validation set
        training_options = get_opts(PREPROCESSED_HDF5_PATH, train_test_times=(training_indices, testing_indices))
        training_options['loss_functions'] = loss_functions.copy()
        training_options['loss_weights'] = loss_weights
        training_options['loss_names'] = list(loss_functions.keys())

        train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_file)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_options['batch_size'],
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=training_options['batch_size'],
            shuffle=False,
            num_workers=1,
            pin_memory=True)

        model_function = getattr(deep_insight.networks, train_dataset.model_function)
        model = model_function(train_dataset, show_summary=False)

        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=train_dataset.learning_rate,
                                     amsgrad=True)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=(loss_functions, loss_weights),
            optimizer=optimizer,
            device=DEVICE,
            use_wandb=USE_WANDB
        )

        trainer.train()

        #torch.save(model.state_dict(), f"models/{rat_name}.pt")

        print("Done!")
        print("TIME:")
        print(time.time() - start_time)
