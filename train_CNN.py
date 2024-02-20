"""

Runs training for decoding CNN

"""
# -----------------------------------------------------------------------

from deep_insight.options import get_opts
from deep_insight.wavelet_dataset import create_train_and_test_datasets
from deep_insight.trainer import Trainer
import deep_insight.loss
import deep_insight.networks
import h5py
import numpy as np
import torch
import wandb
import time
import glob

# -----------------------------------------------------------------------

DATA_DIR = "./data/"
#RAT = "F"
RAT_NAME = "Elliott"
#DATA_FILES = [f for f in glob.glob(f"{DATA_DIR}{RAT}*train.h5") if "PC" not in f and "PFC" not in f]
DATA_FILES = [f"{DATA_DIR}{RAT_NAME}_train.h5"]
USE_WANDB = True

# -----------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

if __name__ == '__main__':
    start_time = time.time()
    if USE_WANDB: wandb.init(project=RAT_NAME,
                             entity="wolgast1")

    hdf5_files = [h5py.File(f, mode='r') for f in DATA_FILES]
    wavelets = [np.array(hdf5_file['inputs/wavelets']) for hdf5_file in hdf5_files]
    frequencies = [np.array(hdf5_file['inputs/fourier_frequencies']) for hdf5_file in hdf5_files]

    loss_functions = {'position': 'euclidean_loss',
                      #'head_direction': 'cyclical_mae_rad',
                      'direction': 'cyclical_mae_rad',
                      'speed': 'mae'
                      }
    # Get loss functions for each output
    for key, item in loss_functions.items():
        function_handle = getattr(deep_insight.loss, item)
        loss_functions[key] = function_handle

    loss_weights = {'position': 1,
                    #'head_direction': 5000,  #was 10, tweaked for MJ
                    'direction': 10,  # was 10, tweaked for MJ
                    'speed': 2500 #was 2 but tweaked for MJ dataset
                    }

    # second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
    training_options = get_opts(hdf5_files, train_test_times=(np.array([]), np.array([])))

    exp_indices = []
    cv_splits = []
    for w in wavelets:
        w_inds = np.arange(0, w.shape[0] - training_options['model_timesteps'])
        exp_indices.append( w_inds )
        cv_splits.append(np.array_split(w_inds, training_options['num_cvs']))

    for cv_run in range(training_options['num_cvs']):
        # For cv
        training_indices = []
        testing_indices = []
        for i in range(len(wavelets)):
            training_indices.append( np.setdiff1d(exp_indices[i], cv_splits[i][cv_run]) ) # All except the test indices
            testing_indices.append( cv_splits[i][cv_run] )

        # reset options for this cross validation set
        training_options = get_opts(hdf5_files, train_test_times=(training_indices, testing_indices))
        training_options['loss_functions'] = loss_functions.copy()
        training_options['loss_weights'] = loss_weights
        training_options['loss_names'] = list(loss_functions.keys())

        train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_files)

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

        torch.save(model.state_dict(), f"models/{RAT_NAME}_{cv_run}.pt")
        print(f"Training Complete: Trained in {time.time() - start_time}")
        exit(0)
