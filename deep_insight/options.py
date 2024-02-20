"""
Defines options dict for training, network params, etc.
"""

# ---------------------------------------------------------------------

RAT_NAME = "Elliott"
MODEL_PATH = f"models/{RAT_NAME}_0.pt"
H5_PATH = f"data/{RAT_NAME}.h5"

# ---------------------------------------------------------------------

def get_opts(fp_hdf_out, train_test_times):
    """
    Returns the options dictionary which contains all parameters needed to train the model
    """
    opts = dict()

    # -------- DATA ------------------------
    opts['fp_hdf_out'] = fp_hdf_out  # Filepath for hdf5 file storing wavelets and outputs
    opts['sampling_rate'] = 512*4 # Sampling rate of the wavelets
    opts['training_indices'] = train_test_times[0]  # Indices into wavelets used for training the model, adjusted during CV
    opts['testing_indices'] = train_test_times[1]  # Indices into wavelets used for testing the model, adjusted during CV
    opts['channels'] = 16

    # -------- MODEL PARAMETERS --------------
    opts['model_function'] = 'Standard_Decoder'  # Model architecture used
    opts['model_timesteps'] = 64  # How many timesteps are used in the input layer, e.g. a sampling rate of 30 will yield 2.13s windows. Has to be divisible X times by 2. X='num_convs_tsr'
    opts['num_convs_tsr'] = 5  # Number of downsampling steps within the model, e.g. with model_timesteps=64, it will downsample 64->32->16->8->4 and output 4 timesteps
    opts['learning_rate'] = 0.0007 # Learning rate
    opts['kernel_size'] = 3  # Kernel size for all convolutional layers
    opts['act_conv'] = 'ELU'  # Activation function for convolutional layers
    opts['act_fc'] = 'ELU'  # Activation function for fully connected layers
    opts['dropout_ratio'] = 0  # Dropout ratio for fully connected layers
    opts['filter_size'] = 64  # Number of filters in convolutional layers
    opts['num_units_dense'] = 1024  # Number of units in fully connected layer
    opts['num_dense'] = 2  # Number of fully connected layers

    # -------- TRAINING----------------------
    opts['batch_size'] = 32  # Batch size used for training the model
    opts['steps_per_epoch'] = 250  # Number of steps per training epoch
    opts['validation_steps'] = 250  # Number of steps per validation epoch #..todo: fix validation
    opts['epochs'] = 50  # Number of epochs
    opts['shuffle'] = False  # If input should be shuffled
    opts['random_batches'] = True  # If random batches in time are used
    opts['num_cvs'] = 5 # the number of cross validation splits

    return opts