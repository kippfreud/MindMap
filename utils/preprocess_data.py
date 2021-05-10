"""
This should perform necessary preprocessing of Matt Jones' data.
"""

#------------------------------------------------------------------------

from deep_insight.options import get_opts
import pynwb
import scipy.io
import h5py
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
import numpy as np
import torch
import wandb
import deep_insight.loss
from deep_insight.wavelet_dataset import create_train_and_test_datasets, WaveletDataset
from wavelets import WaveletAnalysis
import time
from joblib import Parallel, delayed
import numpy as np
import h5py
import tensorflow as tf  # Progress bar only

#------------------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
USE_WANDB = False


def preprocess_input(fp_hdf_out, hdf5_in, average_window=1000, channels=None, window_size=100000,
                     gap_size=50000, sampling_rate=30000, scaling_factor=0.5, num_cores=1, **args):
    """
    Transforms raw neural data to frequency space, via wavelet transform implemented currently with aaren-wavelets (https://github.com/aaren/wavelets)
    Saves wavelet transformed data to HDF5 file (N, P, M) - (Number of timepoints, Number of frequencies, Number of channels)
    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    raw_data : (N, M) file or array_like
        Variable storing the raw_data (N data points, M channels), should allow indexing
    average_window : int, optional
        Average window to downsample wavelet transformed input, by default 1000
    channels : array_like, optional
        Which channels from raw_data to use, by default None
    window_size : int, optional
        Window size for calculating wavelet transformation, by default 100000
    gap_size : int, optional
        Gap size for calculating wavelet transformation, by default 50000
    sampling_rate : int, optional
        Sampling rate of raw_data, by default 30000
    scaling_factor : float, optional
        Determines amount of log-spaced frequencies P in output, by default 0.5
    num_cores : int, optional
        Number of paralell cores to use to calculate wavelet transformation, by default 4
    """
    raw_data = hdf5_in["Data/LFP"]
    # Get number of chunks
    if channels is None:
        channels = np.arange(0, raw_data.shape[1])
    num_points = raw_data.shape[0]
    if window_size > num_points:
        num_chunks = len(channels)
        output_size = raw_data.shape[0]
        mean_signal = np.mean(raw_data, axis=1)
        average_window = 1
        full_transform = True
    else:
        num_chunks = (num_points // gap_size) - 1
        output_size = ((num_chunks + 1) * gap_size) // average_window
        full_transform = False

    # Get estimate for number of frequencies
    (_, wavelet_frequencies, _) = wavelet_transform(np.ones(window_size), None, sampling_rate, average_window, scaling_factor, **args)
    num_fourier_frequencies = len(wavelet_frequencies)
    # Prepare output file
    hdf5_file = h5py.File(fp_hdf_out, mode='a')
    if "inputs/wavelets" not in hdf5_file:
        hdf5_file.create_dataset("inputs/wavelets", [output_size, num_fourier_frequencies, len(channels)], np.float32)
        hdf5_file.create_dataset("inputs/fourier_frequencies", [num_fourier_frequencies], np.float16)
        hdf5_file.create_dataset("outputs/position", [output_size, 2], np.float32)
        hdf5_file.create_dataset("outputs/head_direction", [output_size, 1], np.float32)
        hdf5_file.create_dataset("outputs/speed", [output_size, 1], np.float32)
        hdf5_file.create_dataset("outputs/direction", [output_size, 1], np.float32)
        hdf5_file.create_dataset("outputs/direction_delta", [output_size, 1], np.float32)
    else:
        print("ERROR!")
        exit(0)
    # Makes saving 5 times faster as last index saving is fancy indexing and therefore slow
    hdf5_file.create_dataset("inputs/tmp_wavelets", [len(channels), output_size, num_fourier_frequencies], np.float32)

    outputs = {
        "position": np.array(hdf5_in["Data/Pos"]),
        "head_direction": np.array(hdf5_in["Data/Theta"]),
        "speed": np.array(hdf5_in["Data/Speed"]),
        "direction": np.array(hdf5_in["Data/Direction"]),
        "direction_delta": np.array(hdf5_in["Data/DirectionDelta"])
    }

    # Prepare par pool
    par = Parallel(n_jobs=num_cores, verbose=0)

    # Start parallel wavelet transformation
    print('Starting wavelet transformation (n={}, chunks={}, frequencies={})'.format(
        num_points, num_chunks, num_fourier_frequencies))
    progress_bar = tf.keras.utils.Progbar(num_chunks, width=30, verbose=1, interval=0.05, unit_name='chunk')
    for c in range(0, num_chunks):
        if full_transform:
            raw_chunk = raw_data[:, c] - mean_signal
        else:
            start = gap_size * c
            end = start + window_size

            output_chunk = {
                "position": outputs["position"][start:end],
                "head_direction": outputs["head_direction"][start:end],
                "speed": outputs["speed"][start:end],
                "direction": outputs["direction"][start:end],
                "direction_delta": outputs["direction_delta"][start:end]
            }


            raw_chunk = raw_data[start: end, channels]
            # Process raw chunk
            raw_chunk = preprocess_chunk(raw_chunk, subtract_mean=True, convert_to_milivolt=False)

        # Calculate wavelet transform
        if full_transform:
            (wavelet_power, wavelet_frequencies) = wavelet_transform(raw_chunk,
                                                                        sampling_rate=sampling_rate, scaling_factor=scaling_factor, average_window=average_window, **args)
            print("ERROR!")
            exit(0)
        else:
            wavelet_transformed = np.zeros((raw_chunk.shape[0] // average_window, num_fourier_frequencies, raw_chunk.shape[1]))
            if output_chunk is not None:
                (wavelet_power, wavelet_frequencies, wavelet_obj) = simple_wavelet_transform(raw_chunk[:, 0], sampling_rate,
                                                                                             scaling_factor=scaling_factor,
                                                                                             wave_highpass=2, wave_lowpass=30000)
                for key in output_chunk.keys():
                    try:
                        output_chunk[key] = np.reshape(output_chunk[key],
                                                       (wavelet_power.shape[1] // average_window, average_window,
                                                        output_chunk[key].shape[1]))
                    except:
                        print("oo")
                    if key != "position":
                        output_chunk[key] = np.mean(output_chunk[key], axis=1)
                    else:
                        output_chunk[key] = output_chunk["position"][:, round(output_chunk["position"][:].shape[1]/2), :]

            for ind, (wavelet_power, wavelet_frequencies, output) in enumerate(par(delayed(wavelet_transform)(raw_chunk[:, i], output_chunk, sampling_rate, average_window, scaling_factor, **args) for i in range(0, raw_chunk.shape[1]))):
                wavelet_transformed[:, :, ind] = wavelet_power

        # Save in output file
        if full_transform:
            hdf5_file["inputs/tmp_wavelets"][c, :, :] = wavelet_power
        else:
            wavelet_index_end = end // average_window
            wavelet_index_start = start // average_window
            index_gap = gap_size // 2 // average_window
            if c == 0:
                this_index_start = 0
                this_index_end = wavelet_index_end - index_gap
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[0: -index_gap, :, :]
                hdf5_file["outputs/position"][this_index_start:this_index_end,:] = output_chunk["position"][0:-index_gap,:]
                hdf5_file["outputs/head_direction"][this_index_start:this_index_end,:] = output_chunk["head_direction"][0:-index_gap,:]
                hdf5_file["outputs/direction"][this_index_start:this_index_end, :] = output_chunk["direction"][0:-index_gap, :]
                hdf5_file["outputs/direction_delta"][this_index_start:this_index_end, :] = output_chunk["direction_delta"][0:-index_gap, :]
                hdf5_file["outputs/speed"][this_index_start:this_index_end,:] = output_chunk["speed"][0:-index_gap,:]
            elif c == num_chunks - 1:  # Make sure the last one fits fully
                this_index_start = wavelet_index_start + index_gap
                this_index_end = wavelet_index_end
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[index_gap::, :, :]
                hdf5_file["outputs/position"][this_index_start:this_index_end, :] = output_chunk["position"][index_gap::, :]
                hdf5_file["outputs/head_direction"][this_index_start:this_index_end, :] = output_chunk["head_direction"][index_gap::, :]
                hdf5_file["outputs/direction"][this_index_start:this_index_end, :] = output_chunk["direction"][index_gap::, :]
                hdf5_file["outputs/direction_delta"][this_index_start:this_index_end, :] = output_chunk["direction_delta"][index_gap::, :]
                hdf5_file["outputs/speed"][this_index_start:this_index_end, :] = output_chunk["speed"][index_gap::, :]
            else:
                this_index_start = wavelet_index_start + index_gap
                this_index_end = wavelet_index_end - index_gap
                hdf5_file["inputs/wavelets"][this_index_start:this_index_end, :, :] = wavelet_transformed[index_gap: -index_gap, :, :]

                hdf5_file["outputs/position"][this_index_start:this_index_end, :] = output_chunk["position"][index_gap:-index_gap, :]
                hdf5_file["outputs/head_direction"][this_index_start:this_index_end, :] = output_chunk["head_direction"][index_gap:-index_gap, :]
                hdf5_file["outputs/direction"][this_index_start:this_index_end, :] = output_chunk["direction"][index_gap:-index_gap, :]
                hdf5_file["outputs/direction_delta"][this_index_start:this_index_end, :] = output_chunk["direction_delta"][index_gap:-index_gap, :]
                hdf5_file["outputs/speed"][this_index_start:this_index_end, :] = output_chunk["speed"][index_gap:-index_gap, :]

        hdf5_file.flush()
        progress_bar.add(1)

    # 7.) Put frequencies in and close file
    if full_transform:
        wavelet_power = np.transpose(hdf5_file["inputs/tmp_wavelets"], axes=(1, 2, 0))
        del hdf5_file["inputs/tmp_wavelets"]
        hdf5_file["inputs/wavelets"][:] = wavelet_power
    hdf5_file["inputs/fourier_frequencies"][:] = wavelet_frequencies
    hdf5_file.flush()
    hdf5_file.close()


def preprocess_chunk(raw_chunk, subtract_mean=True, convert_to_milivolt=False):
    """
    Preprocesses a chunk of data.
    Parameters
    ----------
    raw_chunk : array_like
        Chunk of raw_data to preprocess
    subtract_mean : bool, optional
        Subtract mean over all other channels, by default True
    convert_to_milivolt : bool, optional
        Convert chunk to milivolt , by default False
    Returns
    -------
    raw_chunk : array_like
        preprocessed_chunk
    """
    # Subtract mean across all channels
    if subtract_mean:
        raw_chunk = raw_chunk.transpose() - np.mean(raw_chunk.transpose(), axis=0)
        raw_chunk = raw_chunk.transpose()
    # Convert to milivolt
    if convert_to_milivolt:
        raw_chunk = raw_chunk * (0.195 / 1000)
    return raw_chunk

def wavelet_transform(signal, output_chunk, sampling_rate, average_window=1000, scaling_factor=0.25, wave_highpass=2, wave_lowpass=30000):
    """
    Calculates the wavelet transform for each point in signal, then averages
    each window and returns together fourier frequencies
    Parameters
    ----------
    signal : (N,1) array_like
        Signal to be transformed
    sampling_rate : int
        Sampling rate of signal
    average_window : int, optional
        Average window to downsample wavelet transformed input, by default 1000
    scaling_factor : float, optional
        Determines amount of log-spaced frequencies M in output, by default 0.25
    wave_highpass : int, optional
        Cut of frequencies below, by default 2
    wave_lowpass : int, optional
        Cut of frequencies above, by default 30000
    Returns
    -------
    wavelet_power : (N, M) array_like
        Wavelet transformed signal
    wavelet_frequencies : (M, 1) array_like
        Corresponding frequencies to wavelet_power
    """
    (wavelet_power, wavelet_frequencies, wavelet_obj) = simple_wavelet_transform(signal, sampling_rate,
                                                                                 scaling_factor=scaling_factor, wave_highpass=wave_highpass, wave_lowpass=wave_lowpass)

    # Average over window
    if average_window is not 1:
        # if output_chunk is not None:
        #     for key in output_chunk.keys():
        #         output_chunk[key] = np.reshape( output_chunk[key] ,
        #                                         (wavelet_power.shape[1] // average_window, average_window , output_chunk[key].shape[1]) )
        #         output_chunk[key] = np.mean(output_chunk[key], axis=1)
        wavelet_power = np.reshape(
            wavelet_power, (wavelet_power.shape[0], wavelet_power.shape[1] // average_window, average_window))
        wavelet_power = np.mean(wavelet_power, axis=2).transpose()
    else:
        wavelet_power = wavelet_power.transpose()

    return wavelet_power, wavelet_frequencies, output_chunk

def create_or_update(hdf5_file, dataset_name, dataset_shape, dataset_type, dataset_value):
    """
    Create or update dataset in HDF5 file
    Parameters
    ----------
    hdf5_file : File
        File identifier
    dataset_name : str
        Name of new dataset
    dataset_shape : array_like
        Shape of new dataset
    dataset_type : type
        Type of dataset (np.float16, np.float32, 'S', etc...)
    dataset_value : array_like
        Data to store in HDF5 file
    """
    if not dataset_name in hdf5_file:
        hdf5_file.create_dataset(dataset_name, dataset_shape, dataset_type)
        hdf5_file[dataset_name][:] = dataset_value
    else:
        hdf5_file[dataset_name][:] = dataset_value
    hdf5_file.flush()

def preprocess_output(fp_hdf_out, hdf5_with_output, average_window=1000, sampling_rate=30000):
    """
    Write behaviours to decode into HDF5 file
    Parameters
    ----------
    fp_hdf_out : str
        File path to HDF5 file
    raw_timestamps : (N,1) array_like
        Timestamps for each sample in continous
    output : (N,4) array_like
        Position of animal with two LEDs
    output_timestamps : (N,1) array_like
        Timestamps for positions
    average_window : int, optional
        Downsampling factor for raw data and positions, by default 1000
    sampling_rate : int, optional
        Sampling rate of raw ephys, by default 30000
    """
    hdf5_file = h5py.File(fp_hdf_out, mode='a')

    # Get size of wavelets
    input_length = hdf5_file['inputs/wavelets'].shape[0]

    # Get positions of both LEDs
    raw_timestamps = np.array(hdf5_with_output["Data/Timestamps"])


    raw_timestamps = raw_timestamps[()]  # Slightly faster than np.array
    output_x_led1 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                       average_window)], output_timestamps, output[:, 0])
    output_y_led1 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                       average_window)], output_timestamps, output[:, 1])
    output_x_led2 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                       average_window)], output_timestamps, output[:, 2])
    output_y_led2 = np.interp(raw_timestamps[np.arange(0, raw_timestamps.shape[0],
                                                       average_window)], output_timestamps, output[:, 3])
    raw_positions = np.array([output_x_led1, output_y_led1, output_x_led2, output_y_led2]).transpose()

    # Clean raw_positions and get centre
    positions_smooth = pd.DataFrame(raw_positions.copy()).interpolate(
        limit_direction='both').rolling(5, min_periods=1).mean().get_values()
    position = np.array([(positions_smooth[:, 0] + positions_smooth[:, 2]) / 2,
                         (positions_smooth[:, 1] + positions_smooth[:, 3]) / 2]).transpose()

    # Also get head direction and speed from positions
    speed = stats.calculate_speed_from_position(position, interval=1/(sampling_rate//average_window), smoothing=3)
    head_direction = stats.calculate_head_direction_from_leds(positions_smooth, return_as_deg=False)

    # Create and save datasets in HDF5 File
    create_or_update(hdf5_file, dataset_name="outputs/raw_position",
                          dataset_shape=[input_length, 4], dataset_type=np.float16, dataset_value=raw_positions[0: input_length, :])
    create_or_update(hdf5_file, dataset_name="outputs/position",
                          dataset_shape=[input_length, 2], dataset_type=np.float16, dataset_value=position[0: input_length, :])
    create_or_update(hdf5_file, dataset_name="outputs/head_direction", dataset_shape=[
                          input_length, 1], dataset_type=np.float16, dataset_value=head_direction[0: input_length, np.newaxis])
    create_or_update(hdf5_file, dataset_name="outputs/speed",
                          dataset_shape=[input_length, 1], dataset_type=np.float16, dataset_value=speed[0: input_length, np.newaxis])
    hdf5_file.flush()
    hdf5_file.close()

def simple_wavelet_transform(signal, sampling_rate, scaling_factor=0.25, wave_lowpass=None, wave_highpass=None):
    """
    Simple wavelet transformation of signal
    Parameters
    ----------
    signal : (N,1) array_like
        Signal to be transformed
    sampling_rate : int
        Sampling rate of signal
    scaling_factor : float, optional
        Determines amount of log-space frequencies M in output, by default 0.25
    wave_highpass : int, optional
        Cut of frequencies below, by default 2
    wave_lowpass : int, optional
        Cut of frequencies above, by default 30000
    Returns
    -------
    wavelet_power : (N, M) array_like
        Wavelet transformed signal
    wavelet_frequencies : (M, 1) array_like
        Corresponding frequencies to wavelet_power
    wavelet_obj : object
        WaveletTransform Object
    """
    wavelet_obj = WaveletAnalysis(signal, dt=1 / sampling_rate, dj=scaling_factor)
    wavelet_power = wavelet_obj.wavelet_power
    wavelet_frequencies = wavelet_obj.fourier_frequencies

    if wave_lowpass or wave_highpass:
        wavelet_power = wavelet_power[(wavelet_frequencies < wave_lowpass) & (wavelet_frequencies > wave_highpass), :]
        wavelet_frequencies = wavelet_frequencies[(wavelet_frequencies < wave_lowpass) & (wavelet_frequencies > wave_highpass)]

    return (wavelet_power, wavelet_frequencies, wavelet_obj)


if __name__ == '__main__':

    if USE_WANDB: wandb.init(project="my-project")

    HDF5_PATH = 'data/data.mat'
    hdf5_file = h5py.File(HDF5_PATH, mode='r')
    # ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
    training_options = get_opts(HDF5_PATH, train_test_times=(np.array([]), np.array([])))

    # wavelets = np.array(hdf5_file['inputs/wavelets'])
    # frequencies = np.array(hdf5_file['inputs/fourier_frequencies'])
    preprocess_input("data/grid_world.h5", hdf5_file, sampling_rate=training_options['sampling_rate'],
                     average_window=250,
                     channels=list(range(training_options['channels'])))

    # Prepare outputs
    # preprocess_output("data/preprocessed_MJ4.h5",
    #                   hdf5_file
    #                   average_window=250,
    #                   sampling_rate=training_options['sampling_rate'])


    # HDF5_PATH = "data/preprocessed_final.h5"
    # hdf5_file = h5py.File(HDF5_PATH, mode='r')
    # # ..todo: second param is unneccecary at this stage, use two empty arrays to match signature but it doesn't matter
    # training_options = get_opts(HDF5_PATH, train_test_times=(np.array([]), np.array([])))
    #
    # loss_functions = {'position': 'euclidean_loss',
    #                   'head_direction': 'cyclical_mae_rad',
    #                   'speed': 'mae'}
    # # Get loss functions for each output
    # for key, item in loss_functions.items():
    #     function_handle = getattr(deep_insight.loss, item)
    #     loss_functions[key] = function_handle
    #
    # loss_weights = {'position': 1,
    #                 'head_direction': 25,
    #                 'speed': 2}
    #
    # exp_indices = np.arange(0, wavelets.shape[0] - training_options['model_timesteps'])
    # cv_splits = np.array_split(exp_indices, training_options['num_cvs'])
    #
    # for cv_run, cvs in enumerate(cv_splits):
    #     # For cv
    #     training_indices = np.setdiff1d(exp_indices, cvs)  # All except the test indices
    #     testing_indices = cvs
    #     # opts -> generators -> model
    #     # reset options for this cross validation set
    #     training_options = get_opts(HDF5_PATH, train_test_times=(training_indices, testing_indices))
    #     training_options['loss_functions'] = loss_functions.copy()
    #     training_options['loss_weights'] = loss_weights
    #     training_options['loss_names'] = list(loss_functions.keys())
    #
    #     train_dataset, test_dataset = create_train_and_test_datasets(training_options, hdf5_file)

print("0")
