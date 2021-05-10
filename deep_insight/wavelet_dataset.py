from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
from utils.misc import getspeed

def create_train_and_test_datasets(opts, hdf5_file):
    """
    Creates training and test datasets from given opts dictionary and hdf5 file.
    """
    # 1.) Create training generator
    training_generator = WaveletDataset(opts, hdf5_file, training=True)
    # 2.) Create testing generator
    testing_generator = WaveletDataset(opts, hdf5_file, training=False)

    return training_generator, testing_generator

class WaveletDataset(Dataset):
    """
    Dataset containing raw wavelet sequence; __getitem__ returns an (input, output) pair.
    ..todo: better docs
    """
    def __init__(self, opts, hdf5_file, training):
        # 1.) Set all options as attributes
        self.set_opts_as_attribute(opts)
        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = np.array(hdf5_file['inputs/wavelets'])

        self.last_pos = None
        self.last_speed = None
        self.last_ang = None
        self.prev_ind = None

        # Get output(s)
        outputs = []
        # the loss function dict has a key representing each output - loop through them and
        # get the outputs
        for key, value in opts['loss_functions'].items():
            tmp_out = hdf5_file['outputs/' + key]
            outputs.append(tmp_out)
        self.outputs = [np.array(o) for o in outputs]

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(training=training)

    def __len__(self):
        return len(self.cv_indices)

    def __getitem__(self, idx):
        # 1.) Define start and end index
        if self.shuffle:
            idx = np.random.choice(self.cv_indices)
        else:
            idx = self.cv_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size)
        past_cut_range = np.arange(idx-1,  idx+self.sample_size-1)
        # 2.) Above takes consecutive batches, below takes random batches
        if self.random_batches:
            start_index = np.random.choice(self.cv_indices, size=1)[0]
            cut_range = np.arange(start_index, start_index + self.model_timesteps)
            past_cut_range = np.arange(start_index-1, start_index+self.model_timesteps-1)

        # 3.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        # 4.) Get output sample
        output_sample = self.get_output_sample(cut_range, past_cut_range)

        self.prev_ind = idx
        #print(idx)

        return (input_sample, output_sample)

    # -------------------------------------------------------------------------
    # Public Function
    # -------------------------------------------------------------------------

    def prepare_data_generator(self, training):
        # Define sample size and means
        self.sample_size = self.model_timesteps
        # define indices depending on whether training or testing
        if training:
            self.cv_indices = self.training_indices
        else:
            self.cv_indices = self.testing_indices
        self.est_mean = np.median(self.wavelets[self.training_indices, :, :], axis=0)
        self.est_std = np.median(abs(self.wavelets[self.training_indices, :, :] - self.est_mean), axis=0)
        # Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, dummy_output) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[:]

    def set_opts_as_attribute(self, opts):
        """
        Sets all entries in opts dict to internal parameters with names equal to the
        option keys.
        """
        for k, v in opts.items():
            setattr(self, k, v)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        cut_data = self.wavelets[cut_range, :, :]

        # 2.) Normalize input
        cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input
        #cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(2, 0, 1))
        cut_data = cut_data[..., np.newaxis]

        return cut_data

    def get_output_sample(self, cut_range, prev_cut_range):
        # 1.) Cut Ephys
        out_sample = []
        for i, out in enumerate(self.outputs):
            cut_data = out[cut_range, ...]
            pcd = out[prev_cut_range, ...]

            # 3.) Divide evenly and make sure last output is being decoded

            # if self.average_output:
            #     cut_data = cut_data[np.arange(0, cut_data.shape[0] + 1, self.average_output)[1::] - 1]
            # out_sample.append(cut_data)
            ## ..todo: replaced above with below: kipp!
            if i == 0:
                #cut_data_m = np.mean(cut_data,0)
                cut_data_m = cut_data[-1, :]
                out_sample.append(cut_data_m)

                pcdm = pcd[-1,:]
            elif i == 1 or i == 2:
                # ch_y = bookends[1][1] - bookends[0][1]
                # ch_x = bookends[1][0] - bookends[0][0]
                # ang_rad = math.atan2(ch_x, ch_y)
                # out_sample.append(ang_rad*(180/np.pi))
                # return angle diff between cut data m and last pos
                out_sample.append(math.atan2(cut_data_m[1]-pcdm[1], cut_data_m[0]-pcdm[0]))
            elif i == 3:
                spd = getspeed(cut_data_m, pcdm)
                out_sample.append(spd)

        return out_sample

class WaveletDatasetFrey(Dataset):
    """
    Dataset containing raw wavelet sequence; __getitem__ returns an (input, output) pair.
    ..todo: better docs
    """
    def __init__(self, opts, hdf5_file, training):
        # 1.) Set all options as attributes
        self.set_opts_as_attribute(opts)
        # 2.) Load data memmaped for mean/std estimation and fast plotting
        self.wavelets = np.array(hdf5_file['inputs/wavelets'])

        # Get output(s)
        outputs = []
        # the loss function dict has a key representing each output - loop through them and
        # get the outputs
        for key, value in opts['loss_functions'].items():
            tmp_out = hdf5_file['outputs/' + key]
            outputs.append(tmp_out)
        self.outputs = [np.array(o) for o in outputs]

        # 3.) Prepare for training
        self.training = training
        self.prepare_data_generator(training=training)

    def __len__(self):
        return len(self.cv_indices)

    def __getitem__(self, idx):
        # 1.) Define start and end index
        if self.shuffle:
            idx = np.random.choice(self.cv_indices)
        else:
            idx = self.cv_indices[idx]
        cut_range = np.arange(idx, idx + self.sample_size)
        # 2.) Above takes consecutive batches, below takes random batches
        if self.random_batches:
            start_index = np.random.choice(self.cv_indices, size=1)[0]
            cut_range = np.arange(start_index, start_index + self.model_timesteps)

        # 3.) Get input sample
        input_sample = self.get_input_sample(cut_range)

        # 4.) Get output sample
        output_sample = self.get_output_sample(cut_range)

        return (input_sample, output_sample)

    # -------------------------------------------------------------------------
    # Public Function
    # -------------------------------------------------------------------------

    def prepare_data_generator(self, training):
        # Define sample size and means
        self.sample_size = self.model_timesteps
        # define indices depending on whether training or testing
        if training:
            self.cv_indices = self.training_indices
        else:
            self.cv_indices = self.testing_indices
        self.est_mean = np.median(self.wavelets[self.training_indices, :, :], axis=0)
        self.est_std = np.median(abs(self.wavelets[self.training_indices, :, :] - self.est_mean), axis=0)
        # Define output shape. Most robust way is to get a dummy input and take that shape as output shape
        (dummy_input, dummy_output) = self.__getitem__(0)
        # Corresponds to the output of this generator, aka input to model. Also remove batch shape,
        self.input_shape = dummy_input.shape[:]

    def set_opts_as_attribute(self, opts):
        """
        Sets all entries in opts dict to internal parameters with names equal to the
        option keys.
        """
        for k, v in opts.items():
            setattr(self, k, v)

    def get_input_sample(self, cut_range):
        # 1.) Cut Ephys / fancy indexing for memmap is planned, if fixed use: cut_data = self.wavelets[cut_range, self.fourier_frequencies, self.channels]
        cut_data = self.wavelets[cut_range, :, :]

        # 2.) Normalize input
        cut_data = (cut_data - self.est_mean) / self.est_std

        # 3.) Reshape for model input
        #cut_data = np.reshape(cut_data, (self.batch_size, self.model_timesteps, cut_data.shape[1], cut_data.shape[2]))

        # 4.) Take care of optional settings
        cut_data = np.transpose(cut_data, axes=(2, 0, 1))
        cut_data = cut_data[..., np.newaxis]

        return cut_data

    def get_output_sample(self, cut_range):
        # 1.) Cut Ephys
        out_sample = []
        for out in self.outputs:
            cut_data = out[cut_range, ...]

            # 3.) Divide evenly and make sure last output is being decoded
            if self.average_output:
                cut_data = cut_data[np.arange(0, cut_data.shape[0] + 1, self.average_output)[1::] - 1]
            out_sample.append(cut_data)

        return out_sample