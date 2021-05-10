"""
This script is for generally having a look around the matt jones data.
"""
import scipy.io
# Okay first let's figure out how to open the VT1 data
#mat = scipy.io.loadmat('data/MJ.mat')
from deep_insight.wavelet_dataset import WaveletDataset
import h5py
from deep_insight.options import get_opts
import numpy as np
#from deep_insight.wavelet_transform import wavelet_transform
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt

#H5_PATH = 'data/preprocessed_MJ2.h5'
H5_PATH = 'data/preprocessed_MJ3.h5'
OLD_H5_PATH = './data/processed_R2478.h5'

training_options = get_opts(H5_PATH, train_test_times=(np.array([]), np.array([])))

hdf5_file = h5py.File(H5_PATH, mode='r')
old_hdf5_file = h5py.File(OLD_H5_PATH, mode='r')

new_inputs = hdf5_file["inputs/wavelets"]#
old_inputs = old_hdf5_file["inputs/wavelets"]

print(0)
