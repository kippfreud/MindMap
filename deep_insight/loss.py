"""
Custom losses for training
"""

import torch
import numpy as np

def euclidean_loss(y_true, y_pred):
    res = torch.sqrt(torch.sum(torch.square(y_pred - y_true), axis=-1))
    return res

def cyclical_mae_rad(y_true, y_pred):
    ret = torch.mean(torch.min(torch.abs(y_pred - y_true),
                                torch.min(torch.abs(y_pred - y_true + 2*np.pi),
                                          torch.abs(y_pred - y_true - 2*np.pi))),
                      axis=-1)
    return ret

mse = torch.nn.MSELoss

l1 = torch.nn.L1Loss(reduction='none')
def mae(y_true, y_pred):
    ret = torch.squeeze(l1(y_true, y_pred))
    return ret
