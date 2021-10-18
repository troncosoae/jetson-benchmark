import torch
import numpy as np


def suit4pytorch(X, Y):
    X = np.swapaxes(X, 1, 3)
    X_norm = X/255
    X_torch = torch.from_numpy(X_norm).float()
    Y_torch = torch.from_numpy(Y).long()

    return X_torch, Y_torch
