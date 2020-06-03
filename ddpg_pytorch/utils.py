import numpy as np


def feature_normalize(data):
    mu = np.expand_dims(data.mean(axis=1), axis=1)
    std = np.expand_dims(data.std(axis=1), axis=1)
    return (data - mu) / std