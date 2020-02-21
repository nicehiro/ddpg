import numpy as np


def feature_normalize(data):
    mu = data.mean(axis=1).unsqueeze(1)
    std = data.std(axis=1).unsqueeze(1)
    return (data - mu) / std