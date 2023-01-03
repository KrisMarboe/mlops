import torch
import numpy as np

def mnist():
    # exchange with the corrupted mnist dataset
    train = []
    for i in range(5):
        train_i = np.load(f"../../../data/corruptmnist/train_{i}.npz", allow_pickle=True)
        train += list(zip(train_i['images'], train_i['labels']))

    test = np.load("../../../data/corruptmnist/test.npz", allow_pickle=True)
    return train, list(zip(test['images'], test['labels']))
