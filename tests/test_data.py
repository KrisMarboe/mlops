import os
import pickle

import pytest
import torch
from torch.utils.data import Dataset

from tests import _PATH_DATA


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


class TestClass:
    N_train = 40000
    N_test = 5000

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(_PATH_DATA, "processed/corruptmnist_train.pkl")
        ),
        reason="Data files not found",
    )
    def test_train_data(self):
        with open(
            os.path.join(_PATH_DATA, "processed/corruptmnist_train.pkl"), "rb"
        ) as fp:
            images, labels = pickle.load(fp)

        ds = dataset(images, labels)

        assert (
            len(ds.data) == self.N_train
        ), "Training data did not have the correct number of samples"
        assert ds.data.shape == (
            self.N_train,
            1,
            28,
            28,
        ), "Training shape should be (B, 1, 28, 28)"
        assert len(ds.data) == len(
            ds.labels
        ), "Should have an equal amount of data and labels in training"
        assert (
            len(ds.labels.unique()) == 10
        ), "All 10 classes should be present in training"

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(_PATH_DATA, "processed/corruptmnist_test.pkl")),
        reason="Data files not found",
    )
    def test_test_data(self):
        with open(
            os.path.join(_PATH_DATA, "processed/corruptmnist_test.pkl"), "rb"
        ) as fp:
            images, labels = pickle.load(fp)

        ds = dataset(images, labels)

        assert (
            len(ds.data) == self.N_test
        ), "Test data did not have the correct number of samples"
        assert ds.data.shape == (
            self.N_test,
            1,
            28,
            28,
        ), "Test shape should be (B, 1, 28, 28)"
        assert len(ds.data) == len(
            ds.labels
        ), "Should have an equal amount of data and labels in testing"
        assert (
            len(ds.labels.unique()) == 10
        ), "All 10 classes should be present in testing"
