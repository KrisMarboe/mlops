import argparse
import os
import pickle
import random
import sys

import click
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.models.model import MyAwesomeModel


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@click.command()
@click.argument("model_checkpoint")
@click.argument("train_data")
def visualize(model_checkpoint, train_data):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    with open(train_data, "rb") as fp:
        train_images, train_labels = pickle.load(fp)
    with torch.no_grad():
        j = random.randint(0, len(train_images))
        embedding = model.backbone(train_images[j].reshape((1, 28, 28)).float())
        plt.figure(figsize=(6, 6))
        for i in range(embedding.shape[0]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(embedding[i])
        plt.subplot(3, 3, 9)
        plt.imshow(train_images[j])
        plt.tight_layout()
        plt.savefig("reports/figures/embedding.png")


if __name__ == "__main__":
    visualize()
