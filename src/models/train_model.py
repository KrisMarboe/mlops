import argparse
import sys
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import click
import matplotlib.pyplot as plt

from model import MyAwesomeModel

class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    with open("data/processed/corruptmnist_train.pkl", "rb") as fp:
        train_images, train_labels = pickle.load(fp)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainloader = torch.utils.data.DataLoader(dataset(train_images, train_labels), batch_size=16, shuffle=True)
    epochs = 50
    best_loss = float('inf')
    losses = [0]*epochs

    model_dir = "models/day2/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for e in range(epochs):
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1).float()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses[e] += loss.item()
        else:
            losses[e] = losses[e] / len(trainloader)
            print(f"Training loss: {losses[e]}")
            if losses[e] < best_loss:
                best_loss = losses[e]
                torch.save(model.state_dict(), model_dir+'trained_model.pt')
                print(f"Saved new model at epoch {e}")
            plt.figure()
            plt.plot(losses)
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig("reports/figures/day2_model_loss.png")
            plt.close()

if __name__ == "__main__":
    train()





