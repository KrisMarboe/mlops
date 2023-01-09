import os
import pickle

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from model import MyAwesomeModel
from torch.utils.data import Dataset


class dataset(Dataset):
    def __init__(self, images, labels):
        self.data = images.view(-1, 1, 28, 28)
        self.labels = labels

    def __getitem__(self, item):
        return self.data[item].float(), self.labels[item]

    def __len__(self):
        return len(self.data)


@hydra.main(config_path="conf", config_name="config.yaml")
def train(cfg):
    print("Training day and night")
    m_hparams = cfg.model
    t_hparams = cfg.training

    torch.manual_seed(t_hparams.hyperparameters.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Running on device: {device}")

    model = MyAwesomeModel(
        m_hparams.hyperparameters.bb_hidden_channels,
        m_hparams.hyperparameters.cl_hidden_channels,
        m_hparams.hyperparameters.cl_out_channel,
        m_hparams.hyperparameters.stride,
    )
    model = model.to(device)
    with open(t_hparams.hyperparameters.train_file, "rb") as fp:
        train_images, train_labels = pickle.load(fp)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=t_hparams.hyperparameters.lr)
    trainloader = torch.utils.data.DataLoader(
        dataset(train_images, train_labels),
        batch_size=t_hparams.hyperparameters.batch_size,
        shuffle=True,
    )
    epochs = t_hparams.hyperparameters.epochs
    best_loss = float("inf")
    losses = [0] * epochs

    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    for e in range(epochs):
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            # images = images.view(images.shape[0], -1).float()
            optimizer.zero_grad()
            output = model(images.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            losses[e] += loss.item()
        else:
            losses[e] = losses[e] / len(trainloader)
            print(f"Training loss: {losses[e]}")
            if losses[e] < best_loss:
                best_loss = losses[e]
                torch.save(
                    model.state_dict(), "trained_model.pt"
                )  # (model.state_dict(), model_dir+'trained_model.pt')
                print(f"Saved new model at epoch {e}")
            plt.figure()
            plt.plot(losses)
            plt.title("Loss")
            plt.xlabel("Epoch")
            plt.tight_layout()
            plt.savefig("model_loss.png")  # ("reports/figures/day2_model_loss.png")
            plt.close()


if __name__ == "__main__":
    train()
