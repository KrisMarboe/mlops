import argparse
import sys

import torch
import torch.nn as nn
import click

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
    epochs = 50
    best_loss = float('inf')
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1).float()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss / len(trainloader)}")
            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(model.state_dict(), 'trained_model.pt')
                print(f"Saved new model at epoch {e}")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    _, test_set = mnist()
    testloader = torch.utils.data.DataLoader(test_set, batch_size=16, shuffle=True)
    with torch.no_grad():
        # validation pass here
        model.eval()
        for images, labels in testloader:
            images = images.view(images.shape[0], -1).float()
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item() * 100}%')

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    