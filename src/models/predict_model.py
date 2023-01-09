import pickle

import click
import torch
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


@click.command()
@click.argument("model_checkpoint")
@click.argument("test_data")
def evaluate(model_checkpoint, test_data):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    state_dict = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    model.load_state_dict(state_dict)
    with open(test_data, "rb") as fp:
        test_images, test_labels = pickle.load(fp)
    testloader = torch.utils.data.DataLoader(
        dataset(test_images, test_labels), batch_size=16, shuffle=True
    )
    with torch.no_grad():
        # validation pass here
        model.eval()
        for images, labels in testloader:
            # images = images.view(images.shape[0], -1).float()
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f"Accuracy: {accuracy.item() * 100}%")


if __name__ == "__main__":
    evaluate()
