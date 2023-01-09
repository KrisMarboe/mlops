from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self, bb_hidden_channels, cl_hidden_channel, cl_out_channel=10, stride=3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, bb_hidden_channels[0], stride),  # [N, 64, 26]
            nn.LeakyReLU(),
            nn.Conv2d(bb_hidden_channels[0], bb_hidden_channels[1], stride),  # [N, 32, 24]
            nn.LeakyReLU(),
            nn.Conv2d(bb_hidden_channels[1], bb_hidden_channels[2], stride),  # [N, 16, 22]
            nn.LeakyReLU(),
            nn.Conv2d(bb_hidden_channels[2], bb_hidden_channels[3], stride),  # [N, 8, 20]
            nn.LeakyReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * (28 - (stride-1)*4)**2, cl_hidden_channel),
            nn.Dropout(),
            nn.Linear(cl_hidden_channel, cl_out_channel)
        )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape 1, 28, 28')
        return self.classifier(self.backbone(x))