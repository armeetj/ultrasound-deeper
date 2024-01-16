import torch
import torch.nn as nn


class SinTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = "sin-test"
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return nn.Sequential(self.fc1, self.fc2, self.fc3, self.relu)(x)


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = "c2"
        self.c1 = nn.Conv3d(1, 16, kernel_size=2, stride=2)
        self.c2 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.mp = nn.MaxPool3d(kernel_size=3, stride=3)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(6176768, 1025)
        # self.fc2 = nn.Linear(5000, 1025)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.relu(x)

        x = self.fl(x)
        x = self.fc1(x)
        out = x
        return out


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = "c1"
        self.c1 = nn.Conv3d(1, 16, kernel_size=2, stride=2)
        self.c2 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.mp = nn.MaxPool3d(kernel_size=3, stride=3)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(3936, 1025)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.mp(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.mp(x)
        x = self.relu(x)

        x = self.fl(x)
        x = self.fc1(x)
        out = x
        return out


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = "c2"
        self.c1 = nn.Conv3d(1, 16, kernel_size=2, stride=2)
        self.c2 = nn.Conv3d(16, 32, kernel_size=2, stride=2)
        self.mp = nn.MaxPool3d(kernel_size=3, stride=3)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(6176768, 1025)
        # self.fc2 = nn.Linear(5000, 1025)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)

        x = self.c2(x)
        x = self.relu(x)

        x = self.fl(x)
        x = self.fc1(x)
        out = x
        return out


class Net3(nn.Module):
    def __init__(self):
        self.id = "conv1"
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool3d(kernel_size=4, stride=4)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5888, 1025)
        self.fc2 = nn.Linear(1025, 512)
        self.fc3 = nn.Linear(512, 1025)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.relu(self.fc3(x))
        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.rand([5, 1, 1509, 64, 128]).to(device)
    model = Net3().to(device)
    res = model(x)
    print("output:", res.shape)
