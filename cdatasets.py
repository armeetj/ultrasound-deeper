import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class UltrasoundDataset(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.in_path = "/home/peter/data/split"
        self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/RF_{str(index)}.pt")[:1024, :, :],
            # torch.load(f"{self.in_path}/RF_{str(index)}.pt"),
            self.y[index],
        )
        x = x.unsqueeze(0)
        # x = F.normalize(x)
        return x, y


class RawNormUltrasoundDataset(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.id = "raw-1509-norm"
        self.in_path = "/home/peter/data/split"
        self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/RF_{str(index)}.pt") / 30000.0,
            self.y[index],
        )
        x = x.unsqueeze(0)
        # x = F.normalize(x)
        return x, y


class ReducedUltrasoundDataset(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.id = "512x512-norm"
        self.in_path = "/home/peter/data/split"
        self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/RF_{str(index)}.pt")[:1024, :, :] / 30000.0,
            self.y[index],
        )
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        x = F.avg_pool3d(x, (2, 4, 4))
        x = x.flatten(2)
        return x, y


class CutUltrasoundDataset(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.id = "cut-t1024-ynorm"
        self.in_path = "/home/peter/data/split"
        self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/RF_{str(index)}.pt")[200:1224, :, :],
            # torch.load(f"{self.in_path}/RF_{str(index)}.pt"),
            self.y[index],
        )
        x = x.unsqueeze(0)
        # x = F.normalize(x)
        return x, y


if __name__ == "__main__":
    print("loading Ultrasound Dataset...")
    ds_ultrasound = ReducedUltrasoundDataset()
    x, y = ds_ultrasound[1000]
    print(f"x, y: {x.shape}, {y.shape}")
