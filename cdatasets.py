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
    ds_ultrasound = CutUltrasoundDataset()
    x, y = ds_ultrasound[1000]
    print(f"x, y: {x.shape}, {y.shape}")
