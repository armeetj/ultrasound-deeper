import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import glob


class Raw_BFRF(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.id = "raw-bfrf"
        self.in_path = "/home/peter/data/bfrf"
        # self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return len(glob.glob(f"{self.in_path}/**.pt"))

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/{index}.pt").unsqueeze(0),
            torch.rand(1)
            # self.y[index],
        )
        return x, y


class Raw_Map2d(Dataset):
    """Ultrasound RF Dataset"""

    def __init__(self):
        self.id = "raw-bfrf"
        self.in_path = "/home/peter/data/2dmap"
        # self.y = torch.load("/home/peter/data/split/aeration1.pt") / 100.0

    def __len__(self):
        return len(glob.glob(f"{self.in_path}/**.pt"))

    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(
                f"Invalid index: {index}, Indices must be in {range(len(self))}"
            )
        x, y = (
            torch.load(f"{self.in_path}/{index}.pt").unsqueeze(0),
            torch.rand(1)
            # self.y[index],
        )
        return x, y


if __name__ == "__main__":
    ds = Raw_Map2d()
    x, y = ds[1000]
    print(f"x, y: {x.shape}, {y.shape}")
