import torch
from torch.utils.data import Dataset

class UltrasoundDataset(Dataset):
    """ Ultrasound RF Dataset """
    
    def __init__(self):
        self.in_path = "/home/peter/data/split"
        self.y = torch.load("/home/peter/data/split/aeration1.pt")
        
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        if index not in range(len(self)):
            raise IndexError(f"Invalid index: {index}, Indices must be in {range(len(self))}")
        x = torch.load(f"{self.in_path}/RF_{str(index)}.pt")
        x = x[0]
        return x, self.y[index]
        
if __name__ == "__main__":
    print("loading Ultrasound Dataset...")
    ds_ultrasound = UltrasoundDataset()
    x, y = ds_ultrasound[1000]
    print(f"x, y: {x.shape}, {y.shape}")