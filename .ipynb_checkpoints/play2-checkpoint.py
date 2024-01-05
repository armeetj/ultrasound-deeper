import torch
from datasets import UltrasoundDataset
from unet import unet_gh

if __name__ == "__main__":
    # load dataset
    ds = UltrasoundDataset()
    model = unet_gh.build_unet()
    
    x, y = ds[0]
    print(x.shape, y.shape) # [1509, 64, 128], [1025]
    
    x_split = torch.chunk(x, x.shape[0], dim=0)
    print(x_split[0].shape) # [1, 64, 128]
    # print(model(x_split[0]))