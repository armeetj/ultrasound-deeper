from nets import unet
import cdatasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du

def main():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("selected device:", device)

    # dataset
    data = ds.UltrasoundDataset()
    model = unet.Net2().to(device)
    model.load_state_dict(torch.load("models/unet2_01-04-2024_21:39:28.pt"))
    loss_fn = nn.L1Loss()
    loader = du.DataLoader(data, batch_size=100, shuffle=True, num_workers=12)
    
    # load dataset
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    # x = x.unsqueeze(0)
    # y = y.unsqueeze(0)
    print(x.shape, y.shape)
    y_pred = model(x)
    print(y_pred.shape)
    print(loss_fn(y_pred, y).item())

if __name__ == "__main__":
    main()