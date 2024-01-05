import tqdm
from nets import unet
import datasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du
from datetime import datetime
from colorama import Fore, Back, Style

def train_model(model, device, loader, loss_fn, optimizer):
    model.train()
    # return
    with tqdm.tqdm(loader) as pbar:
        for x, y in pbar:
            # move data to device
            x, y = x.to(device), y.to(device)

            # forward pass
            y_pred = model(x)

            # backward pass
            optimizer.zero_grad()
            loss = loss_fn(y, y_pred)
            loss.backward()
            optimizer.step()

            # log
            pbar.set_description(f"{Fore.YELLOW}[loss={str(loss.item())[:7]}]{Fore.RESET}{Back.RESET}")


def test_model(model, device, loader, loss_fn):
    # return
    model.eval()
    count = 1
    test_loss = 0
    with torch.no_grad():
        for x, y in loader:
            count += 1
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            test_loss += loss.item()
    print(f"{Fore.RED}[test_loss={str(test_loss / count)[:7]}]{Fore.RESET}")


def main():
    # torch backend device
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
    train, test = du.random_split(data, [3600, 900])
    train_loader = du.DataLoader(train, batch_size=100, shuffle=True, num_workers=12)
    test_loader = du.DataLoader(test, batch_size=100, shuffle=True, num_workers=12)

    # build model
    lr = 1e-3
    model = unet.Net1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    epochs = 20

    for epoch in tqdm.trange(1, epochs + 1, desc=f"{Fore.GREEN}[epoch]{Fore.RESET}{Back.RESET}"):
        train_model(model, device, train_loader, loss_fn, optimizer)
        test_model(model, device, test_loader, loss_fn)

    curr_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    torch.save(model.state_dict(), f"models/unet_{curr_time}.pt")


if __name__ == "__main__":
    main()
