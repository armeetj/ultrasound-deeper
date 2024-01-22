import tqdm
from nets import unet, dense, cnn
import cdatasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as du
from datetime import datetime
from colorama import Fore, Back, Style
import pickle
import os
import wandb
import argparse

seed = 53252
args = None


def get_device():
    device = "cpu"
    # torch backend device
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")

    print("Selected device:", device)
    return device


def get_formatted_time():
    return datetime.now().strftime("%m-%d-%Y_%H:%M:%S")


def save_checkpoint(model, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)
    torch.save(model.state_dict(), f"{dir}/{filename}")
    print(f"{Fore.MAGENTA}Saved checkpoint: {dir}/{filename}{Fore.RESET}")


def save_metrics(metrics, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(f"{dir}/{filename}", "wb") as file:
        pickle.dump(metrics, file)
    print(f"{Fore.MAGENTA}Saved metrics: {dir}/{filename}{Fore.RESET}")


def load_checkpoint(model, path):
    state_dict = torch.load(path)
    model.load_state_dict(state_dict)
    print(f"{Fore.MAGENTA}Loaded checkpoint: {path}{Fore.RESET}")


def acc_fn(y_pred, y):
    abs_percent_err = torch.abs((y - y_pred) / y)
    avg_percent_err = torch.mean(abs_percent_err, dim=1)
    batch_avg_percent_err = torch.mean(avg_percent_err)
    return 1 - batch_avg_percent_err


def train_model(model, device, loader, loss_fn, optimizer):
    model.train()
    with tqdm.tqdm(loader) as pbar:
        i = 0
        optimizer.zero_grad()
        for x, y in pbar:
            i += 1
            # move data to device
            x, y = x.to(device), y.to(device)

            # forward pass
            y_pred = model(x)

            # backward pass
            loss = loss_fn(y_pred, y)
            loss.backward()

            if i % 2 == 0:
                acc = acc_fn(y_pred, y)
                optimizer.step()
                optimizer.zero_grad()
                if args.wandb:
                    wandb.log({"train_loss": loss, "train_acc": acc})
                # log
                pbar.set_description(
                    f"{Fore.YELLOW}[train_loss={str(loss.item())[:7]}, train_acc={str(acc.item())[:7]}]{Fore.RESET}{Back.RESET}"
                )

    return loss.item(), acc.item()


def test_model(model, device, loader, loss_fn):
    model.eval()
    count = 0
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for x, y in loader:
            count += 1
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_acc += acc_fn(y, y_pred).item() # TODO: figure this out
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
    test_loss /= count
    test_acc /= count
    print(
        f"{Fore.RED}[test_loss={str(test_loss)[:7]}, test_acc={str(test_acc)[:7]}]{Fore.RESET}"
    )
    return test_loss, test_acc


def main(time_str, data, device, model, optimizer, loss_fn):
    # prepare dataset and loaders
    generator = torch.Generator().manual_seed(seed)
    train, test = du.random_split(data, [3600, 900], generator)
    train_loader = du.DataLoader(train, batch_size=8, shuffle=True, num_workers=12)
    test_loader = du.DataLoader(test, batch_size=8, shuffle=True, num_workers=12)

    metrics_train_loss, metrics_test_loss = list(), list()
    metrics = [metrics_train_loss, metrics_test_loss]

    print(
        f"Training started @ {time_str}:\n\tmodel={model.id}, \n\tepochs={epochs}, \n\toptimizer={optimizer}, \n\tloss_fn={loss_fn}"
    )

    for epoch in tqdm.trange(
        1, epochs + 1, desc=f"{Fore.GREEN}[epoch]{Fore.RESET}{Back.RESET}"
    ):
        train_loss, train_acc = train_model(
            model, device, train_loader, loss_fn, optimizer
        )
        test_loss, test_acc = test_model(model, device, test_loader, loss_fn)
        metrics_train_loss.append((train_loss, train_acc))
        metrics_test_loss.append((test_loss, test_acc))
        save_checkpoint(model, f"models/{model.id}_{time_str}", f"cp-{epoch}.pt")
        save_metrics(metrics, f"metrics/{model.id}_{time_str}", "metrics.pkl")
        if args.wandb:
            wandb.log(
                {
                    "train_loss_epoch": train_loss,
                    "train_acc_epoch": train_acc,
                    "test_loss_epoch": test_loss,
                    "test_acc_epoch": test_acc,
                }
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train nets in ./nets/**")

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable logging with Weights and Biases (wandb)",
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Choose acceleration backend device (cuda, mps = gpu)",
    )

    parser.add_argument(
        "-cp",
        "--checkpoint",
        default=None,
        help="path to .pth pytorch model weights dictionary"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # parse script args
    args = parse_arguments()

    # set random seed
    torch.manual_seed(seed)

    # load dataset and device
    time_str = get_formatted_time()
    data = ds.UltrasoundDataset1024x512()
    device = args.device if args.device != "auto" else get_device()

    # load model and checkpoints
    checkpoint_path = None
    if args.checkpoint != None:
        checkpoint_path = args.checkpoint
    model = unet.Net2d_2().to(device)
    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    # hyperparams
    lr = 1e-4
    epochs = 50

    # optimizer and loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss(reduction="mean")

    if args.wandb:
        wandb.init(
            project="ultrasound-trial-1d",
            config={
                "seed": seed,
                "time": time_str,
                "model_id": model.id,
                "ds_id": data.id,
                "device": device,
                "checkpoint_path": checkpoint_path,
                "lr": lr,
                "epochs": epochs,
                "optimizer": optimizer,
                "loss_fn": loss_fn,
            },
        )
    main(time_str, data, device, model, optimizer, loss_fn)
    if args.wandb:
        wandb.finish()
