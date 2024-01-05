from datasets import UltrasoundDataset
from unet import Net

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

# select gpu device
device = torch.device("cuda")

# load dataset
ds = UltrasoundDataset()
model = Net().to(device)


lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.L1Loss()

train_ds, test_ds = random_split(ds, [3600, 900])
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=10, shuffle=True)

# for x, y in train_dl:
#     print(x.shape, y.shape)
#     x = x.unsqueeze(1)
#     print(x.shape)
#     # y_pred = model(x)
#     break

# train model
epochs = 10
model.train()

for epoch in tqdm(range(epochs), desc="[epoch]", leave=True):
    # for x, y in tqdm(train_dl, leave=True):
    pbar = tqdm(train_ds)
    for x, y in pbar:
        # print(x.shape, y.shape)
        x = x.unsqueeze(0).to(device)
        y = y.to(device)
        y_pred = model(x)
        
        loss = loss_fn(y_pred, y).to(device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': str(loss.item())[:6]})
torch.save(model.state_dict(), "unet_v2.pth")
    
