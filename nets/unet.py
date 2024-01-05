import torch
import torch.nn as nn
import torch.nn.functional as F


# verbose = False

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding="same")
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.relu(x)
        return x
    
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        skip = self.conv(x)
        signal = self.pool(skip)
        return skip, signal
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(in_c, out_c)

    def forward(self, x, s):
        x = self.upconv(x)
        x = torch.cat([x, s], axis=1)
        x = self.conv(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # encoders
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        
        # bottleneck
        self.b = conv_block(128, 256)
        
        # decoders
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        
        # output
        self.o1 = nn.Conv2d(64, 1, kernel_size=1)
        self.o2 = nn.AvgPool2d(2, stride=2)
        self.o3 = nn.AvgPool1d(2, stride=2)
    
    def forward(self, x):
        # encoder blocks
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        
        # bottleneck
        b = self.b(p2)
        
        # decoder blocks
        u1 = self.d1(b, s2)
        u2 = self.d2(u1, s1)
        
        # output
        out = self.o1(u2)
        out = self.o2(out)
        out = out.flatten(2)
        out = self.o3(out)
        out = out.squeeze()
        out = F.pad(out, (0, 1), "constant", 0)
        
        # out = u1
        return out
def build():
    return Net()

if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.rand([10, 1, 64, 128]).to(device)
    model = Net().to(device)
    res = model(x)
    print("output:", res.shape)