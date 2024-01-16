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
        self.upconv = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0
        )
        self.conv = conv_block(in_c, out_c)

    def forward(self, x, s):
        x = self.upconv(x)
        x = torch.cat([x, s], axis=1)
        x = self.conv(x)
        return x


class Net1(nn.Module):
    def __init__(self):
        super().__init__()

        # metadata
        self.id = "unet2"

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
        out = out.flatten(1)
        out = F.pad(out, (0, 1), "constant", 0)

        return out


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet2"

        # encoders
        self.e1 = encoder_block(1, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)

        # bottleneck
        self.b = conv_block(256, 512)

        # decoders
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        # output
        self.o1 = nn.Conv2d(64, 1, kernel_size=1)
        self.o2 = nn.AvgPool2d(2, stride=2)
        self.o3 = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        # encoder blocks
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        # bottleneck
        b = self.b(p3)
        # decoder blocks
        u1 = self.d1(b, s3)
        u2 = self.d2(u1, s2)
        u3 = self.d3(u2, s1)

        # output
        out = self.o1(u3)
        out = self.o2(out)
        out = out.flatten(2)
        out = self.o3(out)
        out = out.flatten(1)
        out = F.pad(out, (0, 1), "constant", 0)

        return out


class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet4"

        # encoders
        self.e1 = encoder_block(1, 16)

        # bottleneck
        self.b = conv_block(16, 32)

        # decoders
        self.d1 = decoder_block(32, 16)

        # output
        self.o1 = nn.Conv2d(16, 1, kernel_size=1)
        self.o2 = nn.AvgPool2d(2, stride=2)
        self.o3 = nn.AvgPool1d(2, stride=2)

    def forward(self, x):
        # encoder blocks
        s1, p1 = self.e1(x)

        # bottleneck
        b = self.b(p1)

        # decoder blocks
        u1 = self.d1(b, s1)

        # output
        out = self.o1(u1)
        out = self.o2(out)
        out = out.flatten(2)
        out = self.o3(out)
        out = out.flatten(1)
        out = F.pad(out, (0, 1), "constant", 0)

        return out


class Net2d_1(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet2d-1"

        # encoders
        self.enc1 = encoder_block(1, 64)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.enc4 = encoder_block(256, 512)

        # bottleneck
        self.bottleneck = conv_block(512, 1024)

        # decoders
        self.dec1 = decoder_block(1024, 512)
        self.dec2 = decoder_block(512, 256)
        self.dec3 = decoder_block(256, 128)
        self.dec4 = decoder_block(128, 64)

        # output
        self.o1 = nn.Conv2d(64, 1, kernel_size=1)
        self.o2 = nn.AvgPool2d(4, stride=4)
        self.o3 = nn.AvgPool2d(4, stride=4)
        # self.o4 = nn.AvgPool2d(2, stride=2)

        # flatten and activation
        self.flatten = nn.Flatten(2)
        self.linear = nn.Linear(1024, 1025)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # down
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        # bottle
        b = self.bottleneck(p4)

        # up
        u1 = self.dec1(b, s4)
        u2 = self.dec2(u1, s3)
        u3 = self.dec3(u2, s2)
        u4 = self.dec4(u3, s1)

        out = self.o1(u4)
        out = self.o2(out)
        out = self.o3(out)
        # out = self.o4(out)

        out = self.flatten(out)
        out = self.linear(out)
        out = self.activate(out)

        return out
    
class Net2d_2(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet2d-2"

        # encoders
        self.enc1 = encoder_block(1, 64)
        self.enc2 = encoder_block(64, 128)
        self.enc3 = encoder_block(128, 256)
        self.enc4 = encoder_block(256, 512)

        # bottleneck
        self.bottleneck = conv_block(512, 1024)

        # decoders
        self.dec1 = decoder_block(1024, 512)
        self.dec2 = decoder_block(512, 256)
        self.dec3 = decoder_block(256, 128)
        self.dec4 = decoder_block(128, 64)

        # output
        self.o1 = nn.Conv2d(64, 1, kernel_size=1)
        self.o2 = nn.AvgPool2d(4, stride=4)
        self.o3 = nn.AvgPool2d(4, stride=4)

        # flatten and activation
        self.flatten = nn.Flatten(2)
        self.linear = nn.Linear(2048, 1025)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        # down
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        # bottle
        b = self.bottleneck(p4)

        # up
        u1 = self.dec1(b, s4)
        u2 = self.dec2(u1, s3)
        u3 = self.dec3(u2, s2)
        u4 = self.dec4(u3, s1)

        out = self.o1(u4)
        out = self.o2(out)
        out = self.o3(out)

        out = self.flatten(out)
        out = self.linear(out)
        out = self.activate(out)

        return out


class conv3d_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = nn.Conv3d(in_c, out_c // 2, kernel_size=3, padding="same")
        self.c2 = nn.Conv3d(out_c // 2, out_c, kernel_size=3, padding="same")
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.c1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.c2(x)
        # x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder3d_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv3d_block(in_c, out_c)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        signal = self.pool(skip)
        return skip, signal


class decoder3d_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(in_c, in_c, kernel_size=2, stride=2)
        self.conv = conv3d_block(in_c + out_c, out_c)

    def forward(self, x, s):
        x = self.upconv(x)
        x = torch.cat([x, s], axis=1)
        x = self.conv(x)
        return x


class Net3d_1(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet-3d-1"

        self.avg = nn.AvgPool3d(kernel_size=(4, 1, 1))

        # encoders
        self.e1 = encoder3d_block(1, 64)
        self.e2 = encoder3d_block(64, 128)

        # bottleneck
        self.b = conv3d_block(128, 256)

        # decoders
        self.d1 = decoder3d_block(256, 128)
        self.d2 = decoder3d_block(128, 64)

        # output
        self.o1 = nn.Conv3d(64, 1, kernel_size=4, stride=4)
        self.o2 = nn.AvgPool3d(kernel_size=(8, 2, 2))
        self.o3 = nn.Flatten(start_dim=2)
        self.o4 = nn.Linear(1024, 1025)

    def forward(self, x):
        x = self.avg(x)

        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)

        b = self.b(p2)

        u1 = self.d1(b, s2)
        u2 = self.d2(u1, s1)

        out = self.o1(u2)
        out = F.relu(out)
        out = self.o2(out)
        out = F.relu(out)
        out = self.o3(out).squeeze(0)
        out = F.relu(out)
        out = self.o4(out)
        out = out
        return out


class Net3d_2(nn.Module):
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "unet-3d-2"

        self.avg = nn.AvgPool3d(kernel_size=(4, 1, 1))

        # encoders
        self.e1 = encoder3d_block(1, 64)
        self.e2 = encoder3d_block(64, 128)

        # bottleneck
        self.b = conv3d_block(128, 256)

        # decoders
        self.d1 = decoder3d_block(256, 128)
        self.d2 = decoder3d_block(128, 64)

        # output
        self.o1 = nn.Conv3d(64, 1, kernel_size=4, stride=4)
        self.o2 = nn.AvgPool3d(kernel_size=(8, 2, 2))
        self.o3 = nn.Flatten(start_dim=2)
        self.o4 = nn.Linear(1408, 1025)

    def forward(self, x):
        x = self.avg(x)

        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)

        b = self.b(p2)

        u1 = self.d1(b, s2)
        s1 = s1[:, :, :376, :, :]
        u2 = self.d2(u1, s1)

        out = self.o1(u2)
        out = F.relu(out)
        out = self.o2(out)
        out = F.relu(out)
        out = self.o3(out).squeeze(0)
        out = F.relu(out)
        out = self.o4(out)
        out = F.relu(out)
        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    torch.manual_seed(0)
    
    """
    2d unets:
    batch_size=16
    num_channels=1
    (64 x 128)
    """
    # x = torch.rand([4, 1, 1024, 512]).to(device)
    # y = torch.rand(4, 1, 1025).to(device)
    # model = Net2d_2().to(device)
    # res = model(x)
    # print("output:", res.shape)
    # print(y.shape, res.shape)
    # print(F.l1_loss(y, res))

    """ 
    3d unets:
    batch_size=10
    num_channels=1
    (1024 x 64 x 128)
    """
    x = torch.rand([1, 1, 1024, 64, 128]).to(device)
    model = Net3d_1().to(device)
    res = model(x)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(trainable_params)

    # summary(model)
    # print("output:", res.shape)
