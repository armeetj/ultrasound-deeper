import torch
import torch.nn as nn
import torch.nn.functional as F


# mimic vgg16
class VGG16(nn.Module):
    
    def __init__(self):
        super().__init__()
        # metadata
        self.id = "vgg-16"
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(133120, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1025)
        self.relu = nn.ReLU()

    def forward(self, x):
        return nn.Sequential(
            self.conv1,
            self.conv2,
            self.pool1,
            
            self.conv3,
            self.conv4,
            self.pool2,
            
            self.conv5,
            self.conv6,
            self.pool3,
            
            self.conv7,
            self.conv8,
            self.conv9,
            self.pool4,
            
            self.conv10,
            self.conv11,
            self.conv12,
            self.pool5,
            
            self.flatten,
            self.fc1, 
            self.fc2, 
            self.fc3,
            self.relu
        )(x).unsqueeze(1)
    

if __name__ == "__main__":
    device = torch.device("cuda")
    torch.manual_seed(0)
    
    x = torch.rand([4, 1, 1024, 512]).to(device)
    y = torch.rand(4, 1, 1025).to(device)
    model = VGG16().to(device)
    res = model(x)
    print("output:", res.shape)