
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class VggFeature(nn.Module):
    def __init__(self, drop = 0.2):
        super().__init__()

        # Convolution
        self.conv1a = nn.Conv2d(in_channels=1, out_channels = 32, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(in_channels=32, out_channels = 32, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(in_channels=32, out_channels = 64, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(in_channels=64, out_channels = 64, kernel_size=3, padding=1)

        self.conv3a = nn.Conv2d(in_channels=64, out_channels = 128, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(in_channels=128, out_channels = 128, kernel_size=3, padding=1)

        self.conv4a = nn.Conv2d(in_channels=128, out_channels = 256, kernel_size=3, padding=1)
        self.conv4b = nn.Conv2d(in_channels=256, out_channels = 256, kernel_size=3, padding=1)

        # Pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Batch Normalization
        self.bn1a = nn.BatchNorm2d(32)
        self.bn1b = nn.BatchNorm2d(32)

        self.bn2a = nn.BatchNorm2d(64)
        self.bn2b = nn.BatchNorm2d(64)

        self.bn3a = nn.BatchNorm2d(128)
        self.bn3b = nn.BatchNorm2d(128)

        self.bn4a = nn.BatchNorm2d(256)
        self.bn4b = nn.BatchNorm2d(256)

        # Flatten
        self.lin1 = nn.Linear(256 * 2 * 2, 2048)
        
        # Dropout
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.drop(self.lin1(x)))

        return x