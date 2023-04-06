import torch.nn as nn
import torch.nn.functional as F
import torchvision

class InvertedBlock(nn.Module):
    def __init__(self, squeeze=16, expand = 64):
        super(InvertedBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace = True),

            # Depthwise convolution
            nn.Conv2d(expand, expand, kernel_size=3, stride=1, padding=1, groups=expand, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True),

            # Pointwise Convolution + Linear projection
            nn.Conv2d(expand, squeeze, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)

class Fermini(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()

        def conv_bn(inp, oup,  ks):
            return nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=ks, padding=1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace = True)
            )
        
        def invert(squeeze, expand):
            return InvertedBlock(squeeze, expand)

        self.layer1 = conv_bn(1, 64, 3)
        self.layer2 = conv_bn(64, 128, 5)
        self.layer3 = invert(128, 256)
        self.layer4 = invert(128, 256)
        self.layer5 = invert(128, 512)
        self.layer6 = invert(128, 512)
        self.lin1 = nn.Linear(128*2*2, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.pool(x)

        x = F.relu(self.layer2(x))
        x = self.pool(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.pool(x)

        print(x.shape)

        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x
        