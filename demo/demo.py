import torch.nn as nn
import torch.nn.functional as F
import torchvision

import os
import torch
import sys
sys.path.append('../')
from utils.logger import Logger
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image



class InvertedBlock(nn.Module):
    def __init__(self, squeeze=16, expand=64):

        super(InvertedBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True),
            # Depthwise Convolution
            nn.Conv2d(expand, expand, kernel_size=3, stride=1, padding=1, groups=expand, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True),
            # Pointwise Convolution + Linear projection
            nn.Conv2d(expand, squeeze, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return x + self.conv(x)


class VggFeatures(nn.Module):
    def __init__(self, drop=0.2):
        super().__init__()

        def conv_bn(inp, oup, ks):
            return nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=ks, padding=1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
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
        # print(x.shape)

        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        return x


class Vgg(VggFeatures):
    def __init__(self, drop=0.2):
        super().__init__(drop)
        self.lin3 = nn.Linear(128, 7)

    def forward(self, x):
        x = super().forward(x)
        x = self.lin3(x)
        return x

''''''



def video_capture():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    mu,st = 0,255
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,))
        ])

    lb = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise", 6: "neutral"}
    def check(gray_face):
        img = cv2.resize(gray_face, (40,40)).astype(np.float64)
        img = Image.fromarray(img)
        img = test_transform(img)
        img.unsqueeze_(0)
        outputs = net(img)
        _, preds = torch.max(outputs.data, 1)
        return int(preds.data[0])

    cap = cv2.VideoCapture(0)
    while True:
            # Read the frame
        _, img = cap.read()
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            face = gray[y:y+h,x:x+w]
            a = check(face)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img = cv2.putText(img, lb[a], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 1, cv2.LINE_AA)
        # Display
        cv2.imshow('img', img)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    net = Vgg()
    #net = net.eval()
    epoch = 158
    path = os.path.join('cp_demo', 'epoch_' + str(epoch))
    logger = Logger()
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    logger.restore_logs(checkpoint['logs'])
    net.load_state_dict(checkpoint['params'])
    print("Network Restored!")
    video_capture()