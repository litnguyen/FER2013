import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import cv2

class Fer2013(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.imgs, self.labels = self.readImg()
        self.dict_label = {'angry':0, 'disgust':1, 'fear':2, 'happy':3, 'neutral':4, 'sad':5, 'surprise':6}
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.imgs[idx]
        img = torch.from_numpy(img)
        label = self.labels[idx]
        label = self.dict_label[label]
        label = torch.tensor(label).type(torch.long)
        return (img, label)

    def readImg(self):
        labels = os.listdir(self.path)
        list_labels = []
        list_imgs = []
        for label in labels:
            imgs_path = os.path.join(self.path, label)
            for img in os.listdir(imgs_path):
                img_path = os.path.join(imgs_path,img)
                list_imgs.append(cv2.resize(cv2.imread(img_path, 0),(48,48)))
                list_labels.append(label)
        return list_imgs, list_labels

def get_dataloader(path = 'dataset', bs = 64, augment = True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """
    