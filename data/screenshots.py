import glob, os
from PIL import Image
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from einops import repeat

width = 256
height = 455
size = 64
finger_size = 32
num_per_img = (width-finger_size)*(int(height/2)-finger_size)

class Screenshots(Dataset):
    def __init__(self, img_folder = 'kbd1k/keyboard_dataset/'):
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((height,width))
        ])
        self.blur = T.Compose([ #TODO: add more blur
            T.Resize((size,size))
        ])
        self.img_folder = img_folder
        self.img_names = glob.glob("%s/*.png"%(img_folder))
        self.imgs = []
        for img_name in self.img_names:
            img = Image.open(img_name)
            img = self.transform(img)
            self.imgs.append(img)
        self.num = len(self.img_names)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = self.blur(img)
        img_tensor = T.ToTensor()(img)
        return img_tensor