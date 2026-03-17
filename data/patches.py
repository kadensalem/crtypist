import glob, os
from PIL import Image
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
from einops import repeat
from config import KBD1K_SCREENSHOTS_DIR

width = 256
height = 455
patch_size = 64


class Patches(Dataset):
    def __init__(self, number_row=False, punctuation=False, img_folder=KBD1K_SCREENSHOTS_DIR, screenshot_num=1070):
        """
        :param number_row:
        :param punctuation:
        :param img_folder:
        :param screenshot_num: number of screenshot used, if the number is less than 1070, the last one will be used
        """
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((height, width)),
            # T.ToTensor(),
        ])
        self.img_folder = img_folder
        self.img_names = glob.glob("%s/*.png" % (img_folder))
        self.imgs = []
        count = 0
        for img_name in self.img_names:
            if not number_row:
                if img_name.split('/')[-1].split('.')[0].split('_')[-1] == "2" or \
                        img_name.split('/')[-1].split('.')[0].split('_')[-1] == "3":
                    continue
            if not punctuation:
                if img_name.split('/')[-1].split('.')[0].split('_')[-1] == "4":
                    continue
            # get rid of screenshot with text
            if img_name.split('/')[-1].split('.')[0].split('_')[1] == "1":
                continue
            img = Image.open(img_name)
            img = self.transform(img)
            self.imgs.append(img)
            count += 1
            if count == screenshot_num:
                break
        print("loading {} screenshots".format(len(self.imgs)))
        self.num = (width - patch_size) * (height - patch_size - 140) * len(
            self.imgs)  # skip empty patches (140 pixels)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        # img_path = random.choice(self.img_names)
        # img = Image.open(img_path)
        # img = self.transform(img)
        # top_skip = 267
        # bottom_skip = 30
        skip_hight = 140
        num_per_img = (width - patch_size) * (height - patch_size - skip_hight)
        # img_idx = int(idx / num_per_img)
        img_idx = 0
        img = self.imgs[img_idx]
        idx = idx % num_per_img
        x = idx % (width - patch_size) + int(patch_size / 2)
        y = int(idx / (width - patch_size) + int(patch_size / 2))
        if y > 70: y += skip_hight
        left, top, right, bottom = x - 32, y - 32, x + 32, y + 32
        img = img.crop((left, top, right, bottom))
        img_tensor = T.ToTensor()(img)
        # return img_tensor, np.array([x, y])
        return img_tensor