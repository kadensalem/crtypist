import glob
from gymnasium import Env
from PIL import Image
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import copy
from config import KBD1K_SCREENSHOTS_DIR, KBD1K_PATH
import os.path as osp
import ast

from setting import keys_for_labeling

POSITION_FILE = osp.join(KBD1K_PATH, 'keyboard_label.csv')


class KeyboardEnv(Env, ABC):

    def __init__(self, render_mode=None, img_folder=KBD1K_SCREENSHOTS_DIR, position_file=POSITION_FILE, width=256,
                 height=455):
        self.observation_space = None
        self.action_space = None
        if "chi" in position_file:
            self.keyboard_type = 'chi'
        elif "chubon" in position_file:
            self.keyboard_type = 'chubon'
        elif "kalq" in position_file:
            self.keyboard_type = 'kalq'
        else:
            self.keyboard_type = 'normal'

        """ load keyboard images and positions """
        self.positions = None
        self.keys_for_labeling = None
        self._load_kbd_imgs(img_folder, width, height)
        self._load_kbd_positions(position_file=position_file)
        self.img_index = 0

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def reward(self, action):
        pass

    def get_observation_space(self):
        return self.observation_space.shape

    def get_action_space(self):
        return self.action_space.shape

    @staticmethod
    def _generate_initial_dataframe(keys_for_labeling):
        """
        Generate an empty dataframe
        """
        df = pd.DataFrame(columns=keys_for_labeling, index=['index'])
        return df

    def _load_kbd_imgs(self, img_folder, width, height):
        """ load keyboard images """
        self.width = width
        self.height = height
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((height, width))
        ])
        self.img_folder = img_folder
        self.img_names = sorted(glob.glob("%s/*.png" % (img_folder)))
        # self.img_names = glob.glob("%s/*.png" % (img_folder))
        self.imgs = []
        for img_name in self.img_names:
            img = Image.open(img_name)
            img = transform(img)
            self.imgs.append(img)
        self.img_index = 0
        self.screenshot = self.imgs[self.img_index]

    def _load_kbd_positions(self, position_file=POSITION_FILE):
        """ load keyboard positions """
        df = pd.read_csv(position_file, header=0, index_col=0)
        indexs = df.index
        columns = df.columns
        self.keys_for_labeling = columns.tolist()
        new_df = self._generate_initial_dataframe(columns)
        for index in indexs:
            for column in columns:
                info = df.loc[index, column]
                if isinstance(info, str) and '[' in info:
                    nested_list = ast.literal_eval(info)
                    new_df.loc[index, column] = nested_list
                else:
                    new_df.loc[index, column] = info
        self.position_df = new_df
        filename = self.img_names[self.img_index].split('/')[-1]
        keyboard_index = filename.split('_')[0]
        self.positions = self._get_place_positions(keyboard_index, self.keys_for_labeling)

    def _reset_kbd(self):
        if self.img_index == len(self.imgs) - 1:
            self.img_index = 0
        else:
            self.img_index += 1
        self.screenshot = self.imgs[self.img_index]
        filename = self.img_names[self.img_index].split('/')[-1]
        keyboard_index = filename.split('_')[0]
        self.positions = self._get_place_positions(keyboard_index, self.keys_for_labeling)

    def _get_center(self, place):
        x1, y1, x2, y2 = self.positions[place]
        return np.array([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])

    def _get_place_positions(self, keyboard_index, places, number_row=False):
        # TODO: for current stage the agent is trained without number row, just use one screenshot without number row
        keyboard_index = int(keyboard_index)
        target_df = self.position_df[self.position_df['keyboard_index'] == keyboard_index]
        # current version does not support punctuation
        target_df = target_df[target_df['mode'] != 'punctuation']
        # target_df = target_df[target_df['number_row'] == number_row]
        screenshot_indexs = target_df.index[0]
        positions = copy.deepcopy(self.position_df.loc[screenshot_indexs].to_dict())
        if '<' not in positions.keys() and 'backspace' in positions.keys():
            positions['<'] = positions['backspace']
            positions['>'] = positions['enter']
            positions[' '] = positions['space']
        if self.keyboard_type == 'kalq':
            positions['  '] = positions['space2']
        for place in places:
            if not isinstance(positions[place], list):
                continue

            positions[place][0] = int(positions[place][0] * self.width / 1080)
            positions[place][1] = int(positions[place][1] * self.height / 1920)
            positions[place][2] = int(positions[place][2] * self.width / 1080)
            positions[place][3] = int(positions[place][3] * self.height / 1920)
        return positions

    def _distance(self, point_A, point_B):
        return np.linalg.norm(point_A - point_B)

    def _gaze_on(self, place):
        center = self._get_center(place)
        if place == 'input_box':
            center[0] = center[0] / 2
        key_in_gaze = self._distance(center, self.gaze) <= self.gaze_size / 2

        x1, y1, x2, y2 = self.positions[place]
        if place == 'input_box':
            x2 /= 2
        gaze_in_key = (x1 <= self.gaze[0] <= x2 and y1 <= self.gaze[1] <= y2)

        return key_in_gaze or gaze_in_key

    def _finger_on(self, place):
        if place not in self.positions:
            return False
        x = self.finger[0]
        y = self.finger[1]
        x1,y1,x2,y2 = self.positions[place]
        return x1 <= x < x2 and y1 <= y < y2

    def _gaze_on_finger(self):
        x = self.finger[0]
        y = self.finger[1]
        gaze_x1 = self.gaze[0] - self.gaze_size / 2
        gaze_y1 = self.gaze[1] - self.gaze_size / 2
        gaze_x2 = self.gaze[0] + self.gaze_size / 2
        gaze_y2 = self.gaze[1] + self.gaze_size / 2
        return gaze_x1 <= x < gaze_x2 and gaze_y1 <= y < gaze_y2

    def _where(self):
        for key in self.keys:
            if self._finger_on(key):
                return key
        return None