import os
import random
import logging
import torch
import numpy as np
import pandas as pd
from gym import spaces
import pygame
from PIL import Image
import torchvision.transforms as transforms
from typing_env.kbd_env import KeyboardEnv
from models.vision_encoder import VisionEncoder
from setting import PLACES, PLACES_FIN

class VisionEnv(KeyboardEnv):
    def __init__(self, render_mode=None, img_folder='kbd1k/keyboard_dataset/', position_file='kbd1k/keyboard_label.csv',
                 places=PLACES, width=256, height=455, gaze_size=64, foveal_encoder=None, peripheral_encoder=None):
        """
        Args: render mode: None, 'human'; screenshot_path: path to the screenshot of the keyboard; positions: positions of the places; size: size of the screenshot
        """
        super().__init__(render_mode, img_folder, position_file, width, height)
        self.logger = logging.getLogger(__name__)
        self.places = places
        self.peripheral_size = 64
        self.observation_space = spaces.Dict(
            {
                "gaze": spaces.Box(0, 1, shape=(2,), dtype=np.float32),
                "foveal": spaces.Box(low=0, high=1, shape=(64,), dtype=np.float32), # hidden vector for foveal vision
                "peripheral": spaces.Box(low=0, high=1, shape=(self.peripheral_size,), dtype=np.float32), # hidden vector for peripheral vision
                "target": spaces.Discrete(len(self.places))
            }
        )
        self.action_space = spaces.MultiDiscrete([width, height, 2])  # x * y * tappin

        self.gaze_size = gaze_size  # size of the gaze
        self.foveal_encoder = foveal_encoder
        self.peripheral_encoder = peripheral_encoder

        self._randomize_gaze()

        """ peripheral pixels """
        img = transforms.Resize((self.peripheral_size, self.peripheral_size))(self.screenshot)
        img_tensor = transforms.ToTensor()(img)
        self.z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()

        self.target = random.choice(self.places)
        self.logger.debug("Target place for the trial set to: {%s}" % self.target)
        self.target_center = self._get_center(self.target)
        self.ep_len = 0

    def _get_obs(self):
        x = self.gaze[0]
        y = self.gaze[1]
        left, top, right, bottom = x-self.gaze_size/2, y-self.gaze_size/2, x+self.gaze_size/2, y+self.gaze_size/2
        img = self.screenshot.crop((left, top, right, bottom))
        img_tensor = transforms.ToTensor()(img)
        foveal_z = torch.squeeze(self.foveal_encoder(img_tensor)).detach().numpy()
        return {
            "gaze": self.gaze / np.array([self.width, self.height]),
            "foveal": foveal_z,
            "peripheral": self.z,
            "target": self.places.index(self.target)
        }

    def step(self, action):

        self.ep_len += 1
        movement = self._action_to_movement(action)

        """ gaze movement """
        self.gaze[0] = action[0]
        self.gaze[1] = action[1]
        self.gaze[0] += np.random.normal(0,5) # add noise
        self.gaze[1] += np.random.normal(0,5) # add noise

        observation = self._get_obs()
        reward = self.reward(movement)
        done = self._gaze_on(self.target) or self.ep_len >= 10
        info = {}
        return observation, reward, done, info

    def reset(self, gaze=None, target=None, reset_kbd=False):
        self.ep_len = 0
        if reset_kbd:
            self._reset_kbd()
        if gaze is not None:
            self.gaze = gaze
        else:
            self._randomize_gaze()
        if target is not None:
            self.target = target
        else:
            self.target = random.choice(self.places)
            self.target = random.choice([self.target, 'input_box'])
        self.target_center = self._get_center(self.target)
        self.logger.debug("Target place for the trial set to: {%s}" % (self.target))

        """ peripheral pixels """
        img = transforms.Resize((self.peripheral_size, self.peripheral_size))(self.screenshot)
        img_tensor = transforms.ToTensor()(img)
        self.z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()

        observation = self._get_obs()
        return observation

    def reward(self, movement):  # speed accurate tradeoff
        max_dist = self.width
        dist = self._distance(self.gaze, self.target_center)
        dist = min(dist, max_dist)
        r = 0
        if self._gaze_on(self.target):  # if gaze on the target place, give a positive reward
            r = 1 - (dist/max_dist)**0.4
        else:
            r -= 0.01  # give a time penalty
        return r

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        else:
            print("=====================================")
            print("Gaze: ", self.gaze)
            print("Target: ", self.target)
            print("Target center: ", self.target_center)

    def _render_frame(self):
        render_fps = 0.5
        resize_factor = 2
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('vision agent')
            self.window = pygame.display.set_mode((self.width * resize_factor, self.height * resize_factor))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        x = self.gaze[0] * resize_factor
        y = self.gaze[1] * resize_factor
        foveated_image = Image.open(self.img_names[self.img_index])
        foveated_image = transforms.Resize((self.height * resize_factor, self.width * resize_factor))(foveated_image)
        mode = foveated_image.mode
        size = foveated_image.size
        data = foveated_image.tobytes()
        image = pygame.image.fromstring(data, size, mode)
        font = pygame.font.SysFont(None, 40)
        target_text = font.render("target: %s" % (self.target), True, 'black')
        # action_text = font.render("action: (r %s, θ %s)"%(self.distance, self.direction), True, 'black')
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(image, (0, 0))
        self.window.blit(target_text, (20, 100))
        # self.window.blit(action_text, (10, 80))
        pygame.draw.circle(self.window, 'red', (x, y), self.gaze_size * resize_factor / 2, 3) #(r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.
        # pygame.draw.rect(self.window, 'red',
        #                  (x - self.gaze_size * resize_factor / 2, y - self.gaze_size * resize_factor / 2,
        #                   self.gaze_size * resize_factor, self.gaze_size * resize_factor), 3)

        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(render_fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _randomize_gaze(self):
        self.gaze = np.array([0, 0])
        self.gaze[0] = self.np_random.integers(0, self.width, size=1, dtype=int)[0]
        self.gaze[1] = self.np_random.integers(0, self.height, size=1, dtype=int)[0]

    def _action_to_movement(self, action):
        new_position = np.array([0, 0])
        new_position[0] = action[0]
        new_position[1] = action[1]
        movement = new_position - self.gaze

        return movement
