import random
import logging
import torch
import numpy as np
from gym import spaces
import pygame
from PIL import Image
import torchvision.transforms as transforms
from typing_env.kbd_env import KeyboardEnv
from setting import KEYS, KEYS_FIN
from copy import copy


class FingerEnv(KeyboardEnv):
    def __init__(self, render_mode=None, img_folder='kbd1k/keyboard_dataset', position_file='kbd1k/keyboard_label.csv',
                 places=KEYS,
                 width=256, height=455, with_noise=False, finger_size=32, gaze_size=64, encoder=None):
        """
        Args: render mode: None, 'human'; screenshot_path: path to the screenshot of the keyboard; positions: positions of the places; size: size of the screenshot
        """
        super().__init__(render_mode, img_folder, position_file, width, height)
        self.logger = logging.getLogger(__name__)
        self.keys = places
        self.total_movement = 0
        self.max_length = 20
        self.movement = np.array([0, 0])
        self.shortest_distance = 0
        self.finger = np.array([0, 0])
        self.init_finger = np.array([0, 0])
        self.is_tapping = False
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(low=0, high=1, shape=(gaze_size,), dtype=np.float32), # hidden vector for peripheral pixels
                "finger": spaces.Box(0, 1, shape=(2,), dtype=np.float32),  # finger position
                "tapping": spaces.Discrete(2),  # 0: not tapping, 1: tapping
                "target": spaces.Discrete(len(self.keys))
            }
        )
        # self.action_space = spaces.MultiDiscrete([10,36,2])  # ditances * directions * tapping
        self.action_space = spaces.MultiDiscrete([width, height/2,2])  # x * y * tapping

        self.gaze_size = gaze_size  # size of the gaze
        self.encoder = encoder
        self._randomize_gaze()

        self.text = ''  # text that is being typed

        """ peripheral pixels """
        img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot)
        img_tensor = transforms.ToTensor()(img)
        self.z = torch.squeeze(self.encoder(img_tensor)).detach().numpy()

        self.finger_size = finger_size
        self.with_noise = with_noise
        self._randomize_finger()

        self.target = random.choice(self.keys)
        self.logger.debug("Target place for the trial set to: {%s}" % self.target)
        self.target_center = self._get_center(self.target)
        self.ep_len = 0

    def _get_obs(self):
        return {
            "pixels": self.z,
            "finger": self.finger / np.array([self.width, self.height]),
            "tapping": 1 if self.is_tapping else 0,
            "target": self.keys.index(self.target)
        }

    def step(self, action):

        self.ep_len += 1
        """ gaze movement """
        self.is_tapping, self.movement = self._action_to_pot(action + np.array([0, int(self.height/2), 0], dtype=int))

        """ finger movement """
        self.finger[0] = action[0]
        self.finger[1] = action[1] + int(self.height/2)
        if self.with_noise:
            self.finger[0] += np.random.normal(0,5) # add noise
            self.finger[1] += np.random.normal(0,5) # add noise

        """ tapping """
        if self.is_tapping and self._where():
            key = self._where()
            if key == '<':
                if self.text != '':
                    self.text = self.text[:-1]
            elif key == '>':
                self.text += ''
            else:
                self.text += key

        observation = self._get_obs()
        self.total_movement += np.linalg.norm(self.movement)

        done = self.is_tapping or self.ep_len >= self.max_length  # is_tapping and self._finger_on(self.target) (self._finger_on(self.target) and self.is_tapping)
        reward = self.reward()
        info = {"is_tapping": self.is_tapping, "movement": self.movement, "finger": self.finger}

        return observation, reward, done, info

    def reset(self, gaze=None, finger=None, target=None, clear_text=False, reset_kbd=True):
        self.ep_len = 0
        if reset_kbd:
            self._reset_kbd()
        if clear_text: self.text = ''
        if gaze is not None:
            self.gaze = gaze
        else:
            self._randomize_gaze()
        if finger is not None:
            self.finger = finger
            self.is_tapping = False
        else:
            self._randomize_finger()
        if target is not None:
            self.target = target
        else:
            self.target = random.choice(self.keys)
        self.target_center = self._get_center(self.target)
        self.total_movement = 0
        self.is_tapping = False
        self.movement = np.array([0, 0])
        self.init_finger = copy(self.finger)
        self.shortest_distance = np.linalg.norm(
            self.init_finger - self.target_center)  # distance between initial finger position and target
        self.logger.debug("Target place for the trial set to: {%s}" % self.target)
        """ peripheral pixels """
        img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot)
        img_tensor = transforms.ToTensor()(img)
        self.z = torch.squeeze(self.encoder(img_tensor)).detach().numpy()
        observation = self._get_obs()
        return observation

    def reward(self):
        max_dist = self.width
        dist = self._distance(self.finger, self.target_center)
        dist = min(dist, max_dist)
        r = 0
        if not self._finger_on(self.target) and self.is_tapping:
            if self.shortest_distance > 0:
                r += 0.5 * ( 1 - self._distance(self.finger, self.target_center) / self.shortest_distance ) # reward shapping
            else:
                r += 0.5 * ( 1 - self._distance(self.finger, self.target_center) )
        if self._finger_on(self.target) and self.is_tapping:  # if tap on the target place, give a positive reward
            r = 1 - (dist/max_dist)**0.4
        elif not self._finger_on(self.target) and self.is_tapping:
            r = 0.25 * (1 - (dist/max_dist)**0.4)
        else:
            r = -0.01  # give a time penalty
        return r

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        else:
            print("=====================================")
            print("Gaze: ", self.gaze)
            print("Finger: ", self.finger)
            print("Target: ", self.target)

    def _render_frame(self):
        render_fps = 0.5
        resize_factor = 2
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('finger agent')
            self.window = pygame.display.set_mode((self.width * resize_factor, self.height * resize_factor))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        foveated_image = Image.open(self.img_names[self.img_index])
        foveated_image = transforms.Resize((self.height * resize_factor, self.width * resize_factor))(foveated_image)
        mode = foveated_image.mode
        size = foveated_image.size
        data = foveated_image.tobytes()
        image = pygame.image.fromstring(data, size, mode)
        font = pygame.font.SysFont(None, 40)
        field_text = font.render(self.text, True, 'black')
        target_text = font.render("target: %s" % (self.target), True, 'black')
        finger_text = font.render("finger: %s)" % (self.finger), True, 'black')
        move_or_tap = "tap" if self.is_tapping else "move"
        action_text = font.render("action:dx%s,dy%s" % (self.movement[0], self.movement[1]), True,
                                  'black')
        # The following line copies our drawings from `canvas` to the visible window

        self.window.blit(image, (0, 0))
        self.window.blit(field_text, (20, 46))
        self.window.blit(target_text, (20, 100))
        # self.window.blit(finger_text, (10, 80))
        # self.window.blit(action_text, (10, 90))
        if self.is_tapping:
            pygame.draw.circle(self.window, 'blue', (self.finger[0] * resize_factor, self.finger[1] * resize_factor),
                               16 * resize_factor,
                               0)  # (r, g, b) is color, (x, y) is center, R is radius and filled.
        else:
            pygame.draw.circle(self.window, 'blue', (self.finger[0] * resize_factor, self.finger[1] * resize_factor),
                               16 * resize_factor,
                               3)  # (r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.
        # pygame.draw.rect(self.window,'red',(self.gaze[0]-self.gaze_size/2, self.gaze[1]-self.gaze_size/2, self.gaze_size, self.gaze_size), 3)

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
        self.gaze[0] = self.np_random.integers(self.gaze_size / 2, self.width - self.gaze_size / 2, size=1, dtype=int)[
            0]
        self.gaze[1] = \
            self.np_random.integers(int(self.height / 2) + self.gaze_size / 2, self.height - self.gaze_size / 2, size=1,
                                    dtype=int)[0]  # only gaze on the lower half of the screen (keyboard area)

    def _randomize_finger(self):
        self.finger = np.array([0, 0])
        self.finger[0] = self.np_random.integers(0, self.width, size=1, dtype=int)[0]
        self.finger[1] = self.np_random.integers(int(self.height / 2), self.height, size=1, dtype=int)[0]
        self.is_tapping = False

    def _action_to_pot(self, action, noise=False):  # action to point-and-tap
        x = action[0]
        y = action[1]
        is_tapping = action[2] == 0
        movement = np.array([x - self.finger[0], y - self.finger[1]])
        return is_tapping, movement
