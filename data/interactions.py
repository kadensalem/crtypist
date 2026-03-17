import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import random


class Interactions(Dataset):
    def __init__(self, env, agent, places, num=64000):
        self.env = env
        self.agent = agent
        self.places = places
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        target = random.choice(self.places)
        key = None
        while key is None:
            obs = self.env.reset(target=target)  # reset the environment
            done = False
            while not done:
                action, _states = self.agent.predict(obs, deterministic=True)  # take action
                obs, reward, done, info = self.env.step(action)
            key = self.env._where()
        vision_obs = obs['pixels']
        finger_obs = obs['finger']
        x = torch.from_numpy(np.concatenate((vision_obs, finger_obs))).float()
        # y = F.one_hot(torch.tensor(self.places.index(key)), num_classes=len(self.places)).float()
        y = self.places.index(key)
        return x, y, key