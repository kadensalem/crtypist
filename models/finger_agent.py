import gym
import torch
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


def FingerAgent(env=None, policy="MlpPolicy", load=None, log_path="runs/finger_agent"):
    """_summary_

    Args:
        env (_type_, optional): gym env. Defaults to None.
        policy (str, optional): policy type. Defaults to "MlpPolicy". "MultiInputPolicy". "CnnPolicy"
        load (_type_, optional): model path. Defaults to None.
        log_path (str, optional): log path. Defaults to "runs/finger_agent".

    Returns:
        _type_: agent model
    """
    if load:
        model = PPO.load(load, env=env, device="auto") # env can be None if you only need prediction from a trained model
    else:
        # policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 512])
        model = PPO("MultiInputPolicy", env, tensorboard_log=log_path, learning_rate=0.0003, batch_size=64, gamma=0.99,
                    device="auto")
    return model