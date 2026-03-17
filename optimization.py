import numpy as np
from scipy.stats import norm
from typing_env.internal_env import InternalEnv
from models.supervisor_agent import SupervisorAgent
from metrics import Metrics
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from setting import CHARS, CHARS_FIN, PLACES, PLACES_FIN, KEYS, KEYS_FIN
from copy import copy

""" human behavior metrics """
ERROR_RATE_M = 0.56
ERROR_RATE_SD = 0.71
IKI_M = 380.94
IKI_SD = 50.95
WPM_M = 27.19
WPM_SD = 3.61
NUM_BACKSPACES_M = 2.61
NUM_BACKSPACES_SD = 1.81
GAZE_KBD_RATIO_M = 0.70
GAZE_KBD_RATIO_SD = 0.14
GAZE_SHIFTS_M = 3.91
GAZE_SHIFTS_SD = 1.81
PARAMETERS = [IKI_M, IKI_SD, WPM_M, WPM_SD, NUM_BACKSPACES_M, NUM_BACKSPACES_SD, GAZE_KBD_RATIO_M, GAZE_KBD_RATIO_SD, GAZE_SHIFTS_M, GAZE_SHIFTS_SD]

def kl_divergence(mu1, sigma1, mu2, sigma2):
    return 0.5 * (np.log(sigma2**2 / sigma1**2) + (sigma1**2 + (mu1 - mu2)**2) / sigma2**2 - 1)

def js_distance(mu_p, sigma_p, mu_q, sigma_q):
    # Jensen-Shannon (JS) divergence, which is a symmetric and bounded version of the Kullback-Leibler (KL) divergence. The square root of the Jensen-Shannon divergence is known as the Jensen-Shannon distance, and it ranges from 0 to 1.

    # Calculate the mean and variance of the average distribution M
    mu_m = (mu_p + mu_q) / 2
    variance_m = (sigma_p**2 + sigma_q**2) / 2
    sigma_m = np.sqrt(variance_m)

    kl_pm = kl_divergence(mu_p, sigma_p, mu_m, sigma_m)
    kl_qm = kl_divergence(mu_q, sigma_q, mu_m, sigma_m)
    js_divergence = 0.5 * kl_pm + 0.5 * kl_qm
    js_distance = np.sqrt(js_divergence)
    if js_distance > 2:
        js_distance = 2
    return js_distance

def performance(memory_p, finger_p, vision_p, N = 30):
    env = InternalEnv(img_folder='kbd1k/gboard/', position_file='kbd1k/keyboard_label.csv', text_path='./data/sentences.txt', vision_path='outputs/vision_agent.pt', finger_path='outputs/finger_agent.pt', chars = CHARS, places = PLACES, keys = KEYS)
    agent = SupervisorAgent(env=env, load='outputs/supervisor_agent_gboard.pt')
    ERROR_RATEs = []
    IKIs = []
    WPMs = []
    num_backspaces_list = []
    gaze_kbd_ratios = []
    gaze_shifts = []
    for i in range(N):
        while True:
            obs = env.reset(parameters=[memory_p, finger_p, vision_p])
            done = False
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
            metrics = Metrics(log=env.log, target_text=env.target_text)
            summary = metrics.summary()
            if summary['char_error_rate'] < 0.2: # remove outliers
                ERROR_RATE = summary['char_error_rate'] * 100
                IKI = summary['IKI']
                WPM = summary['WPM']
                num_backspaces = summary['num_backspaces']
                gaze_kbd_ratio = summary['gaze_kbd_ratio']
                gaze_shift = summary['gaze_shift']
                ERROR_RATEs.append(ERROR_RATE)
                IKIs.append(IKI)
                WPMs.append(WPM)
                num_backspaces_list.append(num_backspaces)
                gaze_kbd_ratios.append(gaze_kbd_ratio)
                gaze_shifts.append(gaze_shift)
                break 
        
    
    ERROR_RATE_m = np.mean(ERROR_RATEs)
    ERROR_RATE_sd = np.std(ERROR_RATEs)
    IKI_m = np.mean(IKIs)
    IKI_sd = np.std(IKIs)
    WPM_m = np.mean(WPMs)
    WPM_sd = np.std(WPMs)
    num_backspaces_m = np.mean(num_backspaces_list)
    num_backspaces_sd = np.std(num_backspaces_list)
    gaze_kbd_ratio_m = np.mean(gaze_kbd_ratios)
    gaze_kbd_ratio_sd = np.std(gaze_kbd_ratios)
    gaze_shift_m = np.mean(gaze_shifts)
    gaze_shift_sd = np.std(gaze_shifts)

    return ERROR_RATE_m, ERROR_RATE_sd, WPM_m, WPM_sd, IKI_m, IKI_sd, gaze_kbd_ratio_m, gaze_kbd_ratio_sd, gaze_shift_m, gaze_shift_sd, num_backspaces_m, num_backspaces_sd

def objective_function(memory_p, finger_p, vision_p, N = 50):
    discrepancy_value = 0
    ERROR_RATE_m, ERROR_RATE_sd, WPM_m, WPM_sd, IKI_m, IKI_sd, gaze_kbd_ratio_m, gaze_kbd_ratio_sd, gaze_shift_m, gaze_shift_sd, num_backspaces_m, num_backspaces_sd = performance(memory_p, finger_p, vision_p, N)
    
    # discrepancy_value += ERROR_RATE_m # less error rate is better
    discrepancy_value += js_distance(ERROR_RATE_m, ERROR_RATE_sd, ERROR_RATE_M, ERROR_RATE_SD)
    discrepancy_value += js_distance(WPM_m, WPM_sd, WPM_M, WPM_SD)
    discrepancy_value += js_distance(IKI_m, IKI_sd, IKI_M, IKI_SD)
    discrepancy_value += js_distance(num_backspaces_m, num_backspaces_sd, NUM_BACKSPACES_M, NUM_BACKSPACES_SD)
    discrepancy_value += js_distance(gaze_kbd_ratio_m, gaze_kbd_ratio_sd, GAZE_KBD_RATIO_M, GAZE_KBD_RATIO_SD)
    discrepancy_value += js_distance(gaze_shift_m, gaze_shift_sd, GAZE_SHIFTS_M, GAZE_SHIFTS_SD)
    
    # print("WPM", WPM_m, WPM_sd, WPM_M, WPM_SD)
    # print("IKI", IKI_m, IKI_sd, IKI_M, IKI_SD)
    # print("error rate", ERROR_RATE_m, ERROR_RATE_sd, ERROR_RATE_M, ERROR_RATE_SD)
    # print("#backspaces", num_backspaces_m, num_backspaces_sd, NUM_BACKSPACES_M, NUM_BACKSPACES_SD)
    # print("gaze_kbd_ratio", gaze_kbd_ratio_m, gaze_kbd_ratio_sd, GAZE_KBD_RATIO_M, GAZE_KBD_RATIO_SD)
    # print("#gaze_shift", gaze_shift_m, gaze_shift_sd, GAZE_SHIFTS_M, GAZE_SHIFTS_SD)

    return - discrepancy_value # negative for maximize

def bo():

    # Bounded region of parameter space
    pbounds = {'memory_p': (0, 1), 'finger_p': (0, 1), 'vision_p': (0, 1) }

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=1,
    )

    # logger = JSONLogger(path="./logs/optimization.json")
    # optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        init_points=5,
        n_iter=50,
    )

    print(optimizer.max)
    memory_p = optimizer.max['params']['memory_p']
    finger_p = optimizer.max['params']['finger_p']
    vision_p = optimizer.max['params']['vision_p']
    ERROR_RATE_m, ERROR_RATE_sd, WPM_m, WPM_sd, IKI_m, IKI_sd, gaze_kbd_ratio_m, gaze_kbd_ratio_sd, gaze_shift_m, gaze_shift_sd, num_backspaces_m, num_backspaces_sd = performance(memory_p, finger_p, vision_p)
    print("error rate", ERROR_RATE_m, ERROR_RATE_sd, ERROR_RATE_M, ERROR_RATE_SD)
    print("IKI", IKI_m, IKI_sd, IKI_M, IKI_SD)
    print("WPM", WPM_m, WPM_sd, WPM_M, WPM_SD)
    print("gaze_kbd_ratio", gaze_kbd_ratio_m, gaze_kbd_ratio_sd, GAZE_KBD_RATIO_M, GAZE_KBD_RATIO_SD)
    print("#gaze_shift", gaze_shift_m, gaze_shift_sd, GAZE_SHIFTS_M, GAZE_SHIFTS_SD)
    print("#backspaces", num_backspaces_m, num_backspaces_sd, NUM_BACKSPACES_M, NUM_BACKSPACES_SD)

if __name__ == '__main__':
    bo()
