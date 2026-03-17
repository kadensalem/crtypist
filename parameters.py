import numpy as np
from math import sqrt, atan, pi, log, exp
from setting import *

MAX_STEPS = 2000
TIMESTEP_TIME = 50 # ms
SPEEDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # pix/ms (width = 256 pixels) # , 1.3, 1.4, 1.5
TAPPING_TIME = 150 # Minimal of tapping time from "CHI'21 Supervisory Optimal Control" source code

def fixation_time(parameter):
    emma_K = 0.05 * parameter
    emma_k = 0.4
    dist = 2 # Foveal vision constitutes 1-2 degrees of our visual field.
    freq = 1 / len(PLACES)
    t_enc = emma_K * -log(freq) * exp(emma_k * dist) * 1000
    SACCADE_TIME = 200 # The total time to prepare and execute a saccade closely resembles saccade latencies of approximately 200 ms cited in many previous studies (An Integrated Model of Eye Movements and Visual Encoding)
    fixation_time = t_enc + SACCADE_TIME
    return fixation_time

def error_distance(finger_movement_time, distance, parameter, with_gaze = True):
    K = parameter * 0.18 # 0.12, 0.2, 0.3
    k_a = 0.6
    x = finger_movement_time / 1000
    x_0 = 0.092
    y_0 = 0.0018
    error_distance_sigma = (pow(K / pow(x - x_0, k_a), 1 / (1 - k_a)) + y_0) * distance
    # print(K, error_distance_sigma)
    return error_distance_sigma

def error_wo_gaze(wo_gaze_time):
    time = wo_gaze_time / 1000
    error_distance_sigma = time * 10 # 5
    return error_distance_sigma

def decay(t, parameter):
    K = parameter * 0.3 # * 2 or 0.3
    t = t / 1000 # ms => s
    return np.exp(-K*t)

def time_penalty(target_text, duration):
    penalty = (duration / len(target_text)) / 1000 # NORMALIZED_TIME_PENALTY, 1000ms => 1s
    return penalty

# parameters
memory_p = 0.32052004358604397
finger_p = 0.4075566536411429
vision_p = 0.86587701380511
parameters = [memory_p, finger_p, vision_p]