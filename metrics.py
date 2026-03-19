import numpy as np
from torchmetrics import CharErrorRate
from parameters import *

CM_PER_PIXEL = 14 / 255  # 14 cm / 255 pixels
MS_PER_STEP = 50  # 50 ms / step

class Metrics:
    """ Analyzed Metrics """

    def __init__(self, log, target_text):
        self.log = log
        self.target_text = target_text
        self.cer = CharErrorRate()

    def movement_distance(self):
        gaze_distance = 0
        finger_distance = 0
        gaze = self.log[0]['gaze']
        finger = self.log[0]['finger']
        for index in range(1, len(self.log)):
            event = self.log[index]
            gaze_distance += self._distance(gaze, event['gaze'])
            finger_distance += self._distance(finger, event['finger'])
            gaze = event['gaze']
            finger = event['finger']
        return gaze_distance, finger_distance, gaze_distance * CM_PER_PIXEL, finger_distance * CM_PER_PIXEL

    def gaze_shift(self):
        num = 0
        for i in range(1, len(self.log)):
            pre_event = self.log[i - 1]
            pre_gaze = pre_event['gaze']
            cur_event = self.log[i]
            cur_gaze = cur_event['gaze']
            if pre_gaze[1] > 200 and cur_gaze[1] < 200:
                num += 1
        return num

    def gaze_kbd_ratio(self):
        num = len(self.log)
        gaze_on_kbd = 0
        for i in range(1, len(self.log)):
            gaze = self.log[i]['gaze']
            if gaze[1] > 200:
                gaze_on_kbd += 1
        return gaze_on_kbd / num

    def num_backspaces(self):
        num = 0
        # for event in self.log:
        for i in range(1, len(self.log)):
            event = self.log[i]
            pre_event = self.log[i - 1]
            if event['tapped_key'] == '<' and event['is_tapping']:
                # print("=====")
                # print(pre_event)
                num += 1
        if num >= len(self.target_text):  # TODO: rm outliers
            num = len(self.target_text) - 1
        return num

    def error_rate(self):
        error = 0
        keystroke = 0
        for event in self.log:
            if event['is_tapping']:
                keystroke += 1
                if event['tapped_key'] != event["finger_goal"]:
                    error += 1
        return error / keystroke

    def char_error_rate(self):
        typed_text = self.log[-1]['typed_text']
        return self.cer(typed_text, self.target_text).item()

    def IKI(self):
        # Inter-key interval: time (milliseconds) between consecutive keypresses
        times = []
        movement_time = 0
        for event in self.log:
            if event['is_tapping']:
                times.append(movement_time)
                movement_time = TAPPING_TIME
            else:
                movement_time += MS_PER_STEP
        return np.mean(times)

    def WPM(self):
        # Words per minute: https://www.yorku.ca/mack/RN-TextEntrySpeed.html
        time = MS_PER_STEP * len(self.log) / 1000  # seconds
        typed_text = self.log[-1]['typed_text']  # > is the enter
        if typed_text == "": return 0
        typing_speed = (len(typed_text) - 1) / time * 60 * (1 / 5)
        return typing_speed

    def summary(self):
        # finger_distance, gaze_distance, finger_distance_cm, gaze_distance_cm = self.movement_distance()
        return {
            'target_text': self.target_text,
            'typed_text': self.log[-1]['typed_text'],
            'char_error_rate': self.char_error_rate(),
            'IKI': self.IKI(),
            'WPM': self.WPM(),
            'num_backspaces': self.num_backspaces(),
            'gaze_kbd_ratio': self.gaze_kbd_ratio(),
            'gaze_shift': self.gaze_shift()
        }

    def _distance(self, point_A, point_B):
        return np.linalg.norm(point_A - point_B)


class ChordMetrics(Metrics):
    """
    Extends Metrics with chord-specific statistics for HybridInternalEnv logs.
    """

    def chord_use_rate(self):
        """Fraction of target words typed via chord (not sequential)."""
        chord_count = sum(1 for e in self.log if e.get('chord'))
        total_words = len(self.target_text.split())
        return chord_count / total_words if total_words else 0.0

    def chord_wpm_contribution(self):
        """WPM attributable to chord events alone."""
        chord_chars = sum(len(e['chord_word']) for e in self.log if e.get('chord'))
        time_s = MS_PER_STEP * len(self.log) / 1000
        return (chord_chars / 5) / time_s * 60 if time_s > 0 else 0.0

    def summary(self):
        base = super().summary()
        base['chord_count']            = sum(1 for e in self.log if e.get('chord'))
        base['chord_use_rate']         = self.chord_use_rate()
        base['chord_wpm_contribution'] = self.chord_wpm_contribution()
        return base
