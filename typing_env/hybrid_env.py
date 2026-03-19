import numpy as np
import pygame
from copy import copy
from torchvision import transforms

from typing_env.internal_env import InternalEnv
from models.chord_memory import ChordMemory
from setting import CHORD_DICT
from parameters import (SPEEDS, TAPPING_TIME, TIMESTEP_TIME, MAX_STEPS,
                        error_distance, error_wo_gaze, fixation_time,
                        time_penalty, decay, CHORD_LEARNING_RATE)


class HybridInternalEnv(InternalEnv):
    """
    Extends InternalEnv with opportunistic bimanual chord substitution.

    When the supervisor issues "type next character" and the remaining target
    text begins with a chord word that the simulated user has memorized, a
    chord fires instead: both thumbs move in parallel (Fitts), and all N
    characters of the word are produced simultaneously.

    Supervisor obs/action space is unchanged — no retraining required.
    """

    def __init__(self, chord_dict=None, initial_chord_strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self._chord_dict = chord_dict if chord_dict is not None else CHORD_DICT
        self._initial_chord_strength = initial_chord_strength
        self.chord_memory = ChordMemory(self._chord_dict,
                                        memory_param=self.parameters[0],
                                        initial_strength=initial_chord_strength)
        self.chord_log = []            # chord events this episode
        self._chord_fired_this_step = False  # flag for update_memory
        self._chord_flash = None       # (left_key, right_key, expiry_ep_len) for rendering

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.chord_memory.reset(memory_param=self.parameters[0],
                                initial_strength=self._initial_chord_strength)
        self.chord_log = []
        self._chord_fired_this_step = False
        self._chord_flash = None
        return obs

    # ------------------------------------------------------------------
    # Chord motor model
    # ------------------------------------------------------------------

    def _chord_motor_time(self, left_key, right_key):
        """
        Parallel Fitts movement time for a bimanual chord.

        Right thumb: from self.finger to right_key position.
        Left thumb:  from the mirrored start position to left_key position.
        Movement time = max(time_right, time_left).
        """
        right_target = self._get_center(right_key).astype(float)
        dist_right = float(np.linalg.norm(self.finger - right_target))
        time_right = max(round(dist_right / self.speed), TAPPING_TIME)

        # Left thumb starts at the keyboard-mirrored position of self.finger
        left_start = np.array([self.width - self.finger[0], self.finger[1]], dtype=float)
        left_target = self._get_center(left_key).astype(float)
        dist_left = float(np.linalg.norm(left_start - left_target))
        time_left = max(round(dist_left / self.speed), TAPPING_TIME)

        return max(time_right, time_left)

    # ------------------------------------------------------------------
    # Chord execution
    # ------------------------------------------------------------------

    def _execute_chord(self, word, action):
        """
        Substitute a chord event for sequential typing of `word`.

        Advances ep_len by the chord movement time, updates typed_text and wm,
        and records state for dual-thumb rendering.
        """
        self.speed = SPEEDS[action[2]]
        left_key, right_key = self._chord_dict[word]
        chord_time = self._chord_motor_time(left_key, right_key)

        # Advance clock by the extra steps beyond the one already counted by step().
        # Fix 1: keep gaze timer in sync with the jumped ep_len.
        extra_steps = max(0, chord_time // TIMESTEP_TIME - 1)
        self.ep_len += extra_steps
        extra_time = extra_steps * TIMESTEP_TIME
        self.gaze_movement_time -= extra_time
        if self.gaze_movement_time <= 0 and self.vision_goal:
            self.vision_in_action = False

        # Update typed text
        self.typed_text += word

        # Fix 3: encode chord chars into WM. Set last_proofreading_time so that the
        # parent's next forget() call uses chord_time as its time argument, simulating
        # that certainty decayed by exactly the chord's motor time. This is cleaner than
        # calling wm.forget() directly (which would be overwritten by update_memory()).
        # Using chord_time lets cert naturally approach the proofreading threshold after
        # each chord, so the supervisor eventually proofreads and can correct motor errors.
        for char in word:
            self.wm.encode_key(key=char)
        self.last_proofreading_time = self.ep_len * TIMESTEP_TIME - chord_time
        self.last_guiding_time = self.ep_len * TIMESTEP_TIME

        # Boost chord strength (simulates reinforcement of the mapping)
        self.chord_memory.use(word)

        # Move right finger to the right_key position (so next move starts correctly)
        right_target = self._get_center(right_key).astype(float)
        self.finger[0] = right_target[0]
        self.finger[1] = right_target[1]

        # Fix 2: record chord keys for dual-thumb rendering (show for ~5 steps)
        self._chord_flash = (left_key, right_key, self.ep_len + 5)

        # Log the chord event
        entry = {
            "vision_goal": self.vision_goal,
            "finger_goal": word,
            "speed": self.speed,
            "gaze": copy(self.gaze),
            "finger": copy(self.finger),
            "is_proofreading": self._gaze_on("input_box"),
            "is_tapping": True,
            "target_text": self.target_text,
            "tapped_key": word[0],
            "typed_text": self.typed_text,
            "chord": True,
            "chord_word": word,
            "chord_time_ms": chord_time,
        }
        self.log.append(entry)
        self.chord_log.append(entry)
        self._chord_fired_this_step = True

    # ------------------------------------------------------------------
    # Step (full override with chord pre-check)
    # ------------------------------------------------------------------

    def step(self, action):
        self.ep_len += 1
        self._chord_fired_this_step = False

        # --- Chord pre-check -------------------------------------------
        # Only attempt when the finger is free and supervisor says "type next"
        chord_fired = False
        if not self.finger_in_action and action[1] == 0:
            remaining = self.target_text[len(self.typed_text):]
            chord_word = self.chord_memory.scan(remaining)
            if chord_word and self.chord_memory.try_recall(chord_word):
                self._execute_chord(chord_word, action)
                chord_fired = True
                # Set finger_goal to next target so vision block has a valid place
                self.finger_goal = self.wm.next_char()

        # --- Normal finger-goal setting (skipped when chord fired) -----
        if not chord_fired:
            if not self.finger_in_action:
                if action[1] == 0:
                    self.finger_goal = self.wm.next_char()
                elif action[1] == 1:
                    self.finger_goal = "<"
                self.speed = SPEEDS[action[2]]

                f_obs = {
                    "pixels": self.peripheral_z,
                    "finger": self.finger / np.array([self.width, self.height]),
                    "tapping": 1 if self.is_tapping else 0,
                    "target": self.keys.index(self.finger_goal)
                }
                pos, _ = self.finger_agent.predict(f_obs, deterministic=True)
                next_pos = [pos[0], pos[1] + int(self.height / 2)]
                distance = self._distance(self.finger, next_pos)
                movement_time = round(distance / self.speed)
                if movement_time < TAPPING_TIME:
                    movement_time = TAPPING_TIME
                self.error = error_distance(movement_time, distance,
                                            self.parameters[1], with_gaze=True)
                self.finger_movement_time = movement_time
                self.finger_in_action = True

        # --- Vision (always runs) --------------------------------------
        if not self.vision_in_action:
            if action[0] == 0 and not self._gaze_on(self.finger_goal):
                self.vision_goal = self.finger_goal
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = False
                if self.vision_goal:
                    self.rollout_vision()
                    self.vision_in_action = True
            elif action[0] == 0 and self._gaze_on(self.finger_goal):
                self.vision_goal = self.finger_goal
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = False
                if self.vision_goal:
                    self.rollout_vision()  # still update gaze to track finger
                self.vision_in_action = True
            elif action[0] == 1 and self.vision_goal != 'input_box':
                self.vision_goal = 'input_box'
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = True
                self.rollout_vision()
                self.vision_in_action = True
            elif action[0] == 1 and self._gaze_on('input_box'):
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = True
                self.vision_in_action = True
        else:
            if self.gaze_movement_time <= 0 and self.vision_goal:
                self.vision_in_action = False

        # --- Timers ----------------------------------------------------
        self.gaze_movement_time -= TIMESTEP_TIME
        self.finger_movement_time -= TIMESTEP_TIME

        # --- Right-finger rollout (skipped when chord fired) -----------
        reward_shaping = 0
        reward_shaping_value = 0.03

        self.is_tapping = False
        if not chord_fired and self.finger_in_action and self.finger_movement_time <= 0:
            self.is_tapping, _ = self.rollout_finger()
            if self.is_tapping and self._where():
                key = self._where()

                if self.finger_goal == "<" and key == "<":
                    if self.typed_text == '':
                        self.incorrect_backspaces += 1
                        reward_shaping -= reward_shaping_value
                    elif len(self.typed_text) <= len(self.target_text):
                        if self.typed_text[-1] != self.target_text[len(self.typed_text) - 1]:
                            self.immediate_backspaces += 1
                            reward_shaping += reward_shaping_value
                        elif self.typed_text != self.target_text[0:len(self.typed_text)]:
                            self.delayed_backspaces += 1
                            reward_shaping += reward_shaping_value / 2
                        else:
                            self.incorrect_backspaces += 1
                            reward_shaping -= reward_shaping_value
                    else:
                        self.immediate_backspaces += 1
                        reward_shaping += reward_shaping_value
                else:
                    if key == self.finger_goal:
                        self.correct_type_keys += 1
                        reward_shaping += reward_shaping_value
                    else:
                        self.incorrect_type_keys += 1
                        reward_shaping -= reward_shaping_value

                if key == '<':
                    if self.typed_text != '':
                        self.typed_text = self.typed_text[:-1]
                elif key == ' ':
                    self.typed_text += ' '
                elif key == '>':
                    self.typed_text += ''
                else:
                    self.typed_text += key

            self.finger_in_action = False

        # --- Memory update ---------------------------------------------
        self.update_memory()

        # --- Observation -----------------------------------------------
        observation = self._get_obs()

        # --- Termination -----------------------------------------------
        done = (self.ep_len >= MAX_STEPS
                or (chord_fired and self.typed_text.endswith('>'))
                or (self._finger_on(">") and self.is_tapping))

        # --- Reward ----------------------------------------------------
        reward = self.reward(done) + reward_shaping

        # --- Log (non-chord steps) -------------------------------------
        if not chord_fired:
            self.log.append({
                "vision_goal": self.vision_goal,
                "finger_goal": self.finger_goal,
                "speed": self.speed,
                "gaze": copy(self.gaze),
                "finger": copy(self.finger),
                "is_proofreading": self._gaze_on("input_box"),
                "is_tapping": self.is_tapping,
                "target_text": self.target_text,
                "tapped_key": self._where(),
                "typed_text": self.typed_text,
                "chord": False,
            })

        return observation, reward, done, {}

    # ------------------------------------------------------------------
    # Memory update (adds chord decay)
    # ------------------------------------------------------------------

    def update_memory(self):
        if not self._chord_fired_this_step:
            super().update_memory()
        self.chord_memory.decay_step()

    # ------------------------------------------------------------------
    # Render (adds chord HUD overlay)
    # ------------------------------------------------------------------

    def _render_frame(self):
        super()._render_frame()   # draws keyboard + finger + gaze + display.update()

        if self.render_mode == "human" and self.window is not None:
            resize_factor = 2

            # Fix 2: draw both thumbs during the chord flash window
            if self._chord_flash and self.ep_len < self._chord_flash[2]:
                left_key, right_key, _ = self._chord_flash
                lc = self._get_center(left_key)
                rc = self._get_center(right_key)
                # Orange filled circle = left thumb
                pygame.draw.circle(self.window, (255, 140, 0),
                                   (int(lc[0] * resize_factor), int(lc[1] * resize_factor)),
                                   16 * resize_factor, 0)
                # Blue filled circle = right thumb (overdraw parent's outline with filled)
                pygame.draw.circle(self.window, 'blue',
                                   (int(rc[0] * resize_factor), int(rc[1] * resize_factor)),
                                   16 * resize_factor, 0)

            # Chord counter top-right
            font_sm = pygame.font.SysFont(None, 28)
            chord_count = len(self.chord_log)
            last_word = self.chord_log[-1]['chord_word'] if self.chord_log else None
            label = f"chords: {chord_count}" + (f"  [{last_word}]" if last_word else "")
            hud_text = font_sm.render(label, True, (180, 60, 180))
            x = self.width * resize_factor - hud_text.get_width() - 8
            self.window.blit(hud_text, (x, 8))
            pygame.display.update()
