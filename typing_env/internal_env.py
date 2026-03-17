
import random
import logging
import torch
import numpy as np
from gym import spaces
from PIL import Image
import torchvision.transforms as transforms
from typing_env.kbd_env import KeyboardEnv
from setting import KEYS, PLACES, CHARS
from models.vision_encoder import VisionEncoder
from models.vision_agent import VisionAgent
from models.finger_agent import FingerAgent
from models.memory import Memory
from data.sentences import Sentences
import pygame
from copy import copy
import jamspell
import string
from parameters import *
from config import DEFAULT_MODEL_DIR, DEFAULT_ROOT_DIR
import os.path as osp
from torchmetrics import CharErrorRate

jamspell_corrector = jamspell.TSpellCorrector()
jamspell_corrector.LoadLangModel('outputs/en.bin')
translator = str.maketrans('', '', string.punctuation)
ites = ["None", "JamSpell"] # "Norvig",

class InternalEnv(KeyboardEnv):
    def __init__(self, render_mode=None, img_folder='kbd1k/keyboard_dataset', position_file = 'kbd1k/keyboard_label.csv', text_path='./data/sentences_fin.txt', vision_path='outputs/vision_agent.pt', finger_path='outputs/finger_agent.pt', chars = CHARS, places = PLACES, keys = KEYS, width = 256, height = 455, ite = False, finger_size = 32, gaze_size = 64, parameters = None):
        """
        Args: render mode: None, 'human'; screenshot_path: path to the screenshot of the keyboard; positions: positions of the places; size: size of the screenshot
        """
        super().__init__(render_mode, img_folder, position_file, width, height)
        self.logger = logging.getLogger(__name__)
        self.observation_space = spaces.Dict(
            {
                "certainty": spaces.Box(0, 1, shape=(1,), dtype=np.float32), 
                "correctness": spaces.Box(0, 1, shape=(1,), dtype=np.float32), 
                "P": spaces.Box(0, 1, shape=(3,), dtype=np.float32), # parameters to optimize
                "vision_in_action": spaces.Discrete(2),
                "finger_in_action": spaces.Discrete(2),
            }
        )

        """
        Action space:
        goals for vision agent: 0: look at finger target for visual guidance; 1: look at input box for proofreading; 2: Noop
        goals for finger agent: 0: type next charactor; 1: delete last charactor; 2: Noop
        speeds for finger agent
        """
        self.action_space = spaces.MultiDiscrete([2, 2, len(SPEEDS)])

        self.vision_goal = None
        self.finger_goal = None
        self.speed = 0

        # text information
        self.target_text = "hello" # target text
        self.typed_text = '' # typed text
        self.chars = chars
        self.places = places
        self.keys = keys

        # sub-agents
        self.foveal_encoder = VisionEncoder()
        self.foveal_encoder.load_state_dict(torch.load(osp.join(DEFAULT_MODEL_DIR, 'f_encoder.pt'), map_location=torch.device('cpu'))) # foveal vision
        self.foveal_encoder.eval()
        self.peripheral_encoder = VisionEncoder()
        self.peripheral_encoder.load_state_dict(torch.load(osp.join(DEFAULT_MODEL_DIR, 'p_encoder.pt'), map_location=torch.device('cpu'))) # peripheral vision
        self.peripheral_encoder.eval()
        self.vision_agent = VisionAgent(load=vision_path)
        self.finger_agent = FingerAgent(load=finger_path)
        self.wm = Memory(model_path=osp.join(DEFAULT_MODEL_DIR, 'wm_encoder.pt'))
        self.wm.target(self.target_text)

        """ initialize the environment """
        self.gaze_size = gaze_size # size of the gaze
        self._randomize_gaze()
        self.finger_size = finger_size
        self._randomize_finger()
        img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot) # peripheral pixels
        img_tensor = transforms.ToTensor()(img)
        self.peripheral_z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()
        self.sentence_db = Sentences(load_path=text_path) # database of sentences
        self.log = [] # log the behavior of the agent
        self.ep_len = 0
        self.finger_choice = "right" # right hand default
        self.finger_in_action = False
        self.vision_in_action = False
        self.finger_movement_time = 0
        self.gaze_movement_time = 0
        self.last_proofreading_time = 0
        self.last_guiding_time = 0
        self.ite = ite # JamSpell or None
        self.cer = CharErrorRate()

        """ statistical data """
        self.immediate_backspaces = 0
        self.delayed_backspaces = 0
        self.incorrect_backspaces = 0
        self.correct_type_keys = 0
        self.incorrect_type_keys = 0

        """ initialize the parameters """
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = [0.5,0.5,0.5] # parameters to optimize

    def _get_obs(self):
        observation = {
            "correctness": [self.wm.correctness()], 
            "certainty": [self.wm.memory_certainty], 
            "P": [self.parameters[0], self.parameters[1], self.parameters[2]],
            "vision_in_action": 1 if self.vision_in_action else 0,
            "finger_in_action": 1 if self.finger_in_action else 0,
        }
        return observation

    def step(self, action):
        self.ep_len += 1
        """ set finger """
        if not self.finger_in_action:
            if action[1] == 0: # tap next goal
                self.finger_goal = self.wm.next_char()
            elif action[1] == 1: # tap delete goal
                self.finger_goal = "<"
            self.speed = SPEEDS[action[2]]

            f_obs = {
                "pixels": self.peripheral_z,
                "finger": self.finger / np.array([self.width, self.height]),
                "tapping": 1 if self.is_tapping else 0,
                "target": self.keys.index(self.finger_goal)
            }
            pos, _ = self.finger_agent.predict(f_obs, deterministic=True)
            next_pos = [pos[0], pos[1] + int(self.height/2)]
            distance = self._distance(self.finger, next_pos) # estimate the movement distance based on finger goal
            # print("predictive distance: ", distance)

            movement_time = round(distance / self.speed)
            if movement_time < TAPPING_TIME: movement_time = TAPPING_TIME
            self.error = error_distance(movement_time, distance, self.parameters[1], with_gaze = True) # error distance
            self.finger_movement_time = movement_time
            self.finger_in_action = True

        """ set vision """
        if not self.vision_in_action:
            if action[0] == 0 and not self._gaze_on(self.finger_goal): # look at finger target
                self.vision_goal = self.finger_goal
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = False
                if self.vision_goal:
                    self.rollout_vision() # move gaze
                    self.vision_in_action = True
            elif action[0] == 0 and self._gaze_on(self.finger_goal): # if gaze is already on finger_goal, do not move
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = False
                self.vision_in_action = True
            elif action[0] == 1 and self.vision_goal != 'input_box': # look at input box
                self.vision_goal = 'input_box'
                self.gaze_movement_time = fixation_time(self.parameters[2])
                self.is_proofreading = True
                self.rollout_vision() # move gaze
                self.vision_in_action = True
            elif action[0] == 1 and self._gaze_on('input_box'): # if gaze is already on inputbox, keep proofreading
                self.gaze_movement_time = TIMESTEP_TIME
                self.is_proofreading = True
                self.vision_in_action = True
        else:
            if self.gaze_movement_time <= 0 and self.vision_goal:
                self.vision_in_action = False # gaze movement is done

        """ run vision and gaze """
        self.gaze_movement_time -= TIMESTEP_TIME
        self.finger_movement_time -= TIMESTEP_TIME

        reward_shaping = 0
        reward_shaping_value = 0.03

        self.is_tapping = False
        if self.finger_in_action and self.finger_movement_time <= 0:
            self.is_tapping, _ = self.rollout_finger() # actually run finger
            if self.is_tapping and self._where():

                key = self._where() # tapped key

                """ statistical data & reward shaping """
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
                        self.immediate_backspaces += 1 # over-typed
                        reward_shaping += reward_shaping_value
                else:
                    if key == self.finger_goal:
                        self.correct_type_keys += 1
                        reward_shaping += reward_shaping_value
                    else:
                        self.incorrect_type_keys += 1
                        reward_shaping -= reward_shaping_value

                """ update typed text """
                if key == '<':
                    if self.typed_text != '':
                        self.typed_text = self.typed_text[:-1]
                elif key == ' ':
                    self.typed_text += ' '
                elif key == '>':
                    self.typed_text += ''
                else:
                    self.typed_text += key

            self.finger_in_action = False # finger movement is done

        """ update memory """
        self.update_memory()

        """ get observation """
        observation = self._get_obs()

        """ terminate or not """
        done = self.ep_len >= MAX_STEPS or (self._finger_on(">") and self.is_tapping)

        """ get reward """
        reward = self.reward(done) + reward_shaping

        """ log """
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
            "typed_text": self.typed_text
        })
        info = {}

        return observation, reward, done, info

    def rollout_finger(self):
        f_obs = {
            "pixels": self.peripheral_z,
            "finger": self.finger / np.array([self.width, self.height]),
            "tapping": 1 if self.is_tapping else 0,
            "target": self.keys.index(self.finger_goal)
        }
        action, _ = self.finger_agent.predict(f_obs, deterministic=True)
        is_tapping, _ = self._action_to_pot(action)
        self.finger[0] = action[0]
        self.finger[1] = action[1] + int(self.height/2)

        self.is_guiding_fingers = self._gaze_on(place=self.finger_goal)

        if self.is_guiding_fingers:
            self.last_guiding_time = self.ep_len * TIMESTEP_TIME
        time = self.ep_len * TIMESTEP_TIME - self.last_guiding_time

        self.finger[0] += self.finger_noise_distance(wo_gaze_time = time)
        self.finger[1] += self.finger_noise_distance(wo_gaze_time = time)

        return is_tapping, None

    def finger_noise_distance(self, wo_gaze_time):
        # error_distance = np.random.normal(0,self.error)
        if wo_gaze_time > 0:
            error_distance = np.random.normal(0,self.error + error_wo_gaze(wo_gaze_time))
        else:
            error_distance = np.random.normal(0,self.error)
        return error_distance

    def rollout_vision(self):
        x = self.gaze[0]
        y = self.gaze[1]
        left, top, right, bottom = x-self.gaze_size/2, y-self.gaze_size/2, x+self.gaze_size/2, y+self.gaze_size/2
        img = self.screenshot.crop((left, top, right, bottom))
        img_tensor = transforms.ToTensor()(img)
        foveal_z = torch.squeeze(self.foveal_encoder(img_tensor)).detach().numpy()
        v_obs = {
            "gaze": self.gaze / np.array([self.width, self.height]),
            "foveal": foveal_z,
            "peripheral": self.peripheral_z,
            "target":  self.places.index(self.vision_goal)
        }
        action, _ = self.vision_agent.predict(v_obs, deterministic=True)
        movement = self._action_to_movement(action)
        self.gaze[0] = action[0]
        self.gaze[1] = action[1]
        self.gaze[0] += np.random.normal(0,5) # add gaze movement noise
        self.gaze[1] += np.random.normal(0,5) # add gaze movement noise
        return movement

    def update_memory(self): # update memory when gaze on input box or tap a new key
        if self.is_proofreading:
            """ get all updated typed text """
            self.last_proofreading_time = self.ep_len * TIMESTEP_TIME
            self.wm.proofread(typed_text=self.typed_text)
        elif self.is_tapping and self._where():
            """ forgetting based on memory belief """
            time = self.ep_len * TIMESTEP_TIME - self.last_proofreading_time
            self.wm.forget(time=time, parameter=self.parameters[0])
            """ observe key """
            if self._gaze_on_finger(): # gaze on finger
                """ observe interaction """
                vision_obs = self.peripheral_z
                finger_obs = self.finger / np.array([self.width, self.height])
                x = torch.from_numpy(np.concatenate((vision_obs, finger_obs))).float()
                self.wm.encode(x)
                """ observe key directly """
                # self.wm.encode_key(key = self._where())
            else: # gaze not on finger
                self.wm.encode_key(key = self.finger_goal)

    def reset(self, parameters = None, gaze = None, finger = None, target_text = None, reset_kbd = False):
        if parameters is not None:
            self.parameters = parameters
        else:
            # print(self.parameters)
            memory_param = random.uniform(0, 1)
            finger_param = random.uniform(0, 1)
            vision_param = random.uniform(0, 1)
            self.parameters = [memory_param, finger_param, vision_param]
        self.ep_len = 0
        if reset_kbd:
            self._reset_kbd()
            """ peripheral pixels """
            img = transforms.Resize((self.gaze_size, self.gaze_size))(self.screenshot)
            img_tensor = transforms.ToTensor()(img)
            self.peripheral_z = torch.squeeze(self.peripheral_encoder(img_tensor)).detach().numpy()
        if gaze is not None:
            self.gaze = gaze
        else:
            self._randomize_gaze()
        if finger is not None:
            self.finger = finger
            self.is_tapping = False
        else:
            self._randomize_finger()
        if target_text is not None:
            self.target_text = target_text
        else: # randomly generate a text
            self.target_text = self.sentence_db.random_sentence()
        while len(self.target_text) == 0:
            self.target_text = self.sentence_db.random_sentence()
        self.wm.target(self.target_text)
        self.typed_text = ''
        self.finger_movement_time = 0
        self.gaze_movement_time = 0
        self.last_proofreading_time = 0
        self.last_guiding_time = 0

        """ statistical data """
        self.immediate_backspaces = 0
        self.delayed_backspaces = 0
        self.incorrect_backspaces = 0
        self.correct_type_keys = 0
        self.incorrect_type_keys = 0

        self.log = []
        self.log.append({
                "vision_goal": None,
                "finger_goal": None,
                "speed": 0,
                "gaze": copy(self.gaze),
                "is_proofreading": False,
                "finger": copy(self.finger),
                "is_tapping": self.is_tapping,
                "tapped_key": "",
                "typed_text": self.typed_text
            })
        observation = self._get_obs()
        return observation

    def reward(self, done):
        r = 0
        if done:
            r += (1 - self.cer(self.typed_text, self.target_text).item() ** 0.4) #  ** 0.6 & 0.4
            r -= 1 * time_penalty(self.target_text, self.ep_len * TIMESTEP_TIME)
        return r

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        else:
            print("=====================================")
            print("finger goal: ", self.finger_goal)
            print("target text: ", self.target_text)
            print("typed text:", self.typed_text)
            print("recall text:", self.wm.recall())

    def _render_frame(self):
        render_fps = 20 # 50ms per timestep
        resize_factor = 2
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('crtypist')
            self.window = pygame.display.set_mode((self.width * resize_factor, self.height * resize_factor))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        foveated_image = Image.open(self.img_names[self.img_index])
        foveated_image = transforms.Resize((self.height * resize_factor,self.width * resize_factor))(foveated_image)
        mode = foveated_image.mode
        size = foveated_image.size
        data = foveated_image.tobytes()
        image = pygame.image.fromstring(data, size, mode)
        font = pygame.font.SysFont(None, 40)
        field_text = font.render(self.typed_text, True, 'black')
        target_text = font.render("target: %s"%(self.target_text), True, 'gray')
        recall_text = font.render("memory: %s"%(self.wm.recall()), True, 'black')
        vision_text = font.render("vision goal: %s"%(self.vision_goal), True, 'black')
        finger_text = font.render("finger goal: %s, mt: %s ms"%(self.finger_goal, self.finger_movement_time), True, 'black')
        time_text = font.render("time: %d ms"%(self.ep_len * TIMESTEP_TIME), True, 'black')
        # The following line copies our drawings from `canvas` to the visible window

        self.window.blit(image, (0,0))
        self.window.blit(field_text, (20, 50))
        # self.window.blit(recall_text, (20, 100))
        # self.window.blit(target_text, (20, 140))
        # self.window.blit(time_text, (10, 90))
        # self.window.blit(vision_text, (10, 110))
        # self.window.blit(finger_text, (10, 120))
        if self.is_tapping:
            pygame.draw.circle(self.window, 'blue', (self.finger[0] * resize_factor, self.finger[1] * resize_factor), 16 * resize_factor, 0) #(r, g, b) is color, (x, y) is center, R is radius and filled.
        else:
            if self.finger_in_action:
                pygame.draw.circle(self.window, 'blue', (self.finger[0] * resize_factor, self.finger[1] * resize_factor), 16 * resize_factor, 3)
            else:
                pygame.draw.circle(self.window, 'gray', (self.finger[0] * resize_factor, self.finger[1] * resize_factor), 16 * resize_factor, 3) #(r, g, b) is color, (x, y) is center, R is radius and w is the thickness of the circle border.

        if self.gaze[1] < 200: self.gaze[0] = 40 # TODO: look at the text
        # pygame.draw.circle(self.window, 'red', (self.gaze[0] * resize_factor, self.gaze[1] * resize_factor), self.gaze_size * resize_factor / 2, 3)
        pygame.draw.rect(self.window,'red',((self.gaze[0]-self.gaze_size/2) * resize_factor, (self.gaze[1]-self.gaze_size/2) * resize_factor, self.gaze_size * resize_factor, self.gaze_size * resize_factor), 3)

        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(render_fps)

    def _randomize_gaze(self):
        self.vision_goal = None
        self.gaze = np.array([127, 300]) # gaze on the middle of the keyboard
        self.is_proofreading = False
        self.is_guiding_fingers = False
        self.vision_in_action = False

    def _randomize_finger(self):
        self.finger_goal = None
        self.finger = np.array([255, 454]) # finger on the right of the keyboard
        self.is_tapping = False
        self.finger_in_action = False

    def _action_to_movement(self, action):
        x = action[0]
        y = action[1]
        movement = np.array([x-self.gaze[0], y-self.gaze[1]])
        return movement

    def _action_to_pot(self, action, noise=False): # action to point-and-tap
        is_tapping = action[2] == 0
        return is_tapping, None