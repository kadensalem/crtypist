import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import random
import numpy as np
from parameters import decay
from setting import KEYS, KEYS_FIN
from torchmetrics import CharErrorRate

class Memory:
    def __init__(self, keys = KEYS, model_path = 'outputs/wm_encoder.pt'):
        super(Memory, self).__init__()
        self.memory_certainty = 1.0 # certainty of the memory
        self.max_capacity = 10
        self.recall_text = ''
        self.target_text = ''
        self.keys = keys
        self.clf = MemoryEncoder(input_size = 64+2, num_classes=len(keys))
        self.clf.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.clf.eval()
        self.cer = CharErrorRate()

    def target(self, text):
        self.memory_certainty = 1.0 # certainty of the memory
        self.recall_text = ''
        self.target_text = text

    def encode(self, obs): # obs = [vision_obs, finger_obs]
        obs = repeat(obs, 'n -> c n', c=1)
        y_hat = self.clf(obs)
        _, predicted = torch.max(y_hat.data, 1)
        key = self.keys[predicted]
        if key == '<':
            if self.recall_text != '':
                self.recall_text = self.recall_text[:-1]
        elif key == '>':
            self.recall_text += ''
        elif key is not None:
            self.recall_text += key
        else:
            pass
        return key

    def encode_key(self, key):
        if key == '<':
            if self.recall_text != '':
                self.recall_text = self.recall_text[:-1]
        elif key == '>':
            self.recall_text += ''
        elif key is not None:
            self.recall_text += key
        else:
            pass

    def proofread(self, typed_text):
        self.recall_text = typed_text
        self.memory_certainty = 1.0

    def forget(self, time, parameter):
        self.memory_certainty = decay(time, parameter) # update memory certainty

    def recall(self): # recall text
        before_text = ""
        memory_text = self.recall_text
        if len(self.recall_text) > self.max_capacity:
            before_text = self.recall_text[:-self.max_capacity]
            memory_text = self.recall_text[-self.max_capacity:]
        if len(memory_text) > 0:
            uncertain_num_charactors = int(len(memory_text) * ( 1 - self.memory_certainty )) # forgetting based on memory certainty
            if uncertain_num_charactors > 0:
                recall_text_list = list(memory_text)
                for i in range(uncertain_num_charactors):
                    recall_text_list[i] = '_'
                memory_text = ''.join(recall_text_list)
        return before_text + memory_text
        # return self.recall_text

    def correctness(self):
        correctness = 1.0
        recall_text = self.recall()
        # correctness = 1 - self.cer(recall_text + self.target_text[len(recall_text):], self.target_text) 
        correctness = 1 - self.cer(recall_text + self.target_text[len(recall_text):], self.target_text) ** 0.4
        return correctness

    def next_char(self):
        if len(self.recall_text) == 0:
            return self.target_text[0]
        elif len(self.recall_text) < len(self.target_text):
            return self.target_text[len(self.recall_text)]
        else:
            return ">" # end of text

class MemoryEncoder(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size = 128):
        super(MemoryEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=1)
        return out

    def loss(self, y_hat, y):
        criterion = nn.NLLLoss()
        return criterion(y_hat, y)