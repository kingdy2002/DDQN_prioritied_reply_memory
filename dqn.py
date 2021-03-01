import numpy as np
import time
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR,ExponentialLR
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import datetime
import math
import sys
from PIL import Image
import urllib
import glob
import random
from collections import namedtuple
from sklearn import preprocessing
import Processing

Transition = namedtuple('Transition',('state','action', 'next_state','reward'))

class Replaymemory :
    def __init__(self,  CAPACITY):
        self.CAPACITY = CAPACITY
        self.memory = []
        self.index = 0;
    def input_data(self, *args):
        if len(self.memory) < self.CAPACITY :
            self.memory.append(None)
        self.memory[self.index] = Transition(*args)
        self.index  = (self.index +1) % self.CAPACITY
    def input_data_td(self, td_error):
        if len(self.memory) < self.CAPACITY :
            self.memory.append(None)
        self.memory[self.index] = td_error
        self.index  = (self.index +1) % self.CAPACITY

    def extract_data(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_prioritized_index(self, batch_size):
        TD_ERROR_EPSILON = 0.0001
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON*len(self.memory)
        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexs = []
        index = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list :
            while tmp_sum_absolute_td_error < rand_num :
                tmp_sum_absolute_td_error +=(
                    abs(self.memory[index]) + TD_ERROR_EPSILON
                )
                index +=1
            if index >= len(self.memory) :
                index = len(self.memory) -1
            indexs.append(index)
        return indexs





class NeuralNet(nn.Module) :
    def __init__(self, states, actions):
        super(NeuralNet, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(states,states*4),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.layers = nn.Sequential(
            nn.Linear(states*4,states*4),
            nn.Dropout(p = 0.3),
            nn.ReLU()
        )
        self.last_layer = nn.Sequential(
            nn.Linear(states*4,actions*4),
            nn.Dropout(p = 0.3),
            nn.ReLU(),
            nn.Linear(actions*4,actions)
        )

    def forward(self, x):
        x = self.first_layer(x)
        x = self.layers(x)
        x = self.layers(x)
        x = self.last_layer(x)
        return x

class Brain :
    def __init__(self, n_state, n_action, CAPACITY , batch_size , GAMMA ) :

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_action = n_action
        self.n_state = n_state
        self.GAMMA = GAMMA
        self.batch_size = batch_size
        self.memory = Replaymemory(CAPACITY)
        self.td_error_memory = Replaymemory(CAPACITY)
        self.policy_net = NeuralNet(n_state, n_action).to(self.device)
        self.target_net = NeuralNet(n_state, n_action).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()


    def make_batch(self , epi):
        if epi < 30 :
            transitions = self.memory.extract_data(self.batch_size)
        else :
            indexs = self.td_error_memory.get_prioritized_index(self.batch_size)
            transitions = [self.memory.memory[n] for n in indexs]
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_value(self):
        self.policy_net.eval()
        self.target_net.eval()
        self.state_action_values = self.policy_net(self.state_batch).gather(1, self.action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor).to(self.device)
        a_m[non_final_mask] = self.policy_net(self.non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_state = a_m[non_final_mask].view(-1,1)
        next_state_values[non_final_mask] = self.target_net(self.non_final_next_states).gather(1,a_m_non_final_next_state).detach().squeeze()
        expected_state_action_values = self.reward_batch + self.GAMMA * next_state_values
        return expected_state_action_values

    def update_policy_net(self):
        self.policy_net.train()
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_td_error_memory(self):
        self.policy_net.eval()
        self.target_net.eval()
        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_action_value = self.policy_net(state_batch).gather(1,action_batch)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        a_m = torch.zeros(self.batch_size).type(torch.LongTensor).to(self.device)
        a_m[non_final_mask] = self.policy_net(non_final_next_states).detach().max(1)[1]
        a_m_non_final_next_state = a_m[non_final_mask].view(-1, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,a_m_non_final_next_state).detach().squeeze()
        td_errors = reward_batch + self.GAMMA * next_state_values - state_action_value.squeeze()
        self.td_error_memory.memory = td_errors.detach().numpy().tolist()

    def replay(self,epi):
        if len(self.memory) < self.batch_size :
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_batch(epi)
        self.expected_state_action_values = self.get_expected_state_action_value()
        self.update_policy_net()

    def select_action(self,state, episode,eps_decay, eps_start, eps_end):
        sample = random.random()
        epsilon = eps_end + (eps_start - eps_end) * math.exp(-1. * episode / eps_decay)
        #print("epsilon is :",  epsilon)

        if sample > epsilon:
            self.policy_net.eval()
            return self.policy_net(state).max(1)[1].view(1,1).to(self.device)
        else:
            return torch.LongTensor([[random.randrange(self.n_action)]]).to(self.device)
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def DataNormalization(*args):
    data = preprocessing.normalize(args, norm= 'l1')
    return data

def NpToTensor(numpy) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(numpy).type(torch.LongTensor).to(device)

def TensorToNp(tensor) :
    tensor.to(torch.device("cpu"))
    num = tensor.cpu().numpy()
    return num;

def SpliteActionValue(action,VariousOfLable,VariousOfAction) :
    lis = []
    for epi in range(VariousOfAction):
        index = epi*VariousOfLable/VariousOfAction
        print('shape of action ',action.shape)
        lis.append([action[index:index+VariousOfAction]])
    arr = NpToTensor(np.array(lis))
    return arr
