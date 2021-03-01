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
import dqn
import MySocket
import logging
import Processing

class Agent:
    def __init__(self, n_action, n_state, CAPACITY, batch_size,eps, GAMMA ):
        self.batch_size = batch_size
        self.GAMMA = GAMMA
        self.brain = dqn.Brain(n_state, n_action, CAPACITY, batch_size, GAMMA)
    def get_action(self, state , episode):
        action = self.brain.select_action(state,episode, eps_decay= 200, eps_start=0.95, eps_end= 0.05)
        return action
    def memorize(self,state, action, state_next, reward):
        self.brain.memory.input_data(state,action,state_next,reward)
    def update_Q(self,epi):
        self.brain.replay(epi)
    def update_targetQ(self):
        self.brain.update_target_net()
    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.input_data_td(td_error)

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()

Unity = MySocket.MySocket()

def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_episode = 1000
    epochs = 1000
    GAMMA = 0.98
    batch_size = 64
    CAPACITY = 50000
    train_frequency = 2
    agent = Agent(2,4,CAPACITY,batch_size,max_episode,GAMMA)
    for epoch in range(epochs) :
        done = False
        logging.info('Start epoch %d' %epoch)
        state, _reward, done, hight = Unity.getdata()
        for eps in range(max_episode) :
            #print('lets start epoch : %d eps : %d'%(epoch,eps))
            #print(state.shape)
            action = agent.get_action(state , epoch)
            #print("send action is ",action[0],action[1])
            Unity.senddata(float(action.item()))

            next_state, _reward, done,hight = Unity.getdata()

            _reward = hight - 3
            _reward = 0
            if done == True :
                print("this epoch is done epi is ",eps)
                if eps < 999 :
                    _reward = -10
                    next_state = None

            _reward = np.array([_reward])
            _reward = torch.from_numpy(_reward).type(torch.FloatTensor).to(device)
            agent.brain.memory.input_data(state,action ,next_state, _reward)
            agent.memorize_td_error(0)
            state = next_state



            agent.update_Q(epoch)

            if done :
                break
                agent.update_td_error_memory()
        if epoch % train_frequency == 0 :
            print("transport train data")
            agent.update_targetQ()

    fil = 'D:/unity'
    torch.save(agent.brain.policy_net.state_dict(), f"{fil}/optim_state_dict.pt")

if __name__ == '__main__':
    main()