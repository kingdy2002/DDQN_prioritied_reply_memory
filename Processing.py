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

class actiondata :
    def __init__(self):
        self.actiontable = []
        self.eachactiontimes = []
        # key는 action의 이름 value는 그 행동의 가짓수
    def makeactiontable(self,**kwargs):
        for key, value in kwargs.items():
            self.eachactiontimes.append(key)
            self.actiontable.append(value)



def actionprocessing(actiondatattable, actiondata) :
    arr = dqn.TensorToNp(actiondata)
    returndata = []
    lastindex = 0
    for i in range(len(actiondatattable.actiontable)) :
        index = actiondatattable.eachactiontimes

        data = arr[0][lastindex:lastindex+index]
        lastindex = lastindex + index
        returndata.append(data)

    return returndata


