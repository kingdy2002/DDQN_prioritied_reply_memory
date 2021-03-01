import socket
from struct import *
import dqn
import torch
import numpy as np


class MySocket :
    def __init__(self) :
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ClientSocket = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
        self.ClientSocket.connect(('110.76.78.109',1111))
        self.ClientSocket.setblocking(True)

    def senddata(self, action):
        data = pack("f", action)
        self.ClientSocket.send(data)

    def getdata(self):
        #print("product getdata")
        data = self.ClientSocket.recv(1024)
        #print(data)
        pktFormat = 'ffff?f'
        pktSize = calcsize(pktFormat)
        #print("pktSize is ",pktSize)
        data1, data2, data3, data4,  done, hight = unpack(pktFormat, data[:pktSize])
        #print(data1,data2,data3,data4)
        #train_data = dqn.DataNormalization(np.array([data1, data2, data3, data4]))
        train_data = torch.from_numpy(np.array([[data1, data2, data3, data4]])).type(torch.FloatTensor).to(self.device)
        #train_data = torch.unsqueeze(train_data, 0)
        return (train_data, 0, done, hight)