from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import math
from datetime import datetime
import sys
from torch.utils.data._utils import collate
import random
import numpy as np
from to_onnx import to_onnx
import csv
import cv2
import mobilenetv3_b

torch.set_printoptions(threshold=np.inf)  #显示所有参数内容

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_epochs = 1
    learnrate=0.001
    learnchange = [40*2,65*2,90*2]
    learnchangerate = 0.1
    
    net = mobilenetv3_b.mobilenetv3_small()
    net = net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr=learnrate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=learnchange, gamma=learnchangerate,last_epoch=-1)
    
    loss_cal = nn.MSELoss()
    
    input_data = torch.randn(1, 1, 224, 224, device=device)
    gt_data = torch.rand(1, 288, 4, 4, device=device)
  
    
    for epoch in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        
        output_data = net(input_data)
        
        total_loss = loss_cal(output_data, gt_data)
        
        total_loss.backward()
        optimizer.step()
        
    model_file_name = "./mobilenetv3_small.pkl"

    torch.save(net, model_file_name)
    
    to_onnx()
