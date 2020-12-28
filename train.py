#!/usr/bin/env python
# coding: utf-8
# # 时序序列预测
# 本文要解决的问题主要是根据一段时间内的轨迹来预测后一段时间内的轨迹。此问题与机器翻译问题类似，给出两个对应的时间序列来生成。

import torch
import numpy as np
import time
print(torch.__version__)

import random
from utils import paths, homeDirectory, srcIndex, trgIndex, averageFilter, plotLocationData, handleAngleData

import math


# 打乱顺序
locationDatas = []
TRAIN_SET_RAIDO = 0.7
numPaths = len(paths)
numTrainPaths = math.floor(numPaths*TRAIN_SET_RAIDO)
numTestPaths = numPaths - numTrainPaths
numTrainData = 0
numTestData  = 0
# 0  BrightHipLoc[0] , 1  BrightHipLoc[1] , 2  BrightHipLoc[2],
# 3  BrightKneeLoc[0], 4  BrightKneeLoc[1], 5  BrightKneeLoc[2],
# 6  BleftHipLoc[0]  , 7  BleftHipLoc[1]  , 8  BleftHipLoc[2],
# 9  BleftKneeLoc[0] , 10 BleftKneeLoc[1] , 11 BleftKneeLoc[2],
# 12 ArightHipLoc[0] , 13 ArightHipLoc[1] , 14 ArightHipLoc[2],
# 15 ArightKneeLoc[0], 16 ArightKneeLoc[1], 17 ArightKneeLoc[2],
# 18 AleftHipLoc[0]  , 19 AleftHipLoc[1]  , 20 AleftHipLoc[2],
# 21 AleftKneeLoc[0] , 22 AleftKneeLoc[1] , 23 AleftKneeLoc[2],
# 24 ArightHipSpd[0] , 25 ArightHipSpd[1] , 26 ArightHipSpd[2],
# 27 ArightKneeSpd[0], 28 ArightKneeSpd[1], 29 ArightKneeSpd[2],
# 30 AleftHipSpd[0]  , 31 AleftHipSpd[1]  , 32 AleftHipSpd[2],
# 33 AleftKneeSpd[0] , 34 AleftKneeSpd[1] , 35 AleftKneeSpd[2],
# 36 angleData.x     , 37 angleData.y    ,  38 angleData.z,
# 39 acceleration.x ,  40 acceleration.y ,  41 acceleration.z,
# 42 angular_vel.x  ,  43 angular_vel.y  ,  44 angular_vel.z

for num, path in enumerate(paths):
    locationData = np.load(homeDirectory+'dataset/'+ path)
    filterLocData = averageFilter(locationData[:,trgIndex])
    nums = len(locationData)
    for i in range(nums//10-3):
        locationDatas.append([handleAngleData(locationData[i*10:i*10+10,srcIndex].transpose()), filterLocData[i*10+10:i*10+20,:].transpose()])
        # plotLocationData( srclocationDatas[i], trgLocationDatas[i] )
    if num < numTrainPaths:
        numTrainData += nums//10
    else:
        numTestData  += nums//10

len(locationDatas)
print("the numbers of Paths is : %d"%(numPaths))
print("the numbers of TrainPaths is : %d"%(numTrainPaths))
print("the numbers of TestPaths is : %d"%(numTestPaths))
print("the numbers of TrainData is : %d"%(numTrainData))
print("the numbers of TestData is : %d"%(numTestData))

import copy

# 记录中心位置
centerLocs = []
import math
random.shuffle(locationDatas[0:numTrainData])
random.shuffle(locationDatas[numTrainData:])
srclocationDatas = []
trgLocationDatas = []
for i, locationData in enumerate(locationDatas):
    srclocationDatas.append(locationData[0])
    trgLocationDatas.append(locationData[1])
    centerLoc = copy.deepcopy(locationData[0][0:2,9])
    srclocationDatas[i][0:2, :] = srclocationDatas[i][0:2, :] - centerLoc[:, np.newaxis]
    srclocationDatas[i][2:4, :] = srclocationDatas[i][2:4, :] - centerLoc[:, np.newaxis]
    srclocationDatas[i][4, :] = srclocationDatas[i][4, :] - centerLoc[0, np.newaxis]
    trgLocationDatas[i][:, :] = trgLocationDatas[i][:, :] - centerLoc[:, np.newaxis]
    centerLocs.append(centerLoc)

combineData = srclocationDatas[0][0:2]
combineData = np.concatenate( (combineData, trgLocationDatas[0]), axis=1 )
for i in range(numTrainData - 1):
    combineData = np.concatenate((combineData, srclocationDatas[i+1][0:2], trgLocationDatas[i+1]), axis=1)

mu = np.mean(combineData,1)
sig = np.std(combineData,1)

for i, srclocationData in enumerate(srclocationDatas):
    srclocationDatas[i][0:2] = (srclocationDatas[i][0:2] - mu[:, np.newaxis] ) / sig[:, np.newaxis]
    srclocationDatas[i][2:4] = (srclocationDatas[i][2:4] - mu[:, np.newaxis] ) / sig[:, np.newaxis]
    srclocationDatas[i][4] = (srclocationDatas[i][4] - mu[0, np.newaxis] ) / sig[0, np.newaxis]
    srclocationDatas[i][5] = srclocationDatas[i][5] / 540
    trgLocationDatas[i] = ( trgLocationDatas[i] - mu[:, np.newaxis] ) / sig[:, np.newaxis]

print("the mean of the conbineData is:")
print(mu)
print("the std of the conbineData is:")
print(sig)
print("so the final shape of the conbineData is:")
print(combineData.shape)


import torch
import torch.nn as nn
import torch.optim as optim
from model import model, device
from utils import dataIter, DataIter, srcIndex,trgIndex

batchSize = 2

# 初始化权重
def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    len_iterator = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        len_iterator += 1
    return epoch_loss / len_iterator

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    len_iterator = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            len_iterator += 1
    return epoch_loss / len_iterator

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 40 # 40
CLIP = 1

best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_iterator = DataIter(srclocationDatas[0:numTrainData],trgLocationDatas[0:numTrainData],device,2, centerLocs)
    valid_iterator = DataIter(srclocationDatas[numTrainData:],trgLocationDatas[numTrainData:],device,2, centerLocs)
    # 训练
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    # 验证
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'my-model-test.pt')
    print("Epoch:", epoch+1, "| Time:", epoch_mins, "m", epoch_secs,"s")
    print("\tTrain Loss:", train_loss )
    print("\tVal Loss:", valid_loss )

print('best_valid_loss is ')
print(best_valid_loss)

# 将训练数据和测试数据存储在txt文件中，将mu和sig存在.npy文件中
fileTest = open(homeDirectory+'testPath.txt', mode = 'w')
fileTrain = open(homeDirectory+'trainPath.txt', mode = 'w')
for testPath in paths[int(numTrainPaths):]:
    fileTest.write(testPath)
    fileTest.write('\n')
fileTest.close()

for trainPath in paths[:int(numTrainPaths)]:
    fileTrain.write(trainPath)
    fileTrain.write('\n')
fileTrain.close()

np.save(homeDirectory+'mu.npy', mu)
np.save(homeDirectory+'sig.npy', sig)
