#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
import os
import math
import torch
import copy
homeDirectory = os.getcwd()
testPaths  = []
trainPaths = []
paths = []
if os.path.exists(homeDirectory+"testPath.txt") and os.path.exists(homeDirectory+"trainPath.txt"):
    fileTest = open(homeDirectory+'testPath.txt',mode = 'r')
    testPathsLines = fileTest.readlines()
    for testPathsLine in testPathsLines:
        testPath = testPathsLine.strip('\n')
        testPaths.append(testPath)
    
    fileTrain = open(homeDirectory+'trainPath.txt',mode = 'r')
    trainPathsLines = fileTrain.readlines()
    for trainPathsLine in trainPathsLines:
        trainPath = trainPathsLine.strip('\n')
        trainPaths.append(trainPath)
    paths = trainPaths + testPaths
else:
    paths = os.listdir(homeDirectory + 'dataset')
    paths.sort()
    random.shuffle(paths)
    TRAIN_SET_RAIDO = 0.7
    numPaths = len(paths)
    numTrainPaths = int(math.floor(numPaths*TRAIN_SET_RAIDO))
    testPaths = paths[numTrainPaths:]
    trainPaths = paths[:numTrainPaths]
    # save 
    fileTest = open(homeDirectory+'testPath.txt', mode = 'w')
    fileTrain = open(homeDirectory+'trainPath.txt', mode = 'w')
    for testPath in paths[numTrainPaths:]:
        fileTest.write(testPath)
        fileTest.write('\n')
    fileTest.close()

    for trainPath in paths[:numTrainPaths]:
        fileTrain.write(trainPath)
        fileTrain.write('\n')
    fileTrain.close()

import numpy as np

class dataIter():
    def __init__(self,src, trg, cent):
        self.src = src
        self.trg = trg
        self.cent = cent

srcIndex = [12,13,15,16,18,38] 
trgIndex = [12,13]
batchSize = 2

# mu = np.load(homeDirectory+'mu.npy')
# sig = np.load(homeDirectory+'sig.npy')

mu = np.load(homeDirectory+'mu.npy')
sig = np.load(homeDirectory+'sig.npy')

def DataIter(srclocationDatas,trgLocationDatas,device,batch, centerLocs):
    numTrainData = len(srclocationDatas)
    times = numTrainData//batch
    # f = srclocationDatas[0].shape[0]
    f = centerLocs[0].shape[0]
    data = []
    for time in range(times):
        src = np.empty([10,batch,6],dtype = float)
        for i in range(batch): src[:,i,:] = srclocationDatas[batch*time + i][:,0:10].T
        trg = np.empty([8,batch,2],dtype = float)
        for i in range(batch): trg[:,i,:] = trgLocationDatas[batch*time + i][:,0:8].T
        cent = np.empty([batch,f],dtype = float)
        for i in range(batch): cent[i,:] = centerLocs[batch*time + i]
        yield dataIter(torch.tensor(src,dtype=torch.float32).to(device),
                           torch.tensor(trg,dtype=torch.float32).to(device),
                           torch.tensor(cent,dtype=torch.float32).to(device))


from Kalman import Kalman
from math import pi, cos, sin, sqrt

def averageFilter( data ):
    n = data.shape[0]
    ret = copy.deepcopy(data)
    for i in range(10, n - 10):
        ret[i] = sum(data[i-5:i+5]) / 10
    return ret

import matplotlib.pyplot as plt
def plotLocationData(srcData, trgData):
    fig1 = plt.figure(figsize=[12, 6])
    ax = fig1.add_subplot(121)
    ax.plot(srcData[0,:], srcData[1,:], 'ro', label='srcData')
    ax.plot(trgData[0,:], trgData[1,:], 'go', label='trgData')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ran = [7000, 7000]
    loc = [(xlim[1]+xlim[0])/2, (ylim[1]+ylim[0])/2]
    ran = [loc[0]-ran[0]/2, loc[0]+ran[0]/2, loc[1]-ran[1]/2, loc[1]+ran[1]/2]
    ax.set_xlim(ran[0:2])
    ax.set_ylim(ran[2:4])
    ax.legend()
    ax2 = fig1.add_subplot(122)
    ax2.plot(srcData[5,:])
    ax2.set_ylim( [-270, 270] )
    fig1.show()

def handleAngleData(data):
    initAngle = data[5, 0]
    n = data.shape[1]
    for i in range(1, n):
        if data[5, i] - initAngle >= 180:
            data[5, i] = data[5, i] - 360
        elif data[5, i] - initAngle <= -180:
            data[5, i] = data[5, i] + 360
    return data


rightHip = 9
rightKnee = 10
leftHip = 12
leftKnee = 12


def isDataValid(keyPoint):
    if abs(keyPoint.x) < 1e-6 and abs(keyPoint.y) < 1e-6 and abs(keyPoint.depth) < 1e-6:
        return False
    return True


def isNpDataValid(keyPoint, preKeyPoint):
    if(abs(keyPoint[0] < 1e-6) and abs(keyPoint[1] < 1e-6 and abs(keyPoint[2] < 1e-6))):
        return False
    if(abs(preKeyPoint[0] < 1e-6) and abs(preKeyPoint[1] < 1e-6 and abs(preKeyPoint[2] < 1e-6))):
        return True
    sum = 0
    for i in range(0, 3):
        sum += (preKeyPoint[i] - keyPoint[i]) * (preKeyPoint[i] - keyPoint[i])
    if sqrt(sum) > 0.7:
        return False
    return True


# 只要有一个数据是有效的，那么数据整体就是有效的
def isHumanDataValid(humanDepthList):
    if humanDepthList.num_humans == 0:
        return False
    humanDepth = humanDepthList.human_depth_list[0].body_key_points_with_depth
    if isDataValid(humanDepth[rightHip]) is True:
        return True
    if isDataValid(humanDepth[rightKnee]) is True:
        return True
    return False


class Extractor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.__preRightKneeLoc = np.array([0.0, 0.0, 0.0])
        self.__preRightKneeSpd = np.array([0.0, 0.0, 0.0])
        self.__preRightKneeAcc = np.array([0.0, 0.0, 0.0])
        self.__preRightHipLoc = np.array([0.0, 0.0, 0.0])
        self.__preRightHipSpd = np.array([0.0, 0.0, 0.0])
        self.__preRightHipAcc = np.array([0.0, 0.0, 0.0])
        self.__rightKneeKalX = Kalman()
        self.__rightKneeKalY = Kalman()
        self.__rightKneeKalZ = Kalman()
        self.__rightHipKalX = Kalman()
        self.__rightHipKalY = Kalman()
        self.__rightHipKalZ = Kalman()
        # 左边
        self.__preLeftKneeLoc = np.array([0.0, 0.0, 0.0])
        self.__preLeftKneeSpd = np.array([0.0, 0.0, 0.0])
        self.__preLeftKneeAcc = np.array([0.0, 0.0, 0.0])
        self.__preLeftHipLoc = np.array([0.0, 0.0, 0.0])
        self.__preLeftHipSpd = np.array([0.0, 0.0, 0.0])
        self.__preLeftHipAcc = np.array([0.0, 0.0, 0.0])
        self.__leftKneeKalX = Kalman()
        self.__leftKneeKalY = Kalman()
        self.__leftKneeKalZ = Kalman()
        self.__leftHipKalX = Kalman()
        self.__leftHipKalY = Kalman()
        self.__leftHipKalZ = Kalman()

    def extract(self, humanDepthList, robotLoc):
        if humanDepthList.num_humans == 0:
            return (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
        firstMan = humanDepthList.human_depth_list[0].body_key_points_with_depth
        # 目前只处理了机器人前后移动的任务
        # 处理右髋
        rightHipLoc = np.array([0.0, 0.0, 0.0])
        rightHipPose = firstMan[rightHip]
        rightHipLocX = rightHipPose.x
        rightHipLocY = cos(15.0/180*pi)*rightHipPose.depth - sin(15.0/180*pi)*rightHipPose.y - 0.15
        rightHipLocZ = sin(15.0/180*pi)*rightHipPose.depth + cos(15.0/180*pi)*rightHipPose.y
        rightHipLoc[0] = robotLoc.x + rightHipLocX * cos(robotLoc.theta) - rightHipLocY * sin(robotLoc.theta)
        rightHipLoc[1] = robotLoc.y + rightHipLocX * sin(robotLoc.theta) + rightHipLocY * cos(robotLoc.theta)
        rightHipLoc[2] = rightHipLocZ
        # 处理右膝
        rightKneeLoc = np.array([0.0, 0.0, 0.0])
        rightKneePose = firstMan[rightKnee]
        rightKneeLocX = rightKneePose.x
        rightKneeLocY = cos(15.0/180*pi)*rightKneePose.depth - sin(15.0/180*pi)*rightKneePose.y - 0.15
        rightKneeLocZ = sin(15.0/180*pi)*rightKneePose.depth + cos(15.0/180*pi)*rightKneePose.y
        rightKneeLoc[0] = robotLoc.x + rightKneeLocX * cos(robotLoc.theta) - rightKneeLocY * sin(robotLoc.theta)
        rightKneeLoc[1] = robotLoc.y + rightKneeLocX * sin(robotLoc.theta) + rightKneeLocY * cos(robotLoc.theta)
        rightKneeLoc[2] = rightKneeLocZ
        # 处理左髋
        leftHipLoc = np.array([0.0, 0.0, 0.0])
        leftHipPose = firstMan[leftHip]
        leftHipLocX = leftHipPose.x
        leftHipLocY = cos(15.0/180*pi)*leftHipPose.depth - sin(15.0/180*pi)*leftHipPose.y - 0.15
        leftHipLocZ = sin(15.0/180*pi)*leftHipPose.depth + cos(15.0/180*pi)*leftHipPose.y
        leftHipLoc[0] = robotLoc.x + leftHipLocX * cos(robotLoc.theta) - leftHipLocY * sin(robotLoc.theta)
        leftHipLoc[1] = robotLoc.y + leftHipLocX * sin(robotLoc.theta) + leftHipLocY * cos(robotLoc.theta)
        leftHipLoc[2] = leftHipLocZ
        # 处理左膝
        leftKneeLoc = np.array([0.0, 0.0, 0.0])
        leftKneePose = firstMan[leftKnee]
        leftKneeLocX = leftKneePose.x
        leftKneeLocY = cos(15.0/180*pi)*leftKneePose.depth - sin(15.0/180*pi)*leftKneePose.y - 0.15
        leftKneeLocZ = sin(15.0/180*pi)*leftKneePose.depth + cos(15.0/180*pi)*leftKneePose.y
        leftKneeLoc[0] = robotLoc.x + leftKneeLocX * cos(robotLoc.theta) - leftKneeLocY * sin(robotLoc.theta)
        leftKneeLoc[1] = robotLoc.y + leftKneeLocX * sin(robotLoc.theta) + leftKneeLocY * cos(robotLoc.theta)
        leftKneeLoc[2] = leftKneeLocZ
        return (rightHipLoc, rightKneeLoc, leftHipLoc, leftKneeLoc)

    def extractHumanPose(self, humanDepthList, robotLoc):
        # 提取髋关节和膝关节的坐标位置，注意此处需要进行坐标变换
        # 很有可能需要用多线程加锁的形式实现（此处暂时用的单线程，但是可以预计会有较大延迟）
        # numpy
        record = []
        (rightHipLoc, rightKneeLoc, leftHipLoc, leftKneeLoc) = self.extract(humanDepthList, robotLoc)
        record.extend(rightHipLoc)
        record.extend(rightKneeLoc)
        record.extend(leftHipLoc)
        record.extend(leftKneeLoc)
        # 打印
        # 计算右髋关节的速度和加速度
        if isNpDataValid(rightHipLoc, self.__preRightHipLoc) is False:
            rightHipLoc = self.__preRightHipLoc + 0.1 * self.__preRightHipSpd
            self.__preRightHipAcc = np.array([0.0, 0.0, 0.0])
        else:
            rightHipSpd = 10 * (rightHipLoc - self.__preRightHipLoc)
            self.__preRightHipAcc = 10 * (rightHipSpd - self.__preRightHipSpd)
            self.__preRightHipSpd = rightHipSpd
        # 计算右膝关节的速度和加速度
        if isNpDataValid(rightKneeLoc, self.__preRightKneeLoc) is False:
            rightKneeLoc = self.__preRightKneeLoc + 0.1 * self.__preRightKneeSpd
            self.__preRightKneeAcc = np.array([0.0, 0.0, 0.0])
        else:
            rightKneeSpd = 10 * (rightKneeLoc - self.__preRightKneeLoc)
            self.__preRightKneeAcc = 10 * (rightKneeSpd - self.__preRightKneeSpd)
            self.__preRightKneeSpd = rightKneeSpd
        # 计算左髋关节的速度和加速度
        if isNpDataValid(leftHipLoc, self.__preLeftHipLoc) is False:
            leftHipLoc = self.__preLeftHipLoc + 0.1 * self.__preLeftHipSpd
            self.__preLeftHipAcc = np.array([0.0, 0.0, 0.0])
        else:
            leftHipSpd = 10 * (leftHipLoc - self.__preLeftHipLoc)
            self.__preLeftHipAcc = 10 * (leftHipSpd - self.__preLeftHipSpd)
            self.__preLeftHipSpd = leftHipSpd
        # 计算左膝关节的速度和加速度
        if isNpDataValid(leftKneeLoc, self.__preLeftKneeLoc) is False:
            leftKneeLoc = self.__preLeftKneeLoc + 0.1 * self.__preLeftKneeSpd
            self.__preLeftKneeAcc = np.array([0.0, 0.0, 0.0])
        else:
            leftKneeSpd = 10 * (leftKneeLoc - self.__preLeftKneeLoc)
            self.__preLeftKneeAcc = 10 * (leftKneeSpd - self.__preLeftKneeSpd)
            self.__preLeftKneeSpd = leftKneeSpd
        # 卡尔曼滤波
        rightHipSpd = np.array([0.0, 0.0, 0.0])
        rightKneeSpd = np.array([0.0, 0.0, 0.0])
        leftHipSpd = np.array([0.0, 0.0, 0.0])
        leftKneeSpd = np.array([0.0, 0.0, 0.0])
        # 右
        (rightHipLoc[0], rightHipSpd[0]) = self.__rightHipKalX.run(rightHipLoc[0], self.__preRightHipSpd[0], self.__preRightHipAcc[0])
        (rightHipLoc[1], rightHipSpd[1]) = self.__rightHipKalY.run(rightHipLoc[1], self.__preRightHipSpd[1], self.__preRightHipAcc[1])
        (rightHipLoc[2], rightHipSpd[2]) = self.__rightHipKalZ.run(rightHipLoc[2], self.__preRightHipSpd[2], self.__preRightHipAcc[2])
        (rightKneeLoc[0], rightKneeSpd[0]) = self.__rightKneeKalX.run(rightKneeLoc[0], self.__preRightKneeSpd[0], self.__preRightKneeAcc[0])
        (rightKneeLoc[1], rightKneeSpd[1]) = self.__rightKneeKalY.run(rightKneeLoc[1], self.__preRightKneeSpd[1], self.__preRightKneeAcc[1])
        (rightKneeLoc[2], rightKneeSpd[2]) = self.__rightKneeKalZ.run(rightKneeLoc[2], self.__preRightKneeSpd[2], self.__preRightKneeAcc[2])
        # 左
        (leftHipLoc[0], leftHipSpd[0]) = self.__leftHipKalX.run(leftHipLoc[0], self.__preLeftHipSpd[0], self.__preLeftHipAcc[0])
        (leftHipLoc[1], leftHipSpd[1]) = self.__leftHipKalY.run(leftHipLoc[1], self.__preLeftHipSpd[1], self.__preLeftHipAcc[1])
        (leftHipLoc[2], leftHipSpd[2]) = self.__leftHipKalZ.run(leftHipLoc[2], self.__preLeftHipSpd[2], self.__preLeftHipAcc[2])
        (leftKneeLoc[0], leftKneeSpd[0]) = self.__leftKneeKalX.run(leftKneeLoc[0], self.__preLeftKneeSpd[0], self.__preLeftKneeAcc[0])
        (leftKneeLoc[1], leftKneeSpd[1]) = self.__leftKneeKalY.run(leftKneeLoc[1], self.__preLeftKneeSpd[1], self.__preLeftKneeAcc[1])
        (leftKneeLoc[2], leftKneeSpd[2]) = self.__leftKneeKalZ.run(leftKneeLoc[2], self.__preLeftKneeSpd[2], self.__preLeftKneeAcc[2])
        self.__preRightHipLoc = rightHipLoc
        self.__preRightKneeLoc = rightKneeLoc
        self.__preLeftHipLoc = leftHipLoc
        self.__preLeftKneeLoc = leftKneeLoc
        record.extend([
            rightHipLoc[0]*1000, rightHipLoc[1]*1000, rightHipLoc[2]*1000,
            rightKneeLoc[0]*1000, rightKneeLoc[1]*1000, rightKneeLoc[2]*1000,
            leftHipLoc[0]*1000, leftHipLoc[1]*1000, leftHipLoc[2]*1000,
            leftKneeLoc[0]*1000, leftKneeLoc[1]*1000, leftKneeLoc[2]*1000,
            rightHipSpd[0]*1000, rightHipSpd[1]*1000, rightHipSpd[2]*1000,
            rightKneeSpd[0]*1000, rightKneeSpd[1]*1000, rightKneeSpd[2]*1000,
            leftHipSpd[0]*1000, leftHipSpd[1]*1000, leftHipSpd[2]*1000,
            leftKneeSpd[0]*1000, leftKneeSpd[1]*1000, leftKneeSpd[2]*1000
            ])
        return record
