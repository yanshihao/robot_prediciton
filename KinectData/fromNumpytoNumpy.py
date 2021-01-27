import numpy as np
import csv
import codecs
import os
from math import cos, sin
import matplotlib.pyplot as plt
homeDirectory = "/home/nvidia/catkin_new/src/robot_prediction/KinectData/"

numpyDirectory = homeDirectory + 'Npy/'
outDirectory = homeDirectory


data = [
        "B_rightHipLocX", "B_rightHipLocY", "B_rightHipLocZ",
        "B_rightKneeLocX", "B_rightKneeLocY", "B_rightKneeLocZ",
        "B_leftHipLocX", "B_leftHipLocY", "B_leftHipLocZ",
        "B_leftKneeLocX", "B_leftKneeLocY", "B_leftKneeLocZ",
        "A_rightHipLocX", "A_rightHipLocY", "A_rightHipLocZ",
        "A_rightKneeLocX", "A_rightKneeLocY", "A_rightKneeLocZ",
        "A_leftHipLocX", "A_leftHipLocY", "A_leftHipLocZ",
        "A_leftKneeLocX", "A_leftKneeLocY", "A_leftKneeLocZ",
        "A_rightHipSpdX", "A_rightHipSpdY", "A_rightHipSpdZ",
        "A_rightKneeSpdX", "A_rightKneeSpdY", "A_rightKneeSpdZ",
        "A_leftHipSpdX", "A_leftHipSpdY", "A_leftHipSpdZ",
        "A_leftKneeSpdX", "A_leftKneeSpdY", "A_leftKneeSpdZ",
        "IMU_angleX", "IMU_angleY","IMU_angleZ",
        "IMU_accX", "IMU_accY", "IMU_accZ",
        "IMU_velX", "IMU_velY", "IMU_velZ", "time(s)",
        "odmX", "odmY", "odmTheta"
        ]

b = np.array([0.0007, 0.0021, 0.0021, 0.0007])
a = np.array([1.0000,-2.6236, 2.3147,-0.6855])

def myFilter( b, a, y1 ):
    y2 = np.zeros(np.shape(y1))
    N = np.size(b)
    M = np.size(y1)
    for i in range(N):
        y2[i] = y1[0]
    for i in range(N,M):
        y2[i] = b[0] * y1[i]
        for j in range(1,N):
            y2[i] = y2[i] + b[j] * y1[i-j] - a[j] * y2[i-j]
    return y2


for numpyFilePath in os.listdir(numpyDirectory):
    numpyData = np.load(numpyDirectory + numpyFilePath)
    outputData = np.zeros(np.shape(numpyData))
    outputData[:,0:12] = numpyData[:,0:12]
    outputData[:,18:] = numpyData[:,18:]
    outputData[:,14] = numpyData[:,14]
    outputData[:,17] = numpyData[:,17]
    numpyData[:,12:14] = numpyData[:,12:14] / 1000
    numpyData[:,15:17] = numpyData[:,15:17] / 1000
    preHipLoc = np.copy(numpyData[:,12:14])
    preKneeLoc = np.copy(numpyData[:,15:17])
    # fig1 = plt.figure(figsize=[8,6])
    # ax = fig1.add_subplot(111)
    for i in range( np.shape(outputData)[0] ):
        theta = numpyData[i,48]
        x2robot = cos(theta) * ( numpyData[i,12] - numpyData[i,46] ) + sin(theta) * ( numpyData[i,13] - numpyData[i,47] ) 
        y2robot =-sin(theta) * ( numpyData[i,12] - numpyData[i,46] ) + cos(theta) * ( numpyData[i,13] - numpyData[i,47] ) 
        numpyData[i,12] = x2robot
        numpyData[i,13] = y2robot
        x2robot = cos(theta) * ( numpyData[i,15] - numpyData[i,46] ) + sin(theta) * ( numpyData[i,16] - numpyData[i,47] ) 
        y2robot =-sin(theta) * ( numpyData[i,15] - numpyData[i,46] ) + cos(theta) * ( numpyData[i,16] - numpyData[i,47] ) 
        numpyData[i,15] = x2robot
        numpyData[i,16] = y2robot

    # ax.plot(numpyData[:, 12], 'g-', label='x')
    # ax.plot(numpyData[:, 13], 'r-', label='y')
    
    outputData[:,12] = myFilter(b,a,numpyData[:,12])
    outputData[:,13] = myFilter(b,a,numpyData[:,13])
    outputData[:,15] = myFilter(b,a,numpyData[:,15])
    outputData[:,16] = myFilter(b,a,numpyData[:,16])

    # ax.plot(outputData[:, 12], 'b-', label='aftx')
    # ax.plot(outputData[:, 13], 'b-', label='afty')
    # fig1.show()

    for i in range( np.shape(outputData)[0] ):
        theta = outputData[i,48]
        x2world = cos(theta) * ( outputData[i,12] ) - sin(theta) * ( outputData[i,13] ) +  outputData[i, 46]
        y2world = sin(theta) * ( outputData[i,12] ) + cos(theta) * ( outputData[i,13] ) +  outputData[i, 47]
        outputData[i,12] = x2world
        outputData[i,13] = y2world
        x2world = cos(theta) * ( outputData[i,15] ) - sin(theta) * ( outputData[i,16] ) +  outputData[i, 46]
        y2world = sin(theta) * ( outputData[i,15] ) + cos(theta) * ( outputData[i,16] ) +  outputData[i, 47]
        outputData[i,15] = x2world
        outputData[i,16] = y2world
    '''
    fig2 = plt.figure(figsize=[8,6])
    ax = fig2.add_subplot(111)
    ax.plot(outputData[:, 12], outputData[:, 13] , 'r-', label='aftHip')
    ax.plot(preHipLoc[:,0], preHipLoc[:, 1] , 'b-', label='preHip')
    ax.plot(outputData[:, 15], outputData[:, 16] , 'g-', label='aftKnee')
    ax.plot(preKneeLoc[:,0], preKneeLoc[:, 1] , 'b-', label='preKnee')
    ax.legend()
    ax.axis('equal')
    fig2.show()
    raw_input()
    '''
    np.save(outDirectory + numpyFilePath, outputData )