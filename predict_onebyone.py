#!/usr/bin/env python
# coding: utf-8
from utils import testPaths, dataIter, DataIter, srcIndex, \
            trgIndex,mu,sig,homeDirectory, averageFilter, handleAngleData
import torch
from model import model,device
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

model.load_state_dict(torch.load(homeDirectory+'my-model-test.pt'))
n = input("input the line number")
locationData = np.load(homeDirectory+'dataset/'+ testPaths[n])
# 读取一条文件并画图显示轨迹
'''
fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111)
ax.plot(locationData[:,0],locationData[:,1],'ro',label = 'trajectory')            
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ran = [xlim[1]-xlim[0],ylim[1]-ylim[0]]
loc = [(xlim[1]+xlim[0])/2,(ylim[1]+ylim[0])/2]
maxRan = 7200
ran = [loc[0]-maxRan/2, loc[0]+maxRan/2,loc[1]-maxRan/2, loc[1]+maxRan/2]
ax.set_xlim(ran[0:2])
ax.set_ylim(ran[2:4])
ax.legend()
fig.show()
'''

# 获取中心点，并将数据进行归一化
centerLocs = []
nums = len(locationData)
locationDatas = []
filterLocData = averageFilter(locationData[:,trgIndex])
for i in range(nums//5-3):
    locationDatas.append([copy.deepcopy(handleAngleData(locationData[i*5:i*5+10,srcIndex].transpose())), copy.deepcopy(filterLocData[i*5+10:i*5+20,:].transpose())])
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

for i, srclocationData in enumerate(srclocationDatas):
    srclocationDatas[i][0:2] = (srclocationDatas[i][0:2] - mu[:, np.newaxis] ) / sig[:, np.newaxis]
    srclocationDatas[i][2:4] = (srclocationDatas[i][2:4] - mu[:, np.newaxis] ) / sig[:, np.newaxis]
    srclocationDatas[i][4] = (srclocationDatas[i][4] - mu[0, np.newaxis] ) / sig[0, np.newaxis]
    srclocationDatas[i][5] = srclocationDatas[i][5] / 540
    trgLocationDatas[i] = ( trgLocationDatas[i] - mu[:, np.newaxis] ) / sig[:, np.newaxis]

valid_iterator = DataIter(srclocationDatas,trgLocationDatas,device,1, centerLocs)
t0 = np.ones([8,1],dtype = np.float32)
t1 = np.linspace(0,0.7,8).astype(np.float32).reshape((-1,1))
t2= t1*t1
t3= t2*t1
T = np.concatenate([t0,t1,t2,t3],axis = 1)
T_inv = np.linalg.pinv(T)
for i, batch in enumerate(valid_iterator):
    src = batch.src
    trg = batch.trg
    model.eval()
    output = model(src, trg, 0)
    def real_trajectory_plot(src, trg, cents):
        src_reals = src.cpu().numpy()
        trg_reals = trg.cpu().numpy()
        out_reals = output.cpu().detach().numpy()
        cents = cents.cpu().numpy()
        _,batch_size, _ = src_reals.shape
        
        for i in range(batch_size):
            src_reals[:, i, 0:2] = src_reals[:,i,0:2]*sig[np.newaxis,:]+ mu[np.newaxis,:]+ cents[i][np.newaxis,:]
            src_reals[:, i, 2:4] = src_reals[:,i,2:4]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            src_reals[:, i, 4  ] = src_reals[:,i,4  ]*sig[np.newaxis,0]+ mu[np.newaxis,0] + cents[i][np.newaxis,0]
            src_reals[:, i, 5  ] = src_reals[:, i, 4  ] * 540
            trg_reals[:, i, :] = trg_reals[:,i,:]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            out_reals[:, i, :] = out_reals[:,i,:]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            src_real = src_reals[:,i,:]
            trg_real = trg_reals[:,i,:]
            out_real = out_reals[:,i,:]
            plot_train = out_real
            Weights = np.matmul(T_inv,plot_train)
            plot_real = np.matmul(T,Weights)
            fig = plt.figure(figsize=[8,6])
            ax = fig.add_subplot(111)
            ax.plot(src_real[:,0],src_real[:,1],'ro',label = 'src')
            ax.plot(trg_real[:,0],trg_real[:,1],'go',label = 'trg')
            ax.plot(out_real[:,0],out_real[:,1],'bo',label = 'out')
            ax.plot(plot_real[:,0],plot_real[:,1],linestyle = '--',color = 'brown',label = 'plot_real')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ran = [xlim[1]-xlim[0],ylim[1]-ylim[0]]
            loc = [(xlim[1]+xlim[0])/2,(ylim[1]+ylim[0])/2]
            maxRan = 2000
            ran = [loc[0]-maxRan/2, loc[0]+maxRan/2,loc[1]-maxRan/2, loc[1]+maxRan/2]
            ax.set_xlim(ran[0:2])
            ax.set_ylim(ran[2:4])
            ax.legend()
            fig.show()
    real_trajectory_plot(src, trg, batch.cent)
    time.sleep(1)
    
#为了防止程序直接退出，接受一个字符后才结束
raw_input()