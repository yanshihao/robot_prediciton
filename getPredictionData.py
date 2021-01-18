#!/usr/bin/env python
# coding: utf-8
from utils import testPaths, dataIter, DataIter, srcIndex, \
            trgIndex,mu,sig,homeDirectory, averageFilter, handleAngleData
import torch
import codecs
import csv
from model import model,device
import matplotlib.pyplot as plt
import numpy as np
import copy
import time

model.load_state_dict(torch.load(homeDirectory+'my-model-test.pt'))

for n in range(len(testPaths)):
    locationData = np.load(homeDirectory+'dataset/'+ testPaths[n])
    locationFilePath = homeDirectory + 'predictionData/locations' +testPaths[n][:-4] + '.csv'
    file = codecs.open(locationFilePath, 'w', 'gbk')
    writer = csv.writer(file)
    for i in range(0, np.shape(locationData)[0]):
        writer.writerow(locationData[i,trgIndex])
    file.close()

    angleFilePath = homeDirectory + 'predictionData/angle' +testPaths[n][:-4] + '.csv'
    angleFile = codecs.open(angleFilePath, 'w', 'gbk')
    angleWriter = csv.writer(angleFile)
    predictionFilePath = homeDirectory + 'predictionData/predictions' +testPaths[n][:-4] + '.csv'
    file = codecs.open(predictionFilePath, 'w', 'gbk')
    writer = csv.writer(file)
    # 获取中心点，并将数据进行归一化
    centerLocs = []
    nums = len(locationData)
    locationDatas = []
    filterLocData = averageFilter(locationData[:,trgIndex])
    for i in range(nums-20):
        locationDatas.append([copy.deepcopy(handleAngleData(locationData[i:i+10,srcIndex].transpose())), copy.deepcopy(filterLocData[i+10:i+20,:].transpose())])
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
    t1 = np.linspace(0,1.003,8).astype(np.float32).reshape((-1,1))
    t2= t1*t1
    t3= t2*t1
    T = np.concatenate([t0,t1,t2,t3],axis = 1)
    T_inv = np.linalg.pinv(T)
    for i, batch in enumerate(valid_iterator):
        src = batch.src
        trg = batch.trg
        model.eval()
        output = model(src, trg, 0)
        cents = batch.cent
        src_reals = src.cpu().numpy()
        trg_reals = trg.cpu().numpy()
        out_reals = output.cpu().detach().numpy()
        cents = cents.cpu().numpy()
        _,batch_size, _ = src_reals.shape
        for i in range(batch_size):
            src_reals[:, i, 0:2] = src_reals[:,i,0:2]*sig[np.newaxis,:]+ mu[np.newaxis,:]+ cents[i][np.newaxis,:]
            src_reals[:, i, 2:4] = src_reals[:,i,2:4]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            src_reals[:, i, 4  ] = src_reals[:,i,4  ]*sig[np.newaxis,0]+ mu[np.newaxis,0] + cents[i][np.newaxis,0]
            src_reals[:, i, 5  ] = src_reals[:, i, 5  ] * 540
            trg_reals[:, i, :] = trg_reals[:,i,:]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            out_reals[:, i, :] = out_reals[:,i,:]*sig[np.newaxis,:]+ mu[np.newaxis,:] + cents[i][np.newaxis,:]
            angleWriter.writerow( src_reals[9:10, i, 5  ] )
            writer.writerow(out_reals[:,i,0])
            writer.writerow(out_reals[:,i,1])
            writer.writerow(trg_reals[:,i,0])
            writer.writerow(trg_reals[:,i,1])
    file.close()
    angleFile.close()
#为了防止程序直接退出，接受一个字符后才结束
raw_input()
