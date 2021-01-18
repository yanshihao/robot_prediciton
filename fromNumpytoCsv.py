from utils import homeDirectory
import numpy as np
import csv
import codecs
import os

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
        "IMU_velX", "IMU_velY", "IMU_velZ", "time(s)"
        ]

numpyDirectory = homeDirectory + 'dataset/'
csvDirectory = homeDirectory + 'CSV/'
for numpyFilePath in os.listdir(numpyDirectory):
    numpyData = np.load(numpyDirectory + numpyFilePath)
    csvFilePath = numpyFilePath[:-3] + 'csv'
    csvfile = codecs.open( csvDirectory + csvFilePath, 'w', 'gbk')
    writer = csv.writer(csvfile)
    writer.writerow(data)

    for i in range(0, np.shape(numpyData)[0]):
        writer.writerow(numpyData[i].tolist())
    csvfile.close()