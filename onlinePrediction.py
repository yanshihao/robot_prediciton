#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
from openpose_ros_msgs.msg import HumanDepthList
from geometry_msgs.msg import Pose2D, Vector3, Point
from prediction_msg.msg import FutureTrajectory
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float64
from math import pi, cos, sin, sqrt
from Kalman import Kalman
import numpy as np
import time
# from plotData import plotData
from utils import Extractor,srcIndex, plotLocationData
from model import Model
import matplotlib.pyplot as plt
# import time

t0 = np.ones([8,1],dtype = np.float32)
t1 = np.linspace(0,0.7,8).astype(np.float32).reshape((-1,1))
t2= t1*t1
# t3= t2*t1
T = np.concatenate([t0,t1,t2],axis = 1)
T_inv = np.linalg.pinv(T)

class Prediction:
    def __init__(self):
        rospy.init_node("onlinePrediction", anonymous=True)
        self.__controlSignal = False
        self.__controlState = 0
        self.__isAngleValid = False
        self.__isImuValid = False
        self.__extract = Extractor()
        self.__data = []
        self.__isPredictionValid = False
        self.__model = Model()
        self.__collectData = []
        self.__pubFuture = rospy.Publisher("predicted_trajectory", FutureTrajectory, queue_size=10)
        self.__pubLocation = rospy.Publisher("nowLocation", Point , queue_size=10)
        rospy.Subscriber("robot_odm", Pose2D, self.odmCallback, queue_size=10)
        rospy.Subscriber("openpose_ros/human_depth_list", HumanDepthList, self.humanListCallback,queue_size=10)
        rospy.Subscriber("/robot/controlSignal", Bool, self.controlSignalCallback)
        rospy.Subscriber("/imu", Imu, self.imuCallback, queue_size=10)
        rospy.Subscriber("/imu_angle", Vector3, self.angleCallback)

    def odmCallback(self, odm):
        self.__odm = odm

    def controlSignalCallback(self, controlSignal):
        self.__controlSignal = controlSignal.data

    def humanListCallback(self, humanDepthList):
        if self.__isImuValid is False or self.__isAngleValid is False:
            return
        if self.__controlState == 0:
            # 控制状态为0表明还未开始记录
            if self.__controlSignal is False:
                return
            else:
                print "start recording!"
                self.__controlState = 1
        if self.__controlState == 1:
            if self.__controlSignal is True:
                # 记录数据
                curData = self.__extract.extractHumanPose(humanDepthList, self.__odm)
                curData.extend([self.__angleData.x, self.__angleData.y, self.__angleData.z,
                                self.__imuData.linear_acceleration.x, self.__imuData.linear_acceleration.y, self.__imuData.linear_acceleration.z,
                                self.__imuData.angular_velocity.x, self.__imuData.angular_velocity.y, self.__imuData.angular_velocity.z])
                self.__data.append(curData)
                nowLocation = Point()
                nowLocation.x = curData[0]
                nowLocation.y = curData[1]
                nowLocation.z = 0
                self.__pubLocation.publish(nowLocation)
                if len(self.__data) >= 11:
                    self.__isPredictionValid = True
                if len(self.__data) > 11:
                    del self.__data[0]
                if self.__isPredictionValid is True:
                    # print self.__odm
                    npTrajectory = np.array(self.__data)
                    cur = time.time()
                    npFuture = self.__model.predictFuture(npTrajectory[:,srcIndex])
                    Weights = np.matmul(T_inv,npFuture)
                    # plotLocationData(npTrajectory.transpose(), npFuture.transpose())
                    self.__collectData.append([npTrajectory[:,srcIndex].transpose(), npFuture.transpose()])
                    futureTrajectory = FutureTrajectory()
                    for i in range(0, 8):
                        point = Point()
                        point.x = npFuture[i][0]
                        point.y = npFuture[i][1]
                        point.z = 0
                        futureTrajectory.locations.append(point)
                    for i in range(0, 3):
                        for j in range(0,2):
                            futureTrajectory.weights.append(Float64(Weights[i][j]/1000)) 
                    now = time.time()
                    print( now - cur )
                    self.__pubFuture.publish(futureTrajectory)
            else:
                self.__controlState = 0
                # 将数据存成文件并且清空数据
                nowTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                fileName = nowTime + ' Line3' + '.npy'
                # plotData(self.__data)
                np.save(fileName, self.__data)
                self.__data = []
                self.__extract = Extractor()
                print "save as " + fileName
                '''
                for i in range(len(self.__collectData)):
                    if i % 5 == 0:
                        plotLocationData(self.__collectData[i][0], self.__collectData[i][1])
                '''

    def imuCallback(self, imuData):
        self.__isImuValid = True
        self.__imuData = imuData

    def angleCallback(self, angleData):
        self.__isAngleValid = True
        self.__angleData = angleData

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        prediction = Prediction()
        prediction.run()
    except KeyboardInterrupt:
        pass
