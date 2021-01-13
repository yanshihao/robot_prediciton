#!/usr/bin/python
# -*- coding: utf-8 -*-
import rospy
from openpose_ros_msgs.msg import HumanDepthList
from geometry_msgs.msg import Pose2D, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from math import pi, cos, sin, sqrt
from Kalman import Kalman
import numpy as np
import time
from plotData import plotData
from utils import Extractor,srcIndex, plotLocationData
# import time

class CollectData:
    def __init__(self):
        rospy.init_node("CollectData", anonymous=True)
        rospy.Subscriber("robot_odm", Pose2D, self.odmCallback)
        rospy.Subscriber("openpose_ros/human_depth_list", HumanDepthList, self.humanListCallback)
        rospy.Subscriber("/robot/controlSignal", Bool, self.controlSignalCallback)
        rospy.Subscriber("/imu", Imu, self.imuCallback)
        rospy.Subscriber("/imu_angle", Vector3, self.angleCallback)
        self.__controlSignal = False
        self.__controlState = 0
        self.__isAngleValid = False
        self.__isImuValid = False
        self.__extract = Extractor()
        self.__data = []

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
            else:
                self.__controlState = 0
                # 将数据存成文件并且清空数据
                nowTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
                fileName = nowTime + ' Turning4' + '.npy'
                result = plotData(self.__data)
                if result is True:
                    np.save(fileName, self.__data)
                    print "save as " + fileName
                else:
                    print "wrong, reject"
                self.__data = []
                self.__extract = Extractor()

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
        collect = CollectData()
        collect.run()
    except KeyboardInterrupt:
        pass
