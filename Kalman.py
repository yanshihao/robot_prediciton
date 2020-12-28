#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np


class Kalman:
    def __init__(self, Q=[[0.2420, 0, 0], [0, 0.2584, 0], [0, 0, 2.1601]], R=[[0.414, 0, 0], [0, 0.2406, 0], [0, 0, 1.8386]]):
        self.__T__ = 0.1
        self.__A__ = np.array([
            [1, self.__T__, 0.5*self.__T__*self.__T__],
            [0, 1, self.__T__],
            [0, 0, 1]])
        self.__H__ = np.eye(3)
        self.__Q__ = np.array(Q)
        self.__R__ = np.array(R)
        self.__P__ = 0.1 * np.eye(3)
        self.__start__ = 0

    def run(self, p, dp, ddp):
        if self.__start__ == 0:
            self.__start__ = 1
            # 状态变量
            self.__x__ = np.array([[p], [0], [0]])
            return (self.__x__[0][0], self.__x__[1][0])
        elif self.__start__ == 1:
            self.__start__ = 2
            self.__x__ = np.array([[p], [dp], [0]])
            return (self.__x__[0][0], self.__x__[1][0])
        # 卡尔曼滤波五方程
        # 预测
        xp = np.matmul(self.__A__, self.__x__)
        Pp = np.matmul(np.matmul(self.__A__, self.__P__), self.__A__.transpose()) + self.__Q__
        # 更新
        temp = np.linalg.inv(np.matmul(np.matmul(self.__H__, Pp), self.__H__.transpose()) + self.__R__)
        K = np.matmul(np.matmul(Pp, self.__H__.transpose()), temp)
        z = np.array([[p], [dp], [ddp]])
        self.__x__ = xp + np.matmul(K, z - np.matmul(self.__H__, xp))
        self.__P__ = Pp - np.matmul(np.matmul(K, self.__H__), Pp)
        return (self.__x__[0][0], self.__x__[1][0])


class XYZKalman:
    def __init__(self, XQ=0.5, YQ=0.5, ZQ=0.5, XR=0.1, YR=0.1, ZR=0.1):
        self.__XKalman = Kalman(XQ, XR)
        self.__YKalman = Kalman(YQ, YR)
        self.__ZKalman = Kalman(ZQ, ZR)

    def run(self, xyz, dxyz, ddxyz):
        ret = [0, 0, 0]
        for i in range(0, 3):
            ret[i] = self.__XKalman.run(xyz[i], dxyz[i], ddxyz[i])
        return ret
