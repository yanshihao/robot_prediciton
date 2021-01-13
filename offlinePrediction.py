import rospy
from prediction_msg.msg import FutureTrajectory
from geometry_msgs.msg import Point
from std_msgs.msg import Bool, Float64
import numpy as np
import csv
import codecs
from utils import homeDirectory,testPaths

t0 = np.ones([8,1],dtype = np.float32)
t1 = np.linspace(0,0.7,8).astype(np.float32).reshape((-1,1))
t2= t1*t1
# t3= t2*t1
T = np.concatenate([t0,t1,t2],axis = 1)
T_inv = np.linalg.pinv(T)

class Prediction:
    def __init__(self):
        rospy.init_node("offlinePrediction", anonymous=True)
        self.__controlSignal = False
        self.__pubFuture = rospy.Publisher("predicted_trajectory", FutureTrajectory, queue_size=10)

    def run(self):
        rate = rospy.Rate(13)
        for j in range(len(testPaths)):
            print j, " " , testPaths[j]
        fileIndex = int(input("please input the number: "))
        predictionFilePath = homeDirectory + 'predictionData/predictions' +testPaths[fileIndex][:-4] + '.csv'
        predictionFile = open(predictionFilePath,'r')
        predictionReader = csv.reader(predictionFile)
        predictionData = list(predictionReader)
        predictNum = len(predictionData)/2
        index = 0 
        while not rospy.is_shutdown():
            out_reals = np.zeros([8,2])
            if index == predictNum:
                break
            out_reals[:,0] = predictionData[2*index]
            out_reals[:,1] = predictionData[2*index+1]
            Weights = np.matmul(T_inv,out_reals)
            futureTrajectory = FutureTrajectory()
            for i in range(0, 8):
                point = Point()
                point.x = out_reals[i,0]
                point.y = out_reals[i,1]
                point.z = 0
                futureTrajectory.locations.append(point)
            for i in range(0, 3):
                for j in range(0,2):
                    futureTrajectory.weights.append(Float64(Weights[i,j]/1000)) 
            self.__pubFuture.publish(futureTrajectory)    
            index = index + 1
            rate.sleep()
    

if __name__ == '__main__':
    try:
        prediction = Prediction()
        prediction.run()
    except KeyboardInterrupt:
        pass
