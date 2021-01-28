from utils import ButterworthFitter
import numpy as np
import matplotlib.pyplot as plt
data = np.load("save.npy")
data = data * 1000
length = np.size( data )
filterData = np.zeros(np.shape(data))
b = np.array([0.0007, 0.0021, 0.0021, 0.0007])
a = np.array([1.0000,-2.6236, 2.3147,-0.6855])
butter = ButterworthFitter(b,a)
for i in range(length):
    filterData[i] = butter.run(data[i])
fig1 = plt.figure(figsize=[8,6])
ax = fig1.add_subplot(111)
ax.plot(filterData, 'g-', label='fitered')
ax.plot(data, 'r-', label='data')
ax.legend()
fig1.show()
raw_input()