import matplotlib.pyplot as plt
import numpy as np


def plotData(data):
    data = np.array(data)
    fig1 = plt.figure(figsize=[6, 6])
    ax = fig1.add_subplot(111)
    ax.plot(data[:, 12], data[:, 13], 'ro', label='rightHipLoc')
    ax.plot(data[:, 15], data[:, 16], 'go', label='rightKneeLoc')
    # ax.plot(data[:, 6], data[:, 7], 'bo', label='leftHipLoc')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ran = [7000, 7000]
    loc = [(xlim[1]+xlim[0])/2, (ylim[1]+ylim[0])/2]
    ran = [loc[0]-ran[0]/2, loc[0]+ran[0]/2, loc[1]-ran[1]/2, loc[1]+ran[1]/2]
    ax.set_xlim(ran[0:2])
    ax.set_ylim(ran[2:4])
    ax.legend()
    fig1.show()
    fig2 = plt.figure(figsize=[6, 6])
    ax2 = fig2.add_subplot(121)
    ax2.plot(data[:, 38])
    ax3 = fig2.add_subplot(122)
    ax3.plot(data[:, 18])
    fig2.show()
    print 'please input yes to confirm or no to reject'
    result = ''
    while(True):
        result = raw_input()
        if result == 'yes' or result == 'no':
            break
        print 'please input yes to confirm or no to reject'
    plt.close(fig1)
    plt.close(fig2)
    if result == 'yes':
        return True
    else:
        return False
