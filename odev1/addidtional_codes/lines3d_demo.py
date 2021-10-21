import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math

class data_with_class(object):
    def __init__(self,data,classNum):
        self.data = data
        self.classNum = classNum

if __name__ == "__main__":

    train_data = []
    test_data = []
    for i in range (50):
            for j in range (50):
                if i % 2 == 0 and j % 2 == 0:
                    test_data.append(data_with_class([i*0.02,j*(math.pi/100)],(0.6*i*0.02+0.4*math.cos(j*(math.pi/100)))))
                else:
                    train_data.append(data_with_class([i*0.02,j*(math.pi/100)],(0.6*i*0.02+0.4*math.cos(j*(math.pi/100)))))

    x1 = []
    for i in range(len(train_data)):
        x1.append(train_data[i].data[0])
    x2 = []
    for i in range(len(train_data)):
        x2.append(train_data[i].data[1])
    f = []
    for i in range(len(train_data)):
        f.append(train_data[i].classNum)

    x12 = []
    for i in range(len(test_data)):
        x12.append(test_data[i].data[0])
    x22 = []
    for i in range(len(test_data)):
        x22.append(test_data[i].data[1])
    f2 = []
    for i in range(len(test_data)):
        f2.append(test_data[i].classNum)
    fig = plt.figure()
    fig.suptitle('Train Data', fontsize=16)
    fig2 = plt.figure()
    fig2.suptitle('Test Data', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, f, c='r', marker='o')
    ax2.scatter(x12, x22, f2, c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax2.set_xlabel('X Label')
    ax2.set_ylabel('Y Label')
    ax2.set_zlabel('Z Label')

    plt.show()