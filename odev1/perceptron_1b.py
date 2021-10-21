""" EHB 420 - Yapay Sinir Ağları Ödev - 1 Soru 1-b
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar  040170066                    """

import numpy as np
import random

class perceptron(object):
    def __init__(self,dim,c,iteration):
        self.dim = dim
        self.Weights = np.full(dim+1,0)
        self.iteration = iteration
        self.c = c

    def activation(self, summation):
        if summation > 0:
            return 1
        else:
            return -1
    def create_data(self, data_size):
        group1 = []
        group2 = []
        for i in range(int(data_size/2)): #group one data 
            group1.insert(i,data_with_class([2*i+1,2*i+1,2*i+1,2*i+1,2*i+1],1))
        for j in range(int(data_size/2)): #group two data
            group2.insert(j,data_with_class([2*j,2*j,2*j,2*j,2*j],-1))
        train_data = group1[0:12] + group2[0:13]
        test_data = group1[12:20] + group2[13:20]
        data = train_data + test_data
        return data

    def train(self,train_data):
        for i in range(self.iteration):
            count = 0
            for j in range(len(train_data)):
                with_bias = np.insert(train_data[j].data,self.dim,1)
                dot_product = with_bias.T.dot(self.Weights)
                y = self.activation(dot_product)
                error = train_data[j].classNum - y
                if error == 0:
                    count += 1
                self.Weights = self.Weights + self.c * error * with_bias
            if count == 25:
                print("Weights: ")
                print(self.Weights)
                print("iteration number: ")
                print(i+1)
                break
            if i == self.iteration-1:
                print("İteration number:",i+1)
                print("Maximum iteration done. Data is not linearly seperable")
                return False

    def test(self,test_data):
        true = 0
        false = 0
        for k in range(len(test_data)):
            with_bias = np.insert(test_data[k].data,self.dim,1)
            dot_product = with_bias.T.dot(self.Weights)
            y = self.activation(dot_product)
            if test_data[k].classNum == y:
                true += 1
            else:
                false += 1
        print("Number of true data:",true)

class data_with_class(object):
    def __init__(self,data,classNum):
        self.data = data
        self.classNum = classNum

if __name__ == '__main__':
    p1 = perceptron(5,0.01,500)
    data = p1.create_data(40)
    train_data = data[0:25]
    test_data = data[25:40]

    """shuffled_data = random.sample(train_data, len(train_data))
    p1.train(shuffled_data)"""

    train = p1.train(train_data)
    if train:
        p1.test(test_data)
    else:
        print("data will not test")

