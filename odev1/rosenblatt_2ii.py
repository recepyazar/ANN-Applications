""" EHB 420 - Yapay Sinir Ağları Ödev - 1 Soru 2
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar  040170066                    """

import numpy as np
import random

class rosenblatt(object):
    def __init__(self,dim,c,iteration):
        self.dim = dim
        self.Weights = np.full(dim+2,0)
        self.iteration = iteration
        self.c = c

    def activation(self, summation):
        if summation > 0:
            return 1
        else:
            return -1
    def neurons(self, data):
        out1 = data[0]
        out2 = data[1]
        out3 = data[0]**2+data[1]**2-3
        out = np.array([out1,out2,out3])
        return out

    def train(self,train_data):
        for i in range(self.iteration):
            count = 0
            for j in range(len(train_data)):
                out = self.neurons(train_data[j].data)
                with_bias = np.insert(out,self.dim+1,1)
                dot_product = with_bias.T.dot(self.Weights)
                y = self.activation(dot_product)
                error = train_data[j].classNum - y
                if error == 0:
                    count += 1
                self.Weights = self.Weights + self.c * error * with_bias
            if count == len(train_data):
                print("Weights: ")
                print(self.Weights)
                print("iteration number: ")
                print(i+1)
                return True
            if i == self.iteration-1:
                print("İteration number:",i+1)
                print("Maximum iteration done. Data is not linearly seperable.")
                return False
    def test(self,test_data):
        true = 0
        false = 0
        for k in range(len(test_data)):
            out = self.neurons(test_data[k].data)
            with_bias = np.insert(out,self.dim+1,1)
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
        self.data = np.array(data)
        self.classNum = classNum

if __name__ == '__main__':
    data = [data_with_class([[ 0.],[-1.]],1),
    data_with_class([[ 0.],[  0.]], 1),
    data_with_class([[ 0.],[  1.]], 1),
    data_with_class([[ 1.],[ -1.]], 1),
    data_with_class([[ 1.],[  0.]], 1),
    data_with_class([[ 1.],[  1.]], 1),
    data_with_class([[-1.],[ -1.]], 1),
    data_with_class([[-1.],[  0.]], 1),
    data_with_class([[-0.],[  1.]], 1),
    data_with_class([[-3.],[  3.]],-1),
    data_with_class([[-3.],[  1.]],-1),
    data_with_class([[-3.],[  0.]],-1),
    data_with_class([[-3.],[ -1.]],-1),
    data_with_class([[-3.],[ -3.]],-1),
    data_with_class([[-1.],[  3.]],-1),
    data_with_class([[-1.],[ -3.]],-1),
    data_with_class([[ 0.],[  3.]],-1),
    data_with_class([[ 0.],[ -3.]],-1),
    data_with_class([[ 1.],[  3.]],-1),
    data_with_class([[ 1.],[ -3.]],-1),
    data_with_class([[ 3.],[  3.]],-1),
    data_with_class([[ 3.],[  1.]],-1),
    data_with_class([[ 3.],[  0.]],-1),
    data_with_class([[ 3.],[ -1.]],-1),
    data_with_class([[ 3.],[ -3.]],-1),
    data_with_class([[-2.],[  3.]],-1),
    data_with_class([[-3.],[  2.]],-1),
    data_with_class([[-3.],[ -2.]],-1),
    data_with_class([[-2.],[ -3.]],-1),
    data_with_class([[ 2.],[  3.]],-1),
    data_with_class([[ 3.],[  2.]],-1),
    data_with_class([[ 3.],[ -2.]],-1),
    data_with_class([[ 2.],[ -3.]],-1)]
    
    train_data = data[0:6] + data[9:23]
    test_data = data[6:9] + data[23:33]
    r1 = rosenblatt(2,0.5,500)

    shuffled_data = random.sample(train_data, len(train_data))
    r1.train(shuffled_data)

    #train = r1.train(train_data)
    #if train:
    r1.test(test_data)
    #else:
    print("Data will not test.")
