import numpy as np
import random
import math
import pickle
from matplotlib import pyplot as plt
def data_create():
    data = []
    noises = []
    for i in range(1000):
        noises.append(in_noise())
    targets = [0.1,0.1]
    for i in range(2,1002,1):
        ys1 = targets[i-1]
        ys2 = targets[i-2]
        result = billings(ys1,ys2)
        targets.append(result)
    for i in range(1000):
        data.append(data_with_class(noises[i],targets[i]))
    return data
#Çizdirme işlemleri yapılan fonksiyon
def plot_results(data,autonom):
    x = range(len(data))
    data_array = []
    for i in range(len(data)):
        data_array.append(data[i].target)
    plt.plot(x,data_array)
    plt.plot(x,autonom,'r')
    plt.title("Orijinal Veri ve Ağ Çıktısı")
    plt.show()
def tanh(x,a = 0.5):
    result = np.zeros((len(x),1))
    for i in range(len(x)):
        result[i] = math.tanh(a*x[i,0])
    return result
def tanh_derivative(x,a = 0.5):
    result = np.zeros((len(x),1))
    for i in range(len(x)):
        result[i] = a*(1-(math.tanh(a*x[i,0])*math.tanh(a*x[i,0])))
    return result

def in_noise(a=1):
    return a*(np.random.normal(0,0.01))
#Sistemi modelleyen fonksiyon, indeks, bir önceki çıktısı ve iki öceki çıktısını parametre olarak alıyor
def billings(ys1,ys2):
    y=((0.8-(0.5*np.exp(-1*(ys1**2))))*ys1) - ((0.3 + (0.9*np.exp(-1*(ys1**2))))*ys2) + 0.1*np.sin(math.pi*ys1) + in_noise()
    return y
class elman(object):
    #init metodu, öğrenme hızı ve maksimum iterasyon sayısını alır
    def __init__(self,inDim,hidDim,outDim,c,momentum,iteration):
        self.inDim = inDim
        self.hidDim = hidDim
        self.outDim = outDim
        self.iteration = iteration
        self.c = c
        self.momentum = momentum
        self.load_weights()
    def printw(self):
        print("in weights:" ,self.in_weights)
        print("out weights:" ,self.out_weights)
        print("context weights:" ,self.con_weights)
    def load_weights(self):
        file = open("tursu",'rb')
        self.in_weights = pickle.load(file)
        self.out_weights = pickle.load(file)
        self.con_weights = pickle.load(file)
        self.context_layer = pickle.load(file)
        self.weights_difference_c = np.zeros((self.hidDim,self.hidDim))
        self.weights_difference_in = np.zeros((self.hidDim,self.inDim))
        self.weights_difference_out = np.zeros((self.outDim,self.hidDim))
    def train(self,train_data):
        self.errors = np.zeros((len(train_data)))
        for i in range(self.iteration):
            print("iteration number:",i+1)
            for j in range(len(train_data)):
                dp_in = self.in_weights.dot(train_data[j].data)
                dp_context = self.con_weights.dot(self.context_layer)
                dp = dp_in + dp_context
                first_out = tanh(dp)
                self.context_layer = first_out + 0.4*self.context_layer 
                out = self.out_weights.dot(first_out)
                error = train_data[j].target-out
                self.errors[j] = error
                tmp_conw = self.con_weights
                tmp_inw = self.in_weights
                tmp_outw = self.out_weights
                tmp = self.c*np.multiply((self.out_weights.transpose().dot(error)),(tanh_derivative(dp))) 
                a = np.argmax(np.abs(self.errors))
                #self.printw()
                if ((np.max(np.abs(error)))/np.abs(train_data[a].target))*100 >= 10:
                    self.con_weights += tmp.dot(first_out.transpose())+self.momentum*self.weights_difference_c
                    self.in_weights += tmp.dot(np.transpose(train_data[j].data))+self.momentum*self.weights_difference_in
                    self.out_weights += self.c*error*first_out.transpose()+self.momentum*self.weights_difference_out
                #self.printw()
                self.weights_difference_c = self.con_weights - tmp_conw
                self.weights_difference_in = self.in_weights - tmp_inw
                self.weights_difference_out = self.out_weights - tmp_outw
            abs_error = np.abs(self.errors)
            a = np.argmax(abs_error)
            """print(self.errors)
            print(abs_error)
            print(a)"""
            print(a)
            print(np.max(np.abs(self.errors)))
            print(np.abs(train_data[a].target))
            print(((np.max(np.abs(self.errors)))/np.abs(train_data[a].target))*100)
            if (((np.max(np.abs(self.errors)))/np.abs(train_data[a].target))*100) < 8:
                print("yakınsadı")
                return True
    def test(self,data):
        true = 0
        false = 0
        outs = [] 
        for j in range(len(data)):
            dp_in = self.in_weights.dot(data[j].data)
            #İleri yayılım algoritası yapılan for döngüsü
            dp_context = self.con_weights.dot(self.context_layer)
            dp = dp_in + dp_context
            first_out = tanh(dp)
            self.context_layer = first_out
            out = self.out_weights.dot(first_out)
            out = out[0,0]
            outs.append(out)
            error = data[j].target- out
            if j > 900:
                if abs(error) <= 0.1:
                    true += 1
                else:
                    false += 1 
        print("Number of true data: {} / 100".format(true))  
        return outs
def plot_sp(data,outs):
    axis1d = []
    axis2d = [0.1]
    for i in range(len(data)):
        axis1d.append(data[i].target)
        axis2d.append(data[i].target)
    axis2d = axis2d[:(len(data))]
    plt.plot(axis2d,axis1d,'p')
    plt.title("Billings Sisteminin Durum Portresi")
    plt.xlabel("y(k-1)")
    plt.ylabel("y(k)")
    plt.show()
    plt.plot(axis2d,outs,'o')
    plt.title("Elman ağının Durum Portresi")
    plt.show()
class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

if __name__ == '__main__':
    data = data_create()
    train_data = data[:900]
    """for i in range(len(data)):
        print("data:",data[i].data)
        print("target:",data[i].target)"""
    elman_network = elman(1,8,1,0.05,0.2,3000)
    elman_network.train(train_data)
    test = elman_network.test(data)
    plot_sp(data,test)
    plot_results(data,test)