""" Yapay Sinir Agları
    Odev 3 - Soru 2 (Iris data)
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar 040170066
    --Aynı kısımların yorumları soru1'de mevcut""" 
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#Datayı oluşturan fonksiyon
def create_data():
    #Data dosyası açılır ve satırlardan liste oluşturulur
    file = open('iris.data','r')
    lines = file.readlines()
    #son satır data içermediğnden çıkarılır
    lines = lines[0:len(lines)-1]
    #Targets
    #setosa = -1
    #versicolor = 0
    #virginica = 1
    data = []
    input_size = 4
    #Her satırdaki data targetına göre data listesine ekleniyor
    for i in range(len(lines)):
        if i < 50:
            target = -1
        elif i < 100:
            target = 0
        else:
            target = 1
        lines[i] = lines[i].split(",")
        tmp_in = np.zeros(input_size)    
        for j in range(input_size):
            tmp_in[j] = float(lines[i][j])
        tmp_in = np.reshape(tmp_in,(-1,1))
        data.append(data_with_class(tmp_in,target))
    #Nihai data döndürülüyor
    return data
def plot_data(train_data,test_data):

        #Figürler pluşturuluyor ve veriler çizdiriliyor     
        fig = plt.figure()
        fig.suptitle('Train Data', fontsize=16)
        
        fig2 = plt.figure()
        fig2.suptitle('Test Data', fontsize=16)
        
        
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')

        
        ax.scatter(train_data[0:30,0], train_data[0:30,1], train_data[0:30,2], c='r', marker='o')
        ax.scatter(train_data[30:60,0], train_data[30:60,1], train_data[30:60,2], c='g', marker='o')
        ax.scatter(train_data[60:90,0], train_data[60:90,1], train_data[60:90,2], c='b', marker='o')
        
        ax2.scatter(test_data[0:20,0], test_data[0:20,1], test_data[0:20,2], c='r', marker='o')
        ax2.scatter(test_data[20:40,0], test_data[20:40,1], test_data[20:40,2], c='g', marker='o')
        ax2.scatter(test_data[40:60,0], test_data[40:60,1], test_data[40:60,2], c='b', marker='o')
        

        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        plt.show()
#Data ve targetı beraber tutan class
class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target
if __name__ == '__main__':
    #Data oluşturuluyor
    data = create_data()
    #Eğitim ve test için bölünüyor, üç  3*30 eğitim, 3*20 test olmak üzere 90 eğitim 60 test datası oluşturuluyor.
    train_data = data[0:30] + data[50:80] + data[100:130]
    test_data = data[30:50] + data[80:100] + data[130:150]
    train_data_m = np.zeros((90,4))
    test_data_m = np.zeros((60,4))
    for i in range(len(train_data)):
        train_data_m[i,:] = np.transpose(train_data[i].data)
    for i in range(len(test_data)):
        test_data_m[i,:] = np.transpose(test_data[i].data)
    pca = PCA(n_components=3)
    pca_fit_train = pca.fit_transform(train_data_m)
    pca_fit_test = pca.fit_transform(test_data_m)
    plot_data(pca_fit_train,pca_fit_test)