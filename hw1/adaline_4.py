import numpy as np
import random
import math
from matplotlib import pyplot as plt

class adaline(object):                      #ADALINE yapısı için tüm fonksiyonları içeren class yapısı
    def __init__(self,dim,c,iteration):     #Nesne oluşturulurken boyut, öğrenme hızı ve maksimum iterasyon sayısı verilir.
        self.dim = dim
        self.Weights = np.full(dim+1,0)     #Ağırlıklar varsayılan olarak 0 lardan oluşan bir vektördür, değiştirilip etkisi yorumlanabilir.
        self.iteration = iteration
        self.c = c
    def sigmoid(self,x):                    #Aktivasyon fonksiyonu olarak kullanacağımız sigmoid
        return 1 / (1 + math.exp(-x))
    def sigmoid_derivative(self,x):         #Ağırlık güncellerken kullanılacak olan sigmoid fonksiyonunun türevi
        return math.exp(-x) / (1 + math.exp(-x))**2
    def train(self,train_data):             #Öğrenme işleminin yapıldığı fonksiyon, parametre olarak eğitim verisini alır
        self.error_vector = []                  #Bu error vektörü her iterasyon sonundaki ortalama hatayı tutar, çizdirme işleminde kullanılıyor
        e_vector = np.zeros(len(train_data))    #Bu e vektörü her data için erroru kaydediyor, iterasyon sonunda ortalama hatayı bulmayı sağlıyor
        for i in range(self.iteration):
            for j in range(len(train_data)):
                with_bias = np.insert(train_data[j].data,self.dim,1)
                dot_product = with_bias.T.dot(self.Weights)
                y = (self.sigmoid(dot_product))
                error = (train_data[j].out_value - y)
                e_vector[j] = error
                self.Weights = self.Weights + self.c * error * self.sigmoid_derivative(dot_product) * with_bias
            stop = np.mean(abs(e_vector))
            self.error_vector.append(stop)
            if stop <= 0.0245:
                print("Weights: ")
                print(self.Weights)
                print("iteration number: ")
                print(i+1)
                return True
            if i == self.iteration-1:
                print("İteration number:",i+1)
                print("Maximum iteration done.")
                return False
    def test(self,test_data):
        true = 0
        false = 0
        for k in range(len(test_data)):
            with_bias = np.insert(test_data[k].data,self.dim,1)
            dot_product = with_bias.T.dot(self.Weights)
            y = (self.sigmoid(dot_product))
            error = test_data[k].out_value - y
            if abs(error) <= 0.098:
                true += 1
            else:
                false += 1
        print("Number of true data:",true)
    def plot(self):
        it_number = np.arange(0,len(self.error_vector),1)
        plt.plot(it_number, self.error_vector, 'g')
        plt.title('Mean Taxicab Norm Error vs Iteration Number')
        plt.xlabel('Iteration Number')
        plt.ylabel('Mean Taxicab Norm Error')
        plt.show()
class data_with_class(object):
    def __init__(self,data,out_value):
        self.data = data
        self.out_value = out_value

if __name__ == '__main__':
    train_data = []
    test_data = []
    for i in range (50): #Veri üretilen kısım
        for j in range (50):
            x1 = i*0.02
            x2 = j*(math.pi/100)
            if i % 2 == 0 and j % 2 == 0:
                #Data vektörüne x1 ve x2'nin yanında x1^2,x2^2 ve x1*x2 eklenerek ADALINE yapısının
                # sorudaki fonksiyon formunda kullanılması sağlanıyor.
                test_data.append(data_with_class([x1,x2,x1**2,x2**2,x1*x2],(0.6*x1+0.4*math.cos(x2))))
            else:
                train_data.append(data_with_class([x1,x2,x1**2,x2**2,x1*x2],(0.6*x1+0.4*math.cos(x2))))
    a1 = adaline(5,0.04,5000)

    shuffled_data = random.sample(train_data, len(train_data))
    #train = a1.train(shuffled_data)
    train = a1.train(train_data)

    if train:
        a1.test(test_data)
    else:
        print("data will not test")
    a1.plot()

