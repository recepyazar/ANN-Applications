""" EHB 420 - Yapay Sinir Ağları Ödev - 1 Soru 1-a
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar  040170066                    """

import numpy as np
import random

class perceptron(object):                  #Perceptron yapısı için tüm fonksiyonları içeren class yapısı
    def __init__(self,dim,c,iteration):    #Nesne oluşturulurken boyut, öğrenme hızı ve maksimum iterasyon sayısı parametre olarak girilmelidir.
        self.dim = dim
        self.Weights = np.full(dim+1,0)    #Ağırlıklar varsayılan olarak 0 lardan oluşan bir vektördür, değiştirilip etkisi yorumlanabilir.
        self.iteration = iteration
        self.c = c
    def activation(self, summation):       #Aktivasyon fonksiyonu olarak kullanılacak fonksiyon
        if summation > 0:
            return 1
        else:
            return -1
    def train(self,train_data):         #Eğitim fonksiyonu
        for i in range(self.iteration): #Eğitimin tamamlanana ya da maksimum iterasyon yapılana kadar iterasyon yapan dış for döngüsü
            count = 0
            for j in range(len(train_data)): #Eğitim datası için sırayla işlemleri yaparak ağırlıkları güncelleyen iç for döngüsü
                with_bias = np.insert(train_data[j].data,self.dim,1) #terimin sonuna bias (1) eklenir.
                dot_product = with_bias.T.dot(self.Weights) #Ağırlık vektörünün transpozu ile noktasal çarpımı alınır.
                y = self.activation(dot_product) #Sonuç aktıvasyon fonksiyonuna sokulur.
                error = train_data[j].classNum - y #Aktivasyon çıkışı ile datanın bulunduğu sınıfın farkı error olarak esaplanır
                if error == 0: #Error 0 ise her iterasyonun başında sıfırlanan count değeri 1 arttırılır
                    count += 1
                self.Weights = self.Weights + self.c * error * with_bias #Ağrlıklar errora, öğrenme hızına ve giriş vektörüne bağlı olarak güncellenir
            if count == len(train_data): #İterasyon içindeki her error 0 ise count train data boyutuna eşit olur, bu öğrenmenin yeterli olduğunu artık ağırlıkların güncellenmeyeceğini gösterir.
                print("Weights: ")
                print(self.Weights)
                print("iteration number: ")
                print(i+1)
                return True              #Son ağırlık ve yapılmış iteerasyon sayısı bastırılır, fonksiyon sonlandırılır.
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
    
    """Data additional codesda bulunan linearly_seperable_data_creator.py
    yardımıyla ilgili kodda belirlediğimiz yöntem ile oluşturuldu, daha sonra buraya eklendi."""

    train_data = [data_with_class([[ -7.],[  8.],[  4.],[ -2.],[ -9.]],1),
    data_with_class([[  3.],[  9.],[  5.],[ -5.],[ -9.]],1),
    data_with_class([[ -5.],[  4.],[-10.],[ -5.],[  3.]],1),
    data_with_class([[ -9.],[  0.],[  3.],[  7.],[  0.]],1),
    data_with_class([[ -2.],[  5.],[  5.],[  5.],[ -5.]],1),
    data_with_class([[ -7.],[ -8.],[  2.],[  7.],[  6.]],1),
    data_with_class([[ -4.],[ -6.],[ -3.],[ -5.],[  5.]],1),
    data_with_class([[  8.],[ -5.],[  6.],[ -5.],[ -1.]],1),
    data_with_class([[ -1.],[ -2.],[  5.],[  8.],[  2.]],1),
    data_with_class([[  2.],[ -5.],[  6.],[  8.],[  3.]],1),
    data_with_class([[ -8.],[ -2.],[  3.],[ -4.],[ -3.]],1),
    data_with_class([[ -8.],[  6.],[ -8.],[  6.],[  4.]],1),
    data_with_class([[  9.],[  8.],[  1.],[ -6.],[ -5.]],1),
    data_with_class([[-10.],[  6.],[ -5.],[ -6.],[ -4.]],-1),
    data_with_class([[  8.],[  9.],[ -3.],[  7.],[  1.]],-1),
    data_with_class([[  2.],[ -1.],[  8.],[ -7.],[-10.]],-1),
    data_with_class([[ -7.],[ -4.],[ -4.],[  5.],[  6.]],-1),
    data_with_class([[ -3.],[  4.],[ -9.],[  8.],[  7.]],-1),
    data_with_class([[  5.],[  5.],[  0.],[ -9.],[ -6.]],-1),
    data_with_class([[ -7.],[ -5.],[  2.],[  6.],[  1.]],-1),
    data_with_class([[ -4.],[ -1.],[ -1.],[  1.],[  1.]],-1),
    data_with_class([[ -9.],[  9.],[ -9.],[ -7.],[ -4.]],-1),
    data_with_class([[ -7.],[  2.],[ -7.],[-10.],[ -2.]],-1),
    data_with_class([[ -4.],[ -1.],[ -1.],[ -2.],[ -2.]],-1),
    data_with_class([[ -1.],[  9.],[ -1.],[-10.],[-10.]],-1)]

    test_data = [data_with_class([[ -4.],[ -3.],[  3.],[ -3.],[  1.]],1),
    data_with_class([[  7.],[  3.],[  1.],[ -8.],[  1.]],1),
    data_with_class([[ -8.],[  1.],[  2.],[  2.],[  6.]],1),
    data_with_class([[ -9.],[  4.],[  9.],[  6.],[  8.]],1),
    data_with_class([[  9.],[ -4.],[  4.],[ -6.],[  5.]],1),
    data_with_class([[  9.],[ -4.],[  4.],[ -6.],[  5.]],1),
    data_with_class([[ -6.],[ -1.],[  9.],[ -6.],[  4.]],1),
    data_with_class([[ -2.],[ -2.],[  6.],[  3.],[-10.]],-1),
    data_with_class([[  7.],[  0.],[ -5.],[ -3.],[ -8.]],-1),
    data_with_class([[  9.],[ -1.],[ -6.],[ -3.],[  2.]],-1),
    data_with_class([[  2.],[ -1.],[ -6.],[  3.],[ -5.]],-1),
    data_with_class([[  9.],[  0.],[ -3.],[ -5.],[ -5.]],-1),
    data_with_class([[ -6.],[ -3.],[  4.],[  8.],[ -2.]],-1),
    data_with_class([[ -6.],[  1.],[ -9.],[  7.],[  4.]],-1),
    data_with_class([[ -6.],[-10.],[  0.],[  0.],[  2.]],-1)]

    p1 = perceptron(5,0.01,5000)

    """shuffled_data = random.sample(train_data, len(train_data))
    p1.train(shuffled_data)"""

    train = p1.train(train_data)
    if train:
        p1.test(test_data)
    else:
        print("data will not test")

