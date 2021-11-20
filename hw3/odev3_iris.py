""" Yapay Sinir Agları
    Odev 3 - Soru 2 (Iris data)
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar 040170066
    --Aynı kısımların yorumları soru1'de mevcut""" 
import numpy as np
import random
import math
from matplotlib import pyplot as plt
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
#Kohonen ağını ve ilgili fonksiyonları içeren class
class koh(object):
    def __init__(self,dim,neuronsize):
        #Koh objesi oluşturulduğunda init fonksiyonu çalışır, obje parametre olarak giriş datasının boyutunu
        #ve neuronsize'ı alır. Neuronsize kare şeklinde kullanacağımız ağ katmanının bir kenarındaki nöron sayısıdır.
        #Örneğin; koh(4,7) => kullanılan verinin verinin her biri 4 boyutlu bir vektördür ve ağ 7*7 = 49 nörondan oluşacaktır.
        self.dim = dim
        self.neuronsize = neuronsize
        #Ağırlıklar -0.4 - 0.4 aralığında uniform dağılıma sahip olacak şekilde her nöron için oluşturulur.
        self.weights = np.random.uniform(-0.1,0.1,size=(self.neuronsize,self.neuronsize,self.dim))

    def bestmatch(self,data_point):
        metrics = np.zeros((self.neuronsize,self.neuronsize))        
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                metrics[i,j] = np.linalg.norm(data_point - self.weights[i,j,:])
        index = np.argmin(metrics)
        cordinates = np.array([int(index / self.neuronsize), index % self.neuronsize])
        return cordinates

    def neighbor(self,itnumber,neuronsize):
        sigma0 = (neuronsize / 2) 
        time_constant1 =  1000 / np.log(sigma0)
        sigma = sigma0 * np.exp(-itnumber / time_constant1)
        return sigma

    def eta(self,itnumber):
        eta_zero = 0.15
        time_constant2 = 1000
        eta = eta_zero * np.exp(-itnumber / time_constant2) 
        return eta

    def distance(self,winner,weights):
        distances = np.zeros((self.neuronsize,self.neuronsize))        
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                distances[i,j] = np.linalg.norm(winner - np.array([i,j]))
        return distances

    def update(self,weights,itnumber,neuronsize,winner,datapoint):
        updated = np.zeros((self.neuronsize,self.neuronsize,self.dim))
        distances = self.distance(winner,weights)
        sigma = self.neighbor(itnumber,neuronsize)
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                updated[i,j,:] = weights[i,j,:] + (self.eta(itnumber) * np.exp(-(distances[i,j]**2)/2*(sigma**2))*(np.transpose(datapoint)-weights[i,j,:]))
        return updated

    def train(self,train_data):
        itnumber = 1
        control_count = 0
        counter = 0
        update_count = 0
        winners_double = np.zeros((2,len(train_data),2))
        winners = np.zeros((len(train_data),2))
        while(True):
            winners_double[0,:,:] = winners
            for x in range(len(train_data)):
                winner_neuron = self.bestmatch(train_data[x].data)
                winners[x,:] = winner_neuron
                self.weights = self.update(self.weights,update_count,self.neuronsize,winner_neuron,train_data[x].data)
                update_count += 1
            winners_double[1,:,:] = winners
            if (winners_double[0,:,:] == winners_double[1,:,:]).all():
                #print("Winners are same with old winners")
                if control_count >= 3:
                    if control_count == 4:
                        print(itnumber)
                    counter += 1
                control_count += 1
            if counter == (self.neuronsize**2)*50:
                break
            itnumber += 1
            #print("Iteration number: ",itnumber)
        
        trainset1_winners = np.array([])
        trainset2_winners = np.array([])
        trainset3_winners = np.array([])
        for x in range(len(train_data)):
            winner_neuron = self.bestmatch(train_data[x].data)
            if train_data[x].target == -1:
                trainset1_winners = np.append(trainset1_winners,winner_neuron)
            elif train_data[x].target == 0:
                trainset2_winners = np.append(trainset2_winners,winner_neuron)
            else:
                trainset3_winners = np.append(trainset3_winners,winner_neuron)
        trainset1_winners = np.reshape(trainset1_winners,(int(len(trainset1_winners)/2),2))
        trainset2_winners = np.reshape(trainset2_winners,(int(len(trainset2_winners)/2),2))
        trainset3_winners = np.reshape(trainset3_winners,(int(len(trainset3_winners)/2),2))
        filtered_u1 = []
        filtered_u2 = []
        filtered_u3 = []
        unique1 = np.unique(trainset1_winners,axis=0)
        counter = 0
        for j in unique1:
            for i in range(len(trainset1_winners)):
                if (j == trainset1_winners[i]).all():
                    counter += 1 
                if counter == 9:
                    filtered_u1.append(j)
                    counter += 1
            counter = 0
        unique2 = np.unique(trainset2_winners,axis=0)
        for j in unique2:
            for i in range(len(trainset2_winners)):
                if (j == trainset2_winners[i]).all():
                    counter += 1 
                if counter == 9:
                    filtered_u2.append(j)
                    counter += 1
            counter = 0
        unique3 = np.unique(trainset3_winners,axis=0)
        for j in unique3:
            for i in range(len(trainset3_winners)):
                if (j == trainset3_winners[i]).all():
                    counter += 1 
                if counter == 9:
                    filtered_u3.append(j)
                    counter  += 1
            counter = 0
        unique3 = np.unique(trainset3_winners,axis=0)
        self.uniques = [unique1,unique2,unique3]
        self.filtered = [filtered_u1,filtered_u2,filtered_u3]
        return True
    def test(self,test_data):
        true = 0
        false = 0
        for i in range(len(test_data)):
            controller = False
            winner_neuron = self.bestmatch(test_data[i].data)
            for j in self.filtered[test_data[i].target+1]:
                if winner_neuron.all() == j.all() and controller == False:
                    true += 1
                    controller = True
            if controller == False:
                false += 1

        print("True:",true)
        print("False:",false)
        return True
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
    kohs = []
    koh1 = koh(4,5)
    kohs.append(koh1)
    koh2 = koh(4,5)
    kohs.append(koh2)
    koh3 = koh(4,5)
    kohs.append(koh3)
    koh4 = koh(4,5)
    kohs.append(koh4)
    koh5 = koh(4,5)
    kohs.append(koh5)    
        
    print("sa100")
    for i in range(5):
        print(i)
        #Eğitim datası karıştırılıyor
        random.shuffle(train_data)
        kohs[i].train(train_data)
        #Test işlemi uygulanıyor
        kohs[i].test(test_data)