""" Yapay Sinir Agları
    Odev 3 - Soru 1
    Ramazan Umut Aktaş 040190762
    Recep Salih Yazar  040170066""" 
import numpy as np
import random
import math
from matplotlib import pyplot as plt
import pickle
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

#Kohonen ağını ve ilgili fonksiyonları içeren class
class koh(object):
    def __init__(self,dim,neuronsize):
        #Koh objesi oluşturulduğunda init fonksiyonu çalışır, obje parametre olarak giriş datasının boyutunu
        #ve neuronsize'ı alır. Neuronsize kare şeklinde kullanacağımız ağ katmanının bir kenarındaki nöron sayısıdır.
        #Örneğin; koh(4,7) => kullanılan verinin verinin her biri 4 boyutlu bir vektördür ve ağ 7*7 = 49 nörondan oluşacaktır.
        self.dim = dim
        self.neuronsize = neuronsize
        #Ağırlıklar -0.4 - 0.4 aralığında uniform dağılıma sahip olacak şekilde her nöron için oluşturulur.
        self.weights = np.random.uniform(-0.4,0.4,size=(self.neuronsize,self.neuronsize,self.dim))
        self.trues = [0]
        self.its = [0]
    #Eğitim verisi, test verisi ve son ağırlıkları çizdiren fonksiyon
    def pribt(self):
        print(self.trues)
        print(self.its)
    def plot_data(self,train_data,test_data):
        #Farklı renklerde basmak için veriler sınıflarına göre ayrı arraylere kaydediliyor
        train_set1 = np.zeros((int(len(train_data)/3),3))
        train_set2 = np.zeros((int(len(train_data)/3),3))
        train_set3 = np.zeros((int(len(train_data)/3),3))
        set1c = 0
        set2c = 0
        set3c = 0
        for i in range(len(train_data)):
            if train_data[i].target == -1:
                train_set1[set1c,:] = (train_data[i].data)
                set1c += 1
            elif train_data[i].target == 0:
                train_set2[set2c,:] = (train_data[i].data)
                set2c += 1
            else:
                train_set3[set3c,:] = (train_data[i].data)
                set3c += 1
        
        test_set1 = np.zeros((int(len(test_data)/3),3))
        test_set2 = np.zeros((int(len(test_data)/3),3))
        test_set3 = np.zeros((int(len(test_data)/3),3))
        set1c = 0
        set2c = 0
        set3c = 0
        for i in range(len(test_data)):
            if test_data[i].target == -1:
                test_set1[set1c,:] = (test_data[i].data)
                set1c += 1
            elif test_data[i].target == 0:
                test_set2[set2c,:] = (test_data[i].data)
                set2c += 1
            else:
                test_set3[set3c,:] = (test_data[i].data)
                set3c += 1

        #Figürler pluşturuluyor ve veriler çizdiriliyor     
        fig = plt.figure()
        fig.suptitle('Train Data', fontsize=16)
        
        fig2 = plt.figure()
        fig2.suptitle('Test Data', fontsize=16)
        
        fig3 = plt.figure()
        fig3.suptitle('Weights', fontsize=16)
        
        ax = fig.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        ax3 = fig3.add_subplot(111, projection='3d')
        
        ax.scatter(train_set1[:,0], train_set1[:,1], train_set1[:,2], c='r', marker='o')
        ax.scatter(train_set2[:,0], train_set2[:,1], train_set2[:,2], c='g', marker='o')
        ax.scatter(train_set3[:,0], train_set3[:,1], train_set3[:,2], c='b', marker='o')
        
        ax2.scatter(test_set1[:,0], test_set1[:,1], test_set1[:,2], c='r', marker='o')
        ax2.scatter(test_set2[:,0], test_set2[:,1], test_set2[:,2], c='g', marker='o')
        ax2.scatter(test_set3[:,0], test_set3[:,1], test_set3[:,2], c='b', marker='o')
        
        ax3.scatter(self.weights[:,:,0], self.weights[:,:,1], self.weights[:,:,2], c='b', marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        plt.show()
    #Giren veri ile tüm nöronlardaki ağırlıklar arasındaki farkın öklidyen normunu hesaplayıp kazanan nöronu bulan bestnatch fonksiyonu
    def bestmatch(self,data_point):
        #Tüm nöronlara olan mesafeler bir matrisde tutuluyor
        metrics = np.zeros((self.neuronsize,self.neuronsize))        
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                #Uzaklık hesaplanıyor
                metrics[i,j] = np.linalg.norm(data_point - self.weights[i,j,:])
        #Matristeki indexi alınıyor, buradan kazanan nöronun kordinatları bulunup return ettiriliyor
        index = np.argmin(metrics)
        cordinates = np.array([int(index / self.neuronsize), index % self.neuronsize])
        return cordinates
    """Ağıtlık Güncellemede kullanılan fonksiyonlar"""
    #Komşuluk fonksiyonu
    def neighbor(self,itnumber,neuronsize):
        sigma0 = (neuronsize / 2) 
        time_constant1 =  10000 / np.log(sigma0)
        sigma = sigma0 * np.exp(-itnumber / time_constant1)
        return sigma
    #Öğrenme hızı fonksiyonu
    def eta(self,itnumber):
        eta_zero = 0.15
        time_constant2 = 1000
        eta = eta_zero * np.exp(-itnumber / time_constant2) 
        return eta
    #İşbirliği kısmında kazanan nörona olan uzaklıklar kullanıldığından kazanan nöronun tüm nöronlara uzaklığını hesaplayan 
    #mesafe fonksiyonu
    def distance(self,winner,weights):
        distances = np.zeros((self.neuronsize,self.neuronsize))        
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                distances[i,j] = np.linalg.norm(winner - np.array([i,j]))
        return distances
    #Ağırlıkların güncellenmiş halini döndüren güncelleme fonksiyonu
    def update(self,weights,itnumber,neuronsize,winner,datapoint):
        #Ağırlık boyutunda yeni ağırlıklar için sıfırlardan oluşan matris oluşturuluyor
        updated = np.zeros((self.neuronsize,self.neuronsize,self.dim))
        distances = self.distance(winner,weights)
        sigma = self.neighbor(itnumber,neuronsize)
        for i in range(self.neuronsize):
            for j in range(self.neuronsize):
                updated[i,j,:] = weights[i,j,:] + (self.eta(itnumber) * np.exp(-(distances[i,j]**2)/2*(sigma**2))*(datapoint-weights[i,j,:]))
        return updated
    #Eğitim fonksiyonu
    def train(self,train_data):
        #İterasyon numarası
        itnumber = 1 
        #Ardışıl iterasyonlarda kazananların aynı olduktan sonra tekrar degismedigini kontrol eden degisken           
        control_count = 0       
        #Kazananlar sürekli aynı çıkmaya başladıktan sonra belirli sayıda yapılan iterasyonu kontrol eden counter 
        counter = 0
        #Her agırlık güncellemesinden sonra artan counter
        update_count = 0
        #Durdurma kriteri için eski kazananları ve yeni kazananları birlikte tutan matris
        winners_double = np.zeros((2,len(train_data),2))
        #Kazananları tutan matris
        winners = np.zeros((len(train_data),2))
        while(True):
            #Eski kazananlar winners_double'a kaydediliyor
            winners_double[0,:,:] = winners
            for x in range(len(train_data)):
                #kazanan nöron belirleniyor
                winner_neuron = self.bestmatch(train_data[x].data)
                #Kazananlara kaydediliyor
                winners[x,:] = winner_neuron
                #Kğırlıklar güncelleniyor (İşbirliği)
                self.weights = self.update(self.weights,update_count,self.neuronsize,winner_neuron,train_data[x].data)
                """Eta ve sigma fonksiyonunda belirtilen "itnumber" tüm datalar için işlem yapılması sayısını değil,
                #her bir ağırlık güncellemesini temsil ediyor, bu yüzden her ağırlık güncellemesinde update_count 1 arttırılır, 
                #update_count sigma ve eta fonksiyonunda kullanılır"""
                update_count += 1
            #Yeni kazananlar winners_double'a kaydediliyor
            winners_double[1,:,:] = winners
            #Eski ve yeni kazananlar karşılaştırılıyor, 3 defa birbirlerine eşit olduktan sonra
            if (winners_double[0,:,:] == winners_double[1,:,:]).all():
                #print("Winners are same with old winners")
                if control_count >= 3:
                    if control_count == 4:
                        print(itnumber)
                    counter += 1
                control_count += 1
            #Daha sonra nöron sayısının 30 katı kadar iterasyon yaptırılır
            if counter == (self.neuronsize**2)*30:
                break
            itnumber += 1
            #print("Iteration number: ",itnumber)
        #Eğitim verisi sınıflarına göre ayrılıyor
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
        #Kazanan nöronlardan aynı olanları bir defa bulunacak şekilde değiştiriliyor
        unique1 = np.unique(trainset1_winners,axis=0)
        unique2 = np.unique(trainset2_winners,axis=0)
        unique3 = np.unique(trainset3_winners,axis=0)
        filtered_u1 = []
        filtered_u2 = []
        filtered_u3 = []
        #Daha sonra veri kümesindeki belirli bir miktarın altındaki, yanlış sınıflamaya sebep olavilecek
        #veri sayısı için kazanan nöronlar ayıklaınr.
        counter = 0
        for j in unique1:
            for i in range(len(trainset1_winners)):
                if (j == trainset1_winners[i]).all():
                    counter += 1 
                if counter == self.neuronsize*6:
                    filtered_u1.append(j)
                    counter += 1
            counter = 0
        
        for j in unique2:
            for i in range(len(trainset2_winners)):
                if (j == trainset2_winners[i]).all():
                    counter += 1 
                if counter == self.neuronsize*6:
                    filtered_u2.append(j)
                    counter += 1
            counter = 0
        
        for j in unique3:
            for i in range(len(trainset3_winners)):
                if (j == trainset3_winners[i]).all():
                    counter += 1 
                if counter == self.neuronsize*6:
                    filtered_u3.append(j)
                    counter  += 1
            counter = 0
        unique3 = np.unique(trainset3_winners,axis=0)
        #Ayıklanmış kazanan nöronlar tek bir listeye kaydedilir
        self.filtered = [filtered_u1,filtered_u2,filtered_u3]
        return True
    def test(self,test_data):
        #Test fonksiyonunda az iterasyon yapıldığında farklı sınıflar için kazanan nöron aynı olabilir, bu sebeple test sonucu yanıltıcı
        #olabilir. Bunun çözümünü bulamadık, farklı test yöntemleri denedik. Ornegin her noron icin bir sınıf atama gibi.
        # Daha farklı sıkıntılar cıktı. En uygununun bu olduğunu düşündük. 
        true = 0
        false = 0
        for i in range(len(test_data)):
            controller = False
            #Test için kazanan nöron hesaplanıyor
            winner_neuron = self.bestmatch(test_data[i].data)
            for j in self.filtered[test_data[i].target+1]:
                if winner_neuron.all() == j.all() and controller == False:
                    true += 1
                    controller = True
            if controller == False:
                false += 1
        self.trues.append(true)

        print("True:",true)
        print("False:",false)
        return True



#Data ve targetı beraber tutan class
class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

if __name__ == '__main__':
    #data_create.py dosyasında oluşturulup kaydedilen "tursu" dosyasından train_Data ve test_data çekiliyor
    file = open("tursu",'rb')
    train_data = pickle.load(file)
    test_data = pickle.load(file)
    #Kohonen ağı oluşturuluyor
    #Eğitim datası karıştırılıyor
    random.shuffle(train_data)
    #Eğitim işlemi Uygulanıyor
    kohs = []
    koh1 = koh(3,5)
    kohs.append(koh1)
    koh2 = koh(3,5)
    kohs.append(koh2)
    koh3 = koh(3,5)
    kohs.append(koh3)
    koh4 = koh(3,5)
    kohs.append(koh4)
    koh5 = koh(3,5)
    kohs.append(koh5)    
        
    print("sa10000")
    for i in range(5):
        print(i)
        #Eğitim datası karıştırılıyor
        random.shuffle(train_data)
        kohs[i].train(train_data)
        #Test işlemi uygulanıyor
        kohs[i].test(test_data)
    #Eğitim verisi, test verisi ve son ağırlıklar çizdiriliyor
    #koh.plot_data(train_data,test_data)