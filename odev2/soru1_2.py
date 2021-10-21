import numpy as np
import random
import math
from math import sqrt
from matplotlib import pyplot as plt
#Datayı oluşturan fonksiyon
def create_data():
    #Data dosyası açılır ve satırlardan liste oluşturulur
    file = open('iris.data','r')
    lines = file.readlines()
    #son satır data içermediğnden çıkarılır
    lines = lines[0:len(lines)-1]
    #Targets
    #setosa = [1,0,0]
    #versicolor = [0,1,0]
    #virginica = [0,0,1]
    data = []
    input_size = 4
    #Her satırdaki data targetına göre data listesine ekleniyor
    for i in range(len(lines)):
        if i < 50:
            target = np.array([1,0,0])
        elif i < 100:
            target = np.array([0,1,0])
        else:
            target = np.array([0,0,1])
        lines[i] = lines[i].split(",")
        tmp_in = np.zeros(input_size)    
        for j in range(input_size):
            tmp_in[j] = float(lines[i][j])
        tmp_in = np.reshape(tmp_in,(-1,1))
        data.append(data_with_class(tmp_in,target))
    #Nihai data döndürülüyor
    return data
#a parametresi sigmoid ve türevinde katsayı olarak kullanılıyor (f(x) = sgm(a*x))
def sigmoid(x,a = 1):
        return 1 / (1 + math.exp(-x*a))
def sigmoid_derivative(x,a = 1):
        return a * math.exp(-a*x) / (1 + math.exp(-x*a))**2
#Tek bir nöronu modelleyen class
class neuron(object):
    def __init__(self,imput, weights): 
        self.imput = imput
        self.weights = weights
    #Giriş vektörünü ve ağırlıkların nokta çarpımını döndüren metod
    def dp(self):
        return self.imput.dot(self.weights)
    #Nokta çarpımını sigmoide sokan metod
    def out(self):
        self.y = sigmoid(self.dp(),5)
        return self.y
#Bir katmanı modelleyen class yapısı, neuron classı içinde kullanılıyor 
class hidden_layer(object):
    #init metodu, weighsi matris olarak alır, nörın sayısını alır, nöronları tutacağı boş bir liste oluşturur
    def __init__(self,imput,weights,neuron_count):
        self.imput = imput
        self.weights = weights
        self.neuron_count = neuron_count 
        self.neurons = [] 
    #İleri yayılım metodu
    def ffw(self):
        #Giriş vektörünün sonuna bias eklenir
        self.with_bias = np.insert(self.imput,len(self.imput),1)
        #Katman çıkış vektörü sıfırlardan oluşturulur
        self.out = np.zeros(self.neuron_count)
        #Nöron sayısı kadar dönen for döngüsü, nöron oluşturulur ve çıkışı katman çıkışı listesine eklenir
        for i in range(self.neuron_count):
            self.neurons.append(neuron(self.with_bias,self.weights[i][:]))
            self.out[i] = self.neurons[i].out()           
        return self.out
    #Daha sonra kullanılabileceği için nöron listesini döndüren metod
    def Neurons(self):
        return self.neurons
#Çok katmanlı algılayıcı classı
class ml_per(object):
    #init metodu, öğrenme hızı ve maksimum iterasyon sayısını alır
    def __init__(self,c,iteration):
        self.iteration = iteration
        self.c = c
        self.weights = []
    #Eğitim metodu, eğitilecek datayı, algılayıcı şeklini ve momentum katsayısını alır
    """Algılayıcı şekli array tipindedir, arrayin ilk elemanı giriş sayısını, son elemanı öıkış sayısını,
    aradaki elemanlar aradaki katmanların nöron sayısını belirler. Bu sayede katman sayısı ve katmanlardaki nöron
    sayısı iateğe göre değiştirilebilir."""
    def train(self,train_data,structure,momentum):
        #Her data için çıkıştaki error vektörünü tutar
        error_vectors = np.zeros((structure[len(structure)-1],len(train_data)))
        #Her data için çıkıştaki error vektöründen hesaplanan ortalama hatayı tutar
        mean_errors = []
        self.weights_difference = []
        #Ağırlıklar olusturuluyor
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(0,sqrt(1/structure[i]),(structure[i+1],structure[i]+1)))
            self.weights_difference.append(np.zeros((structure[i+1],structure[i]+1)))
        #Iterasyonları yapan for döngüsü
        stop = 0
        for a in range(self.iteration):
            mean_errors = np.zeros(len(train_data))
            #Iterasyon içinde her datayı eğiten for döngüsü
            for j in range(len(train_data)):
                #Geriye yayılımda kullanılacak parametreleri tutacak boş listeler
                all_derivatives = []
                derivatives = []
                last_derivatives = []
                all_gradients = []
                gradients = []
                last_gradients = []
                #Başlangıçta ilk data out olarak tanımlanıyor
                out = train_data[j].data 
                #Geriye yayılımda kullanmak için her katmandaki nöronları ve katman çıkışlarını tutan boş listeler
                all_neurons = []
                all_outs = []
                all_outs.append(out)
                #İLERİ YAYILIM
                #İleri yayılım algoritası yapılan for döngüsü
                for k in range(len(structure)-1):
                    #katman oluşturukuyor 
                    layer = hidden_layer(out,self.weights[k],structure[k+1]) 
                    #Katman girişleri ağırlıklar ile çarpılıp sigmoide sokuluyor
                    out = layer.ffw()
                    #Daha sonra kullanılmak için her çıkış ve tüm nöronlar kaydediliyor
                    all_outs.append(out)
                    all_neurons.append(layer.Neurons())
                #Çıkıştaki hata ve ortalama hata hesaplanıyor            
                error = train_data[j].target - out
                Error = error.dot(np.transpose(error))/2
                #Tüm datalar için error vektörleri ve ortalam hatalar kaydediliyor
                mean_errors[j] = Error
                error_vectors[:,j] = error
                #GERİYE YAYILIM
                #Son katmanın girişlerinin türevleri hesaplanıyor
                for i in range(structure[len(structure)-1]):
                    last_derivatives.append(sigmoid_derivative(all_neurons[len(structure)-2][i].dp(),5))
                #Son katmanın gradyanları hesaplanıyor ve tüm gradyanlar listesine ekleniyor
                last_gradients = last_derivatives * error
                all_gradients.append(last_gradients)
                #Son katmmandan önceki katmanların türevlerini ve sonrasında gradyanlarını bulunup kaydediliyor
                for i in range(len(structure)-2):
                    for j in range(structure[len(structure)-2-i]):
                        derivatives.append(sigmoid_derivative(all_neurons[len(structure)-3-i][j].dp(),5))
                    all_derivatives.append(derivatives)
                    #Gradyanın bulunması için sıradaki ağırlık matrisinden bias sütunu çıkarılıyor ve transpozu alınıyor
                    w = (self.weights[len(structure)-2-i])
                    without_bias = np.transpose(w[:,0:w.shape[1]-1]) 
                    transpose = np.transpose(all_gradients[i])
                    #Gradyan vektörü hesaplanıyor
                    gradients = (without_bias).dot(transpose)*derivatives
                    all_gradients.append(gradients)
                    #For içinde geçici olarak kullanılan listeler sonraki adım için boşaltılıyor
                    derivatives = []
                    gradients = []
                #Çıkışlar biassız olarak kaydedildiğinden bias ekleniyor
                for i in range(len(all_outs)-1):
                    all_outs[i] = np.insert(all_outs[i],structure[i],1)
                last_gradients = np.reshape(last_gradients,(-1,1))
                #Son ağırlıklar güncelleniyor
                temp_weights = self.weights[len(structure)-2] 
                self.weights[len(structure)-2] = self.weights[len(structure)-2] + (((last_gradients)*(all_outs[len(structure)-2])*self.c) + (self.weights_difference[len(structure)-2]*momentum))
                self.weights_difference[len(structure) - 2] = self.weights[len(structure)-2] - temp_weights
                #Son ağırlık matrisinden önceki ağırlıklar güncelleniyor
                for i in range(len(structure)-2):
                    all_gradients[i+1] = np.reshape(all_gradients[i+1],(-1,1))
                    #Ağırlıklar, ağırlık farkı bulmak için kullanılacağından yedekleniyor
                    temp_weights = self.weights[len(structure)-3-i]
                    self.weights[len(structure)-3-i] = self.weights[len(structure)-3-i] + (all_gradients[i+1])*((((all_outs[len(structure)-3-i]))))*self.c + (self.weights_difference[len(structure)-3-i]*momentum)
                    #Son ağırlıkları güncellerken input vektörü kullanılıyor, np.matrix np.array uyuşmazlığı olduğundan burada matrix çok boyutlu arraye döndürülüyor
                    self.weights[len(structure)-3-i] = np.squeeze(np.asarray(self.weights[len(structure)-3-i]))
                    #Ağırlık farkı momentum terimiyle kullanılacağından hesaplanıyor
                    self.weights_difference[len(structure)-3-i] = self.weights[len(structure)-3-i] - temp_weights
            #Durdurma kriteri için iki iterasyon sonundaki ortalama hata arasındaki fark hesaplanıyor
            stop = abs(np.mean(mean_errors) - stop)
            #print(a+1) -iterasyon sayıyor
            #print(out) -ağ çıkış vektörünü basıyor
            #Durdurma bölgesi
            if stop <= 1e-2 and a>50 :
                print("iteration number: ")
                print(a+1)
                return True
            if a == self.iteration:
                print("İteration number:",i+a)
                print("Maximum iteration done.")
                return False
    #Test metodu
    def test(self,test_data,structure):
        true = 0
        false = 0
        for j in range(len(test_data)):
            #Başlangıçta ilk data out olarak tanımlanıyor
            out = test_data[j].data 
            #İleri yayılım algoritası yapılan for döngüsü
            for k in range(len(structure)-1):
                #katman oluşturukuyor 
                layer = hidden_layer(out,self.weights[k],structure[k+1]) 
                #Katman girişleri ağırlıklar ile çarpılıp sigmoide sokuluyor
                out = layer.ffw()
            #Error vekötünün mutlak değeri en büyük elemanı error olarak atanıyor, epsilon değerinden küçük ise doğru sayılacak  
            error = max(abs(test_data[j].target - out)) 
            #print(error)
            if error <= 0.1:
                true += 1
            else:
                false += 1
        print("Number of true data:",true)
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
    #random.shuffle(train_data)
    test_data = data[30:50] + data[80:100] + data[130:150]
    #Çok katmanlı algılayıcı oluşturuluyor
    mlp = ml_per(0.05,3000)
    #Ağ yapısı
    structure = [4,20,10,10,3]
    #Momentum terimi
    momentum = 0.2
    #Train yapılıyor, başarılı olursa teste sokuluyor.
    train = mlp.train(train_data,structure,momentum)
    if train:
        mlp.test(test_data,structure)
    else:
        print("Data will not be tested.")



