import numpy as np
import random
import math
from math import sqrt
from matplotlib import pyplot as plt

def create_data():
    #Sistem başlangıcı yolarak y(0) ve y(1) değerleri 0 olarak alınıyorl sonuçlar listesine ekleniyor
    results=[0,0]
    #Ağ girişi için kullanılacak 2 boyutlu vektörlerin listesi, [y(0), y(1)] vektörü ekleniyor
    in_vectors = [np.matrix([[0],[0]])]
    #Datanın kaydedileceği boş liste
    data = []
    #Fonksiyonu y(2) den y(102)ye kadar hesaplayıp kaydeden döngü
    for i in range(2,132,1):
        ys1 = results[i-1]
        ys2 = results[i-2]
        result = billings(ys1,ys2)
        results.append(result)
        in_vectors.append(np.matrix([[results[i-1]],[results[i]]]))
        data.append(data_with_class(in_vectors[i-2],results[i]))
    return data
#Çizdirme işlemleri yapılan fonksiyon
def plot_results(data,autonom,mlp_out):
    x = range(130)
    autonom = autonom[2:]
    data_array = []
    for i in range(len(data)):
        data_array.append(data[i].target)
    plt.plot(x,data_array)
    plt.plot(x,mlp_out,'r')
    plt.title("Orijinal Veri ve Ağ Çıktısı")
    plt.show()
    plt.plot(x,data_array)
    plt.plot(x,autonom,'g')
    plt.title("Orijinal Veri ve 85. veriden sonraki otonom davranış")
    plt.show()
#Göürültü eklemek için fonksiyon, a parametresi 10^a çarpanı olarak kullanılıyor
def noise(a=4):
    return np.random.normal(0,1)*(math.pow(10,-1*a))*np.random.randint(0,5)
#Sistemi modelleyen fonksiyon, indeks, bir önceki çıktısı ve iki öceki çıktısını parametre olarak alıyor
def billings(ys1,ys2):
    y=((0.8-(0.5*np.exp(-1*(ys1**2))))*ys1) - ((0.3 + (0.9*np.exp(-1*(ys1**2))))*ys2) + 0.1*np.sin(math.pi*ys1) + noise()
    return y
#a parametresi tanh ve türevinde katsayı olarak kullanılıyor (f(x) = (a*x))
def tanh(x,a = 1):
        return math.tanh(a*x)
def tanh_derivative(x,a = 1):
        return a*(1-math.tanh(a*x)**2)

#Tek bir nöronu modelleyen class
class neuron(object):
    def __init__(self,imput, weights): 
        self.imput = imput
        self.weights = weights
    #Giriş vektörünü ve ağırlıkların nokta çarpımını döndüren metod
    def dp(self):
        return self.imput.dot(self.weights)
    #Nokta çarpımını tanhe sokan metod
    def out(self):
        self.y = tanh(self.dp())
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
                    #Katman girişleri ağırlıklar ile çarpılıp tanhe sokuluyor
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
                    last_derivatives.append(tanh_derivative(all_neurons[len(structure)-2][i].dp()))
                #Son katmanın gradyanları hesaplanıyor ve tüm gradyanlar listesine ekleniyor
                last_gradients = last_derivatives * error
                all_gradients.append(last_gradients)
                #Son katmmandan önceki katmanların türevlerini ve sonrasında gradyanlarını bulunup kaydediliyor
                for i in range(len(structure)-2):
                    for j in range(structure[len(structure)-2-i]):
                        derivatives.append(tanh_derivative(all_neurons[len(structure)-3-i][j].dp()))
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
    def test(self,data,structure):
        true = 0
        false = 0
        outs = [0,0] 
        for j in range(len(data)):
            if j <= 100:
                out = data[j].data
            #İleri yayılım algoritası yapılan for döngüsü
            for k in range(len(structure)-1):
                #katman oluşturukuyor 
                layer = hidden_layer(out,self.weights[k],structure[k+1]) 
                #Katman girişleri ağırlıklar ile çarpılıp tanhe sokuluyor
                out = layer.ffw()
            outs.append(out)
            print(out)
            error = data[j].target- out 
            if abs(error) <= 0.1:
                true += 1
            else:
                false += 1 
            if j >= 100:          
                out = np.array([outs[j+1],out])
        true -= 100
        print("Number of true data: {} / 30".format(true))  
        return outs
    def train_records(self,data,structure):
        outs = []
        for j in range(len(data)):
            out = data[j].data 
            for k in range(len(structure)-1):
                layer = hidden_layer(out,self.weights[k],structure[k+1]) 
                out = layer.ffw()
            outs.append(out)
        return outs


class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

if __name__ == '__main__':
    data = create_data()
    train_data = data[:100]
    mlp = ml_per(0.05,300)
    structure = [2,15,10,1]
    momentum = 0.2
    train = mlp.train(train_data,structure,momentum)
    if train:
        test = mlp.test(data,structure)
        cka_out = mlp.train_records(data,structure)
        plot_results(data,test,cka_out)

    else:
        print("Data will not be tested.")





