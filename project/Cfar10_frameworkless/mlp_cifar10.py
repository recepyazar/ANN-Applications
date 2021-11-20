import numpy as np
import random
import math
from matplotlib import pyplot as plt
import pickle
#Pickle kütüphanesi yardımı ile kod ile aynı dizinde bulunan dosyanın çeken fonksiyon
def unpickle(file):
    with open(file, 'rb') as fo:
        object = pickle.load(fo, encoding='bytes')
    return object
#Unpickle fonksiyonu kullanılarak datanın import edildiği ve kullanıma hazır hale getirildiği fonksiyon
def create_data():
    #Tüm batch dosyaları yüklenir ve data ve target (x ve y) kısımları ayrılır.
    #Datanın yüklenmesi için tüm dosylar kod ile aynı dizinde olmalıdır!
    batch_1 = unpickle("data_batch_1")
    b1_data = batch_1[b'data']
    b1_targets = np.array(batch_1[b"labels"])
    batch_2 = unpickle("data_batch_2")
    b2_data = batch_2[b'data']
    b2_targets = np.array(batch_2[b"labels"])
    batch_3 = unpickle("data_batch_3")
    b3_data = batch_3[b'data']
    b3_targets = np.array(batch_3[b"labels"])
    batch_4 = unpickle("data_batch_4")
    b4_data = batch_4[b'data']
    b4_targets = np.array(batch_4[b"labels"])
    batch_5 = unpickle("data_batch_5")
    b5_data = batch_5[b'data']
    b5_targets = np.array(batch_5[b"labels"])
    test_batch = unpickle("test_batch")
    test_data = test_batch[b'data']
    test_targets = np.array(test_batch[b"labels"])
    #Train ve test için data ve targetlar birleştirilir
    train_data = np.concatenate((b1_data, b2_data, b3_data, b4_data, b5_data), axis = 0) 
    train_targets = np.concatenate((b1_targets, b2_targets, b3_targets, b4_targets, b5_targets), axis = 0)
    train_data_wc = []
    test_data_wc = []
    """Matrislerdeki değerler 0-255 arasında değer alabilen RGB degerleridir, her değer 255'e bölünerek 
    normalize edilir, bu şekilde tüm girişler 0-1 aralığında olmuş olur."""
    train_data = train_data / 255
    test_data = test_data / 255
    """Datasetin verilen halinde verilen resmin class değeri 0-9 aralıgında bir integer değerdir.
    Çok katmanlı alglayıcı ile çalışmak için, target (10,) boyutunda bir vektöre çevrilir. Class 
    numarası kaç ise o indeksi 1 yapılır. Diğer indeksler 0 olacaktır. Bununla beraber train data
    ve test data ödevde kullanılan data with class veri yapısı şeklinde tekrar oluşturulur."""
    for i in range(len(train_targets)+len(test_targets)):
        target = np.zeros((10))
        if i < len(train_targets):
            index = train_targets[i]
            target[index-1] = 1
            train_data_wc.append(data_with_class(train_data[i],target))
        else:
            index = test_targets[i-len(train_targets)]
            target[index-1] = 1
            test_data_wc.append(data_with_class(test_data[i-len(train_targets)],target))
    #Fonksiyon kullanıma hazır train ve test datasını return eder.
    return train_data_wc,test_data_wc
# Cross entropy loss fonksiyonu
def categorical_cross_entropy(actual, predicted):
    #Fonksiyon y ve yd değerlerini alıyor, tüm datalar için ve vektörlerdeki tüm indisler için bir toplama yapıyor
	sum_score = 0.0
	for i in range(len(actual)):
		for j in range(len(actual[i])):
            #1e-15 değeri oldukça küçük olduğundan sonucu pek etkilemezken log(0) çıkmasını engelliyor.
			sum_score += actual[i][j] * math.log(1e-15 + predicted[i][j])
	mean_sum_score = sum_score / len(actual)
	return -mean_sum_score
"""Aktivasyon fonksiyonları ve türevleri"""
#RELU fonksiyonu, sigmoide alternatif, aktivasyon fonksiyonu olarak kullanılabilir.
def relu(x):
    if x < 0:
        return 0
    else:
        return x
#RELU fonksiyonunun türevi
def relu_derivative(x):
    if x < 0:
        return 0
    else:
        return 1
#Softmax fonksiyonu, çıkış katmanında aktivasyon fonksiyonu olarak kullanılabiliyor.
def softmax(x,i):
    # x : Nöronların girişindeki dot product değerlerinin oluşturduğu vektör
    # i : Hesaplanan nöronun indisi
    return np.exp(x[i]-np.max(x))/np.exp(x-np.max(x)).sum()
#Softmax fonksiyonunun türevi
def softmax_derivative(x,i):
    return softmax(x,i)*(1-softmax(x,i))
#Sigmoid fonksiyonu, a parametresi sigmoid ve türevinde katsayı olarak kullanılıyor (f(x) = sgm(a*x))
def sigmoid(x,a = 1):
    return 1 / (1 + math.exp(-x*a))
#Sigmoid fonksiyonunun türevi
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
    #Nokta çarpımını aktivasyon fonksiyonuna sokan metod
    def out(self,neuron_number,dps,out_layer = False):
        """Aktivasyon fonksiyonları burada ne seçilmişse geriye yayılım kısmında da çıkış ve diğer katmanlar
        için aynı fonksiyonların türevleri seçilmelidir."""
        #Çıkış öncesi katmanların aktivasyon fonksiyonu
        if out_layer == False:
            self.y = relu(self.dp())
            #self.y = sigmoid(self.dp())
        #Çıkış katmanının aktivasyon fonksiyonu
        else:
            #self.y = relu(self.dp())
            self.y = sigmoid(self.dp())
            #self.y = softmax(dps,neuron_number)
        return self.y
#Bir katmanı modelleyen class yapısı, neuron classı içinde kullanılıyor 
class hidden_layer(object):
    #init metodu, weighsi matris olarak alır, nörın sayısını alır, nöronları tutacağı boş bir liste oluşturur
    def __init__(self,imput,weights,neuron_count,out_layer=False):
        self.imput = imput
        self.weights = weights
        self.neuron_count = neuron_count 
        self.neurons = []
        self.out_layer = out_layer 
    #Dropout fonksiyonu, verilen orana göre random numaralar üreterek kullanılmayacak nöronları belirler 
    def dropout(self,insize,rate):
        kill_count = int(insize*rate)
        kill_neurons = np.random.randint(0,insize,(kill_count))
        return kill_neurons
    #İleri yayılım metodu
    def ffw(self,drop = 0):
        #Giriş vektörünün sonuna bias eklenir
        self.with_bias = np.insert(self.imput,len(self.imput),1)
        #Katman çıkış vektörü sıfırlardan oluşturulur
        self.out = np.zeros(self.neuron_count)
        #Nöron sayısı kadar dönen for döngüsü, nöron oluşturulur ve çıkışı katman çıkışı listesine eklenir
        self.dot_products = []
        #Dropout için, kullanılmayacak nöronların ağırlıkları 0 yapılır.
        self.kill_neurons = self.dropout(self.weights.shape[0],drop)
        for i in self.kill_neurons:
            self.weights[i] = 0
        for i in range(self.neuron_count):
            neuro = neuron(self.with_bias,self.weights[i][:])
            self.neurons.append(neuro)
            self.dot_products.append(neuro.dp())
        for i in range(self.neuron_count):
            self.out[i] = self.neurons[i].out(i,self.dot_products,self.out_layer)         
        return self.out,self.dot_products
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
        self.weights_difference = []
        #Ağırlıklar olusturuluyor
        for i in range(len(structure)-1):
            self.weights.append(np.random.normal(0,math.sqrt(1/structure[i]),(structure[i+1],structure[i]+1)))
            self.weights_difference.append(np.zeros((structure[i+1],structure[i]+1)))
        #Iterasyonları yapan for döngüsü
        for a in range(self.iteration):
            print("Epoch number: ",a+1)
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
                all_dps = []
                all_outs.append(out)
                #İLERİ YAYILIM
                #İleri yayılım algoritası yapılan for döngüsü
                for k in range(len(structure)-1):
                    #katman oluşturukuyor 
                    if k == len(structure)-2:
                        layer = hidden_layer(out,self.weights[k],structure[k+1],out_layer=True)
                    else: 
                        layer = hidden_layer(out,self.weights[k],structure[k+1])
                    #Katman girişleri ağırlıklar ile çarpılıp sigmoide sokuluyor
                    out,dps = layer.ffw()
                    #out,dps = layer.ffw(0.1) 0.1 olasılıklı dropout için bu kod kullanılabilir
                    #Daha sonra kullanılmak için her çıkış ve tüm nöronlar kaydediliyor
                    all_outs.append(out)
                    all_dps.append(dps)
                    all_neurons.append(layer.Neurons())
                #Çıkıştaki hata ve ortalama hata hesaplanıyor            
                error = train_data[j].target - out
                #GERİYE YAYILIM
                #Son katmanın girişlerinin türevleri hesaplanıyor
                #Çıkış katmanının aktivasyon fonksiyonuna göre satır seçilebilir. 
                for i in range(structure[len(structure)-1]):
                    #last_derivatives.append(softmax_derivative(all_dps[-1],i))
                    last_derivatives.append(sigmoid_derivative(all_neurons[-1][i].dp()))
                    #last_derivatives.append(relu_derivative(all_neurons[-1][i].dp()))
                #Son katmanın gradyanları hesaplanıyor ve tüm gradyanlar listesine ekleniyor
                last_gradients = last_derivatives * error
                all_gradients.append(last_gradients)
                #Son katmmandan önceki katmanların türevlerini ve sonrasında gradyanlarını bulunup kaydediliyor
                for i in range(len(structure)-2):
                    for j in range(structure[len(structure)-2-i]):
                        derivatives.append(relu_derivative(all_neurons[len(structure)-3-i][j].dp()))
                        #derivatives.append(sigmoid_derivative(all_neurons[len(structure)-3-i][j].dp()))
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
            #print(a+1) -iterasyon sayıyor
            #print(out) -ağ çıkış vektörünü basıyor
            self.test_for_loss(train_data,structure)
            if a == self.iteration-1:
                print("İteration number:",i+a)
                print("Maximum iteration done.")
                return True
    #Test metodu
    def test(self,test_data,structure):
        true = 0
        false = 0
        cm = np.zeros((structure[-1],structure[-1]))
        for j in range(len(test_data)):
            #Başlangıçta ilk data out olarak tanımlanıyor
            out = test_data[j].data 
            #İleri yayılım algoritası yapılan for döngüsü
            for k in range(len(structure)-1):
                #katman oluşturukuyor
                if k == len(structure)-2:
                    layer = hidden_layer(out,self.weights[k],structure[k+1],out_layer=True)
                else: 
                    layer = hidden_layer(out,self.weights[k],structure[k+1])
                #Katman girişleri ağırlıklar ile çarpılıp sigmoide sokuluyor
                out,dps = layer.ffw()
            real = np.argmax(abs(test_data[j].target))
            predict =  np.argmax(abs(out))
            cm[real,predict] += 1
            if  real == predict:
                true += 1
            else:
                false += 1
        print("Accuracy:",true/len(test_data))
        print("Confussion Matrix: ",cm)
    def test_for_loss(self,data,structure):
        true = 0
        false = 0
        outs_of_data = []
        targets_of_data = []
        for j in range(len(data)):
            targets_of_data.append(data[j].target)
            #Başlangıçta ilk data out olarak tanımlanıyor
            out = data[j].data 
            #İleri yayılım algoritası yapılan for döngüsü
            for k in range(len(structure)-1):
                #katman oluşturukuyor
                if k == len(structure)-2:
                    layer = hidden_layer(out,self.weights[k],structure[k+1],out_layer=True)
                else: 
                    layer = hidden_layer(out,self.weights[k],structure[k+1])
                #Katman girişleri ağırlıklar ile çarpılıp sigmoide sokuluyor
                out,dps = layer.ffw()
            outs_of_data.append(out)       
            if np.argmax(abs(data[j].target)) == np.argmax(abs(out)):
                true += 1
            else:
                false += 1
        loss = categorical_cross_entropy(targets_of_data,outs_of_data)
        print("loss:",loss)
        accuracy = true/len(data)
        print("accuracy:",accuracy)
#Data ve targetı beraber tutan class
class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

if __name__ == '__main__':
    train_data, test_data = create_data()
    #Eğitim ve test dataları oluşturuluyor
    random.shuffle(train_data)
    #Çok katmanlı algılayıcı oluşturuluyor
    mlp = ml_per(0.001,20)
    #Ağ yapısı belirleniyor, 3072 giriş, 10 çıkış, sırayla katmanlarda 20,15,10 nöron.
    structure = [3072,20,15,10]
    #Momentum katsayısı
    momentum = 0.9
    #Train yapılıyor, sonrasında teste sokuluyor.
    train = mlp.train(train_data,structure,momentum)
    if train:
        mlp.test(test_data,structure)
    else:
        print("Data will not be tested.")





