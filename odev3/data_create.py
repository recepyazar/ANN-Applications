import numpy as np
import pickle

#Data ve targetı beraber tutan class
class data_with_class(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target

if __name__ == "__main__":     
    #Her bir veri grubu için ortalamalar belirleniyor
    set1_mean = np.array([3,3,3])
    set2_mean = np.array([5,-1,0])
    set3_mean = np.array([8,2,-1])
    #Her bir veri grubu için kovaryans matrisi belirleniyor
    set1_cov = [[0.1,0,0],[0,6,0],[0,0,5]]
    set2_cov = [[0.7,0,0],[0,0.7,0],[0,0,0.7]]
    set3_cov = [[2,0,0],[0,1,0],[0,0,4]]
    #Veri grupları eğitim ve test için oluşturuluyor
    train_set1 = np.random.multivariate_normal(set1_mean,set1_cov,150)
    train_set2 = np.random.multivariate_normal(set2_mean,set2_cov,150)
    train_set3 = np.random.multivariate_normal(set3_mean,set3_cov,150)

    test_set1 = np.random.multivariate_normal(set1_mean,set1_cov,50)
    test_set2 = np.random.multivariate_normal(set2_mean,set2_cov,50)
    test_set3 = np.random.multivariate_normal(set3_mean,set3_cov,50)
    #Eğitim ve test verileri sınıf bilgisi ile beraber bir veri tpipinde kaydediliyor.
    train_data = []
    test_data = []

    for i in range(450):
        if i < 150:
            train_data.append(data_with_class((train_set1[:][i]),-1))
        elif i < 300:
            train_data.append(data_with_class((train_set2[:][i-150]),0))
        else:
            train_data.append(data_with_class((train_set3[:][i-300]),1))

    for i in range(150):
        if i < 50:
            test_data.append(data_with_class((test_set1[:][i]),-1))
        elif i < 100:
            test_data.append(data_with_class((test_set2[:][i-50]),0))
        else:
            test_data.append(data_with_class((test_set3[:][i-100]),1))
    #Veriler ağ kodunda kullanılmak üzere pickle kütüphanesi yardımıyla "tursu" dosyasına kaydediliyor
    f = open("tursu","wb") 
    pickle.dump(train_data,f)     
    pickle.dump(test_data,f)      