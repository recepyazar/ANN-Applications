#Recep Salih Yazar 040170066 ------------- Ramazan Umut Aktaş 040190762



import numpy as np
import random
import math
import pickle
from matplotlib import pyplot as plt

class rln(object):
    def __init__(self,x,teta,v,omega,epoch):
        self.x = x
        self.teta = teta
        self.v = v
        self.omega = omega
        self.st_vrbls = np.array([x,teta,v,omega])
        self.epoch = epoch
        self.ase_ws = np.random.normal(0,0.05,size=4)
        self.ace_ws = np.random.normal(0,0.05,size=4)
        self.init_parameters()
    def init_parameters(self):
        self.alfa = 0.2 #Bu değeri değiştirince bir şey değişmiyor
        self.beta = 1e-3
        self.ateb = 0.5 #Bu değeri değiştirince bir şey değişmiyor
        self.gama = 0.95 #Sapmalar bu değerin artışı azalışı ile orantılı ancak aşağı bir değerin altına değiştirmiyor.
        self.lmbda = 0.8 # Bu değerin artışı NaN olmasına yol açabiliyor ama azalışı pek etkilemiyor.
        self.delta = 0.15 #lmbda ile aynı davranış
        self.elig  = np.full(4,0.02) #+ - olması konumun değişim yönünü etkiliyor.
        self.g = 9.8
        self.mc = 1
        self.m = 1
        self.l = 0.5
        self.uc = 5e-4
        self.up = 2e-6
        self.m1 = self.mc + self.m
        self.old_pt = 0
        self.pick = np.full(4,0)
        self.it_number=[]
        self.x_plot=[]
        self.v_plot=[]
        self.teta_plot=[]
        self.omega_plot=[]
        self.rew1,self.rew2 = [],[]
    def f1_2(self,F):
        f1 = self.m * self.l * (self.omega**2) * math.sin(self.teta) - self.uc * np.sign(self.v) #signum'un içini -1 ile çarpınca bir şey değişmiyor.
        f2 = -self.g * math.sin(self.teta) + math.cos(self.teta) * ((-F-f1)/self.m1) - ((self.up * self.omega)/(self.m*self.l))
        return f1, f2
    def update_values(self,F):
        f1, f2 = self.f1_2(F)
        x = self.x + self.beta * self.v
        teta = self.teta + self.beta * self.omega
        v = self.v + self.beta * ((F+(f1-self.m*self.l*f2*math.cos(self.teta))/self.m1))
        omega = self.omega + self.beta * (f2/(self.l*((4/3)-((self.m*(math.cos(self.teta)**2))/(self.m1)))))
        self.pick = self.lmbda*self.pick + (1-self.lmbda)*self.st_vrbls

        self.x_plot.append(self.x)
        self.v_plot.append(self.v)
        self.teta_plot.append(self.teta)
        self.omega_plot.append(self.omega)
        if self.reward() == 1:
            self.x = x
            self.teta = teta
            self.v = v
            self.omega = omega
            self.st_vrbls = np.array([self.x,self.teta,self.v,self.omega])
        else:
            self.x = 0
            self.teta = 0
            self.v=0
            self.omega =0
            self.st_vrbls = np.array([self.x,self.teta,self.v,self.omega])
    def reward(self):
        if (self.x < 2.4 and self.x > -2.4) and (self.teta < 12 and self.teta > -12):
            return 1
        else:
            return -1

    def predict(self):
        tmp_pt = self.ace_ws.dot(self.st_vrbls).T
        return (math.tanh(tmp_pt)+1)/2 #Burda tmp_pt 0 ile çarpılınca bir şey değişmiyor. Kontrol et!!!
    def noise(self,a=0.1):
        return a*(np.random.normal(0,0.01))
    def process(self):
        for i in range(self.epoch):
            self.it_number.append(i)
            pt = self.predict()
            rt = self.reward()
            self.rew1.append(rt)
            new_rt = rt + self.gama * pt - self.old_pt
            self.rew2.append(new_rt)
            self.old_pt = pt
            self.ace_ws = self.ace_ws + self.ateb*new_rt*self.pick
            dp = self.ase_ws.dot(self.st_vrbls.T)
            y=np.tanh(3*(dp + self.noise()))
            self.ase_ws = self.ase_ws + self.alfa*new_rt*self.elig 
            self.elig = self.delta*self.elig + (1-self.delta)*y*self.st_vrbls
            self.F = 10*y
            self.update_values(self.F)
            
    def plotcu(self):
        fig1, axs = plt.subplots(2,2)
        fig1.suptitle('Durum Değişkenlerinin Zamana Göre Değişimi')
        axs[0,0].plot(self.it_number, self.x_plot)
        axs[0,0].set_title('Konum vs İterasyon')
        axs[0,0].set_xlabel('İterasyon Sayısı')
        axs[0,0].set_ylabel('Konum')
        axs[0,1].plot(self.it_number, self.v_plot)
        axs[0,1].set_title('Hız vs İterasyon')
        axs[0,1].set_xlabel('İterasyon Sayısı')
        axs[0,1].set_ylabel('Hız')
        axs[1,1].plot(self.it_number, self.teta_plot)
        axs[1,1].set_title('Açı vs İterasyon')
        axs[1,1].set_xlabel('İterasyon Sayısı')
        axs[1,1].set_ylabel('Açı')
        axs[1,0].plot(self.it_number, self.omega_plot)
        axs[1,0].set_title('Açısal Hız vs İterasyon')
        axs[1,0].set_xlabel('İterasyon Sayısı')
        axs[1,0].set_ylabel('Açısal Hız')

        fig2, axs1 = plt.subplots(2,1)
        fig2.suptitle('Hata, Beklenti Hatası vs. İterasyon Sayısı')
        axs1[0].plot(self.it_number,self.rew1)
        axs1[0].set_title('Hata vs İterasyon')
        axs1[0].set_xlabel('İterasyon Sayısı')
        axs1[0].set_ylabel('Hata')
        axs1[1].plot(self.it_number,self.rew2)
        axs1[1].set_title('Beklenen Hata vs İterasyon')
        axs1[1].set_xlabel('İterasyon Sayısı')
        axs1[1].set_ylabel('Beklenen Hata')

        plt.plot()
        plt.show()

if __name__ == "__main__":
    rln = rln(0.55,0.1,0.22,0.66,4000)
    rln.process()
    rln.plotcu()