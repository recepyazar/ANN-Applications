********************************************************************************
EHB 420 - Final Projesi 
CIFAR10 VERİ SETİ SINIFLANDIRMA PROBLEMİ
Ramazan Umut Aktaş
Recep Salih Yazar
********************************************************************************
Proje dosyası içerisinde proje raporu, README dosyası ve iki kısım için iki 
farklı klasör mevcuttur.

Bu proje iki kısımdan oluşmaktadır. Birinci kısımda sınıflandırma problemi
hazır bir veriden öğrenme kütüphanesi olan keras kullanılmış olup CNN ve MLP 
olmak üzere iki farklı model kullanılmıştır. Bunun için Cifar10_keras 
klasöründe cnn ve mlp olmak üzere iki adet ipynb dosyası mevcuttur, bu kısımda
veri seti keras kütüphanesi aracılığıyla internetten otomatik 
olarak indirilmektedir.

İkinci kısım ise herhangi bir veriden öğrenme kütüphanesi kullanılmadan
bizim tarafımızdan yazılan bir kod ile oluşturulmuştur. Bu kodda MLP yapısı
modellenmiştir, kod Cfar10_kutuphanesiz klasörünün içindedir. Kodun veri setini
import edebilmesi için ilgili klasörün içinde py dosyasının yanında aşağıdaki 
linkten indirilebilecek olan cifar10 sıkıştırılmış dosyasının çıkartılmış
olması gerekmektedir. 

CIFAR10 İndirme linki: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz