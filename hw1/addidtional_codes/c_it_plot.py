import numpy as np
from matplotlib import pyplot as plt

class perceptron(object):
    def __init__(self,dim,c,iteration):
        self.Weights = np.full(dim+1,0)
        self.iteration = iteration
        self.c = c

    def activation(self, summation):
        if summation > 0:
            return 1
        else:
            return -1

    def train(self,train_data,train_class_vector):
        for i in range(self.iteration):
            count = 0
            for j in range(train_class_vector.size):
                with_bias = np.insert(train_data[j],5,1)
                dot_product = with_bias.T.dot(self.Weights)
                y = self.activation(dot_product)
                error = train_class_vector[j] - y
                if error == 0:
                    count += 1
                self.Weights = self.Weights + self.c * error * with_bias
            if count == 25:
                print("Weights: ")
                print(self.Weights)
                print("iteration number: ")
                print(i)
                break
        return i        
    def test(self,test_data,test_class_vector):
        true = 0
        false = 0
        for k in range(test_class_vector.size):
            with_bias = np.insert(test_data[k],5,1)
            dot_product = with_bias.T.dot(self.Weights)
            y = self.activation(dot_product)
            if test_class_vector[k] == y:
                true += 1
            else:
                false += 1
        print("Number of true data:",true)


if __name__ == '__main__':
    train_ones = np.ones(13,dtype=int) #train data için -1/1 değerleri
    train_minus_ones = np.full((12),-1,dtype=int)
    train_class_vector = np.concatenate([train_ones,train_minus_ones])
    test_ones = np.ones(7,dtype=int) #test data için -1/1 değerleri
    test_minus_ones = np.full((8),-1,dtype=int)
    test_class_vector = np.concatenate([test_ones,test_minus_ones])
    train_data = np.array([[[ -7.],[  8.],[  4.],[ -2.],[ -9.]],
 [[  3.],[  9.],[  5.],[ -5.],[ -9.]],
 [[ -5.],[  4.],[-10.],[ -5.],[  3.]],
 [[ -9.],[  0.],[  3.],[  7.],[  0.]],
 [[ -2.],[  5.],[  5.],[  5.],[ -5.]],
 [[ -7.],[ -8.],[  2.],[  7.],[  6.]],
 [[ -4.],[ -6.],[ -3.],[ -5.],[  5.]],
 [[  8.],[ -5.],[  6.],[ -5.],[ -1.]],
 [[ -1.],[ -2.],[  5.],[  8.],[  2.]],
 [[  2.],[ -5.],[  6.],[  8.],[  3.]],
 [[ -8.],[ -2.],[  3.],[ -4.],[ -3.]],
 [[ -8.],[  6.],[ -8.],[  6.],[  4.]],
 [[  9.],[  8.],[  1.],[ -6.],[ -5.]],
 [[-10.],[  6.],[ -5.],[ -6.],[ -4.]],
 [[  8.],[  9.],[ -3.],[  7.],[  1.]],
 [[  2.],[ -1.],[  8.],[ -7.],[-10.]],
 [[ -7.],[ -4.],[ -4.],[  5.],[  6.]],
 [[ -3.],[  4.],[ -9.],[  8.],[  7.]],
 [[  5.],[  5.],[  0.],[ -9.],[ -6.]],
 [[ -7.],[ -5.],[  2.],[  6.],[  1.]],
 [[ -4.],[ -1.],[ -1.],[  1.],[  1.]],
 [[ -9.],[  9.],[ -9.],[ -7.],[ -4.]],
 [[ -7.],[  2.],[ -7.],[-10.],[ -2.]],
 [[ -4.],[ -1.],[ -1.],[ -2.],[ -2.]],
 [[ -1.],[  9.],[ -1.],[-10.],[-10.]]])
    test_data = np.array([[[ -4.],[ -3.],[  3.],[ -3.],[  1.]],
 [[  7.],[  3.],[  1.],[ -8.],[  1.]],
 [[ -8.],[  1.],[  2.],[  2.],[  6.]],
 [[ -9.],[  4.],[  9.],[  6.],[  8.]],
 [[ -6.],[ -4.],[  5.],[ -4.],[  2.]],
 [[  9.],[ -4.],[  4.],[ -6.],[  5.]],
 [[ -6.],[ -1.],[  9.],[ -6.],[  4.]],
 [[ -2.],[ -2.],[  6.],[  3.],[-10.]],
 [[  7.],[  0.],[ -5.],[ -3.],[ -8.]],
 [[  9.],[ -1.],[ -6.],[ -3.],[  2.]],
 [[  2.],[ -1.],[ -6.],[  3.],[ -5.]],
 [[  9.],[  0.],[ -3.],[ -5.],[ -5.]],
 [[ -6.],[ -3.],[  4.],[  8.],[ -2.]],
 [[ -6.],[  1.],[ -9.],[  7.],[  4.]],
 [[ -6.],[-10.],[  0.],[  0.],[  2.]]])
    it_number = np.zeros(100)
    c_value = np.zeros(100)
    for a in range(100):
        p1 = perceptron(5,0.01*a,500)
        c_value[a] = a*0.01
        result = p1.train(train_data,train_class_vector)
        it_number[a] = result
        p1.test(test_data,test_class_vector)
    plt.plot(c_value, it_number, 'b')
    plt.title('Öğrenme Hızı vs İterasyon Sayısı')
    plt.xlabel('Öğrenme Hızı')
    plt.ylabel('İterasyon Sayısı')
    plt.show()

    
    

