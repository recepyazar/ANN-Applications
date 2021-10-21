import numpy as np

if __name__ == "__main__":
    class1_count = 0
    class2_count = 0
    train_count1 = 0
    test_count1 = 0
    train_count2 = 0
    test_count2 = 0
    class1 = np.zeros((20,5,1))
    class2 = np.zeros((20,5,1))
    train_data1 = np.zeros((13,5,1))
    test_data1 = np.zeros((7,5,1))
    train_data2 = np.zeros((12,5,1))
    test_data2 = np.zeros((8,5,1))
    surface_vector = np.array([-1, 3, 4, -2, 5])
    result = 0

    while(1):
        random_vector = np.random.randint(-10,10,[5,1])
        """ax¹+bx²+cx³+dx⁴+ex⁵ = f = equation of surface
        Classes can be define as:
        [a b c d e].[x]>f =class1
        [a b c d e].[x]<f =class2
        [a b c d e] and f determined as:
        [a b c d e] = [-1 3 4 -2 5]
        f = 0"""
        result = int(np.dot(surface_vector,random_vector))
        if result > 0:
            if class1_count < 20:    
                if result < 10 and train_count1 < 13:
                    class1[class1_count] = random_vector
                    class1_count += 1
                    train_data1[train_count1] = random_vector
                    train_count1 += 1
                elif result > 10 and test_count1 < 7:
                    class1[class1_count] = random_vector
                    class1_count += 1
                    test_data1[test_count1] = random_vector
                    test_count1 += 1
        else:
            if class2_count < 20:    
                if result > -10 and train_count2 < 12:
                    class2[class2_count] = random_vector
                    class2_count += 1
                    train_data2[train_count2] = random_vector
                    train_count2 += 1
                elif result < -10 and test_count2 < 8:
                    class2[class2_count] = random_vector
                    class2_count += 1
                    test_data2[test_count2] = random_vector
                    test_count2 += 1
        if class1_count == 20 and class2_count == 20:
            train_data = np.concatenate([train_data1,train_data2])
            test_data = np.concatenate([test_data1,test_data2])
            #dataset created
            break
    f = open("data.txt", "w")
    f.write("class 1:\n")
    class1_str = str(class1)
    f.write(class1_str + "\n")
    f.write("class2:\n")
    class2_str = str(class2)
    f.write(class2_str + "\n")
    f.write("train data:\n")
    train_data_str = str(train_data)
    f.write(train_data_str + "\n")
    f.write("test_data:\n")
    test_data_str = str(test_data)
    f.write(test_data_str + "\n")
    f.close()




