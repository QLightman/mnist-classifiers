#Used to evaulate the performance of DecisionTree on modified MNIST data
import random
import sys
import numpy as np
import time
import pretreatment
from scipy import io as spio
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

def test_mlp():
    print (time.strftime('%Y-%m-%d %H:%M:%S'))
    #train_img_normlization,train_label,test_img_normlization,test_label = get_data()
    train_img_normlization,train_label,test_img_normlization,test_label = get_init_data()

    print("start training")
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=5)

    clf.fit(train_img_normlization, train_label)
    print("finish training")
    #joblib.dump(clf, "60000_model.m")

    predictions = [int(a) for a in clf.predict(test_img_normlization)]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_label))
    for num in range(0,10):
        correctNum = sum(int(a ==y)&int(y==num) for a, y in zip(predictions, test_label))
        SumOfNum = sum(int(y==num) for y in test_label)
        print("number %s :%s of %s test values correct. The accuracy is %s."%(num,correctNum,SumOfNum,1.0*correctNum/SumOfNum))
    print ("%s of %s test values correct." % (num_correct, len(test_label)))
    print ("The accuracy is %s"%(1.0*num_correct/len(test_label)))
    print (time.strftime('%Y-%m-%d %H:%M:%S'))

# get the modified MNIST data, size 45*45 
def get_init_data():
    train_data= np.fromfile('mnist_train_data',dtype=np.uint8)
    test_data= np.fromfile('mnist_test_data',dtype=np.uint8)
    test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
    train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
    return [np.reshape(train_data/255,(60000,-1)),train_label,np.reshape(test_data/255,(10000,-1)),test_label]

# get the enlarged rotated data, together 120000 training data, size 20*20
def get_enlarged_data():
    train_data1=pretreatment.pretreatment(size=60000,save_file=0,save_images=0,rotate=0,hog=0)
    train_data2=pretreatment.pretreatment(size=60000,save_file=0,save_images=0,rotate=1,hog=0)
    train_data=np.concatenate((train_data1,train_data2),axis=0)
    test_data=pretreatment.pretreatment(size=10000,save_file=0,save_images=0,rotate=0,hog=0,file_path="mnist_test_data")
    test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
    train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
    train_label=np.concatenate((train_label,train_label),axis=0)
    return [np.reshape(train_data/255,(120000,-1)),train_label,np.reshape(test_data/255,(10000,-1)),test_label]
    #return [np.reshape(np.round(train_data/255),(120000,-1)),train_label,np.reshape(np.round (test_data/255),(10000,-1)),test_label]

#get the data operated by HOG size 18*18
def get_hog_data():
    train_data=pretreatment.pretreatment(size=60000,save_file=0,save_images=0,hog=1)
    test_data=pretreatment.pretreatment(size=10000,save_file=0,save_images=0,rotate=0,hog=1,file_path="mnist_test_data")
    test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
    train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
    return [np.reshape(train_data/255,(60000,-1)),train_label,np.reshape(test_data/255,(10000,-1)),test_label]
    #return [np.reshape(np.round(train_data/255),(60000,-1)),train_label,np.reshape(np.round (test_data/255),(10000,-1)),test_label]

#get the data after noise reduction and size adjustment, size 20*20
def get_data():
    train_data=pretreatment.pretreatment(size=60000,save_file=0,save_images=0,rotate=0,hog=0)
    test_data=pretreatment.pretreatment(size=10000,save_file=0,save_images=0,rotate=0,hog=0,file_path="mnist_test_data")
    test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
    train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
    return [np.reshape(train_data/255,(60000,-1)),train_label,np.reshape(test_data/255,(10000,-1)),test_label]
    #return [np.reshape(np.round(train_data/255),(60000,-1)),train_label,np.reshape(np.round (test_data/255),(10000,-1)),test_label]

def main():
    test_mlp()

if __name__ == "__main__":
    main()