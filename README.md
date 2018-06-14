# mnist-classifiers
SJTU class project for CS420

## Pretreatment

This part was realized in the [code](https://github.com/QLightman/mnist-classifiers/blob/master/traditional_methods/pretreatment.py).
First you have to put the **mnist_test_data**, **mnist_train_data**, **mnist_test_label**, **mnist_test_label** in the correct path.

You can type 
```
python pretreatment.py --size=10 ----save_images=1
```
 to see the results of the output images. You can also `import pretreatment` in other code, and  use this line  
  ```
  train_data=pretreatment.pretreatment(size=60000,save_file=0,save_images=0,rotate=0,hog=0)
  ```
   to obtained 60000 pretreated images.
Type 
```
python pretreatment.py -h 
```
to see more commands.

If you get the `ImportError: No module named cv2`, use this command
```
pip install opencv-python
```
to install the cv2 module.

## Traditional Methods
For this part, we implment five algorithms, SVM, DecisionTree, Logistic Regression , MLP and RandomForest.

The usage of each traditional algorithms are similar, type
```
python ${file_name}.py
```
to see the results. You can also use the original modified MNIST data, the enlarged data, the data operated by HOG to see the results.

## Deeplearing Methods
For this part, we implemented three different deep learning algorithm, CNN, CapsNet and DenseNets. Their usage are as follows:
### CNN
Type
```
python main.py
```
to see the results.

### CapsNet
Type
```
python main.py
```
to see the results.

### DenseNets
Type
```
python Densenet_MNIST.py
```
to see the results.

## Models
Here, we also include some models of the traditional methods, we don't put the models for deeplearing methods since it is too large.

The twos models are the best results we get in handling traditional methods. The accuracy of [60000_hog_svm_9930.m](https://github.com/QLightman/mnist-classifiers/blob/master/model/60000_hog_svm_9930.m) is 99.3%, the arruracy of [120000_hog_svm_9933.m](https://github.com/QLightman/mnist-classifiers/blob/master/model/120000_hog_svm_9933.m) is 99.33%.

## Collaborators
[LeoNardo10521](https://github.com/LeoNardo10521)

[AeroAiro](https://github.com/AeroAiro)