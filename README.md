# mnist-classifiers
SJTU class project for CS420

## Pretreatment

This part was realized in the [code](https://github.com/QLightman/mnist-classifiers/blob/master/traditional_methods/pretreatment.py).
You can use type 
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
## TraditionalMethods
The usage of each traditional algorithms are similar, type
```
python ${file_name}.py
```
to see the results. Yo ucan also use the original modified MNIST data, the enlarged data, the data operated by HOG to see the results.

## DeeplearingMethods

## Collaborators
[LeoNardo10521](https://github.com/LeoNardo10521)

[AeroAiro](https://github.com/AeroAiro)