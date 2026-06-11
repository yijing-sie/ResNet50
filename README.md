# ResNet50


Intro to Deep Learning assignment:

## Convolution Neural Network

The first part of this assignment is to implement from scratch [Conv1d and Conv2d layers](mytorch/conv.py) and a [CNN classifier](hw2/cnn.py) **using NumPy only**

* [mytorch/conv.py](mytorch/conv.py) contains the forward and backward passes implementation of **Conv1D** and **Conv2D** that are similar to **torch.nn.Conv1d** and **torch.nn.Conv2d** respectively
> [mytorch](mytorch) is my custom deep learning library, built entirely in NumPy, that functions similarly to established DL lebraries like PyTorch or TensorFlow

*  [hw2/cnn.py](hw2/cnn.py) is a pure-Numpy CNN classifier class

## ResNet-50

For the second part, the goal is to implement from scratch a ResNet50 model to perform face classification and face verification.

* They are both Kaggle competitions, and all the details can be found here:

[face-classification](https://www.kaggle.com/competitions/idl-fall21-hw2p2s1-face-classification)

* For face classification, my model achieves **0.87339** accuracy on `test` set

* It's ranked number **14** in a class of 300+ students

[face-verification](https://www.kaggle.com/competitions/idl-fall21-hw2p2s2-face-verification)

* For face verification, my model achieves **0.94032** AUC on the `test` set

* It's ranked top **15%** in a class of 300+ students

* All my work for this part can be found in [resnet50.ipynb](resnet50.ipynb)




