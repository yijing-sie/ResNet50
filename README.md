# ResNet50


Intro to Deep Learning assignment:

## Convolution Neural Network

The first part of this assignment is to implement from scratch a numpy-based **[CNN](hw2/cnn.py)** model **without using Pytorch**

> For this, I implemented the **Conv1D** and **Conv2D** classes in [mytorch/conv.py](mytorch/conv.py) so that it has similar usage and functionality to **torch.nn.Conv1d** and **torch.nn.Conv2d**

*  I also implemented **padding** and **dilation** functions for 2D convolution in [mytorch/conv.py](mytorch/conv.py)
*  MyTorch is my own custom deep learning library, which act similar to other deep learning libraries like PyTorch or Tensorflow
*  [hw2/cnn.py](hw2/cnn.py) is my implementation of the numpy-based CNN model

## ResNet-50

For the second part, the goal is to implemented from scratch a ResNet50 model to perform face classification and face verification.

* They are both kaggle competitions, and all the details can be found here:

[face-classification](https://www.kaggle.com/competitions/idl-fall21-hw2p2s1-face-classification)


[face-verification](https://www.kaggle.com/competitions/idl-fall21-hw2p2s2-face-verification)

* All the work for this part can be found in [resnet50.ipynb](resnet50.ipynb)
* For face classification, my model achieves 0.87339 accuracy on `test` set
* For face verification, my model achieves 0.94032 AUC on `test` set





