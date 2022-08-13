# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)

        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(self.W.shape, dtype = self.W.dtype)
        self.db = np.zeros(self.b.shape, dtype = self.b.dtype)

        self.momentum_W = np.zeros(self.W.shape, dtype = self.W.dtype)
        self.momentum_b = np.zeros(self.b.shape, dtype = self.b.dtype)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        self.out = np.matmul(self.x, self.W) + self.b
        return self.out
        raise NotImplemented

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        self.dW = np.matmul(self.x.T, delta) / delta.shape[0]
        self.db = np.mean(delta, axis = 0, keepdims = True)
        self.dx = np.matmul(delta, self.W.T)
        return self.dx
        raise NotImplemented
