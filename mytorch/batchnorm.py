# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.x = x
        if eval:
            return ((self.x - self.running_mean)/np.sqrt(self.running_var + self.eps)) * self.gamma + self.beta


        # First compute the mean of the input, mean is a 1xM row vector.
        self.mean = np.mean(self.x, axis=0, keepdims = True)
        # Then compute the variance of the input, var is a 1xM row vector.
        self.var = np.var(self.x, axis=0, keepdims = True)
        # Norm the input X using the mean and variance computed from above. Remember to add
        self.norm = (self.x - self.mean)/np.sqrt(self.var + self.eps)
        # Do the affine transformation, gamma and beta are 1xM row vectors.
        self.y = self.norm * self.gamma + self.beta
        # Update the running_mean and running_var, alpha is a scaler to control the updating speed.
        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
        return self.y

        # raise NotImplemented


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        #get batch size
        b = delta.shape[0]
        #we’ll be using this term a lot – better make a constant!
        self.sqrt_var_eps = np.sqrt(self.var + self.eps)
        #Find the derivative of gamma and beta for gradient descent.
        self.dgamma = np.sum(self.norm * delta, axis = 0, keepdims = True)
        self.dbeta = np.sum(delta, axis = 0, keepdims = True)
        #Find the derivative of norm
        self.gradNorm = self.gamma * delta
        #Find the derivative of variance (this looks complicated but isn’t too bad!)
        self.gradVar = -.5*(np.sum((self.gradNorm * (self.x - self.mean))/ self.sqrt_var_eps**3, axis = 0))
        #Find the derivative of the mean. Again, looks harder than it actually is J
        self.first_term_dmu = -(np.sum(self.gradNorm / self.sqrt_var_eps, axis = 0))
        self.second_term_dmu = - (2/b)*(self.gradVar)*(np.sum(self.x - self.mean, axis = 0))
        self.gradMu = self.first_term_dmu + self.second_term_dmu
        #use all the derivative we have found to get our final result!
        self.first_term_dx = self.gradNorm/self.sqrt_var_eps
        self.second_term_dx = self.gradVar * (2/b) * (self.x - self.mean)
        self.third_term_dx = self.gradMu * (1/b)
        return self.first_term_dx + self.second_term_dx + self.third_term_dx

        # raise NotImplemented
