# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        self.in_width = x.shape[2]
        self.out_width = round(self.in_width - self.kernel_size) // self.stride + 1
        self.out = np.zeros((x.shape[0], self.out_channel, self.out_width))
        for i in range(self.out_width):
            self.out[:,:,i] = np.tensordot( x[:, :, (i * self.stride): (i * self.stride) + self.kernel_size], self.W, ([1, 2], [1, 2])) + self.b
        return self.out       
#%%
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """        
        #initialize
        self.dW, self.db, self.dx = np.zeros_like(self.W), np.zeros_like(self.b), np.zeros_like(self.x)
        #check stride size -> dilation
        if self.stride > 1:
            dilated_factor = self.stride - 1
            dilated_delta = np.zeros((delta.shape[0], 
                                      self.out_channel, 
                                      delta.shape[2] + (delta.shape[2] - 1) * dilated_factor), dtype = delta.dtype)
            dilated_delta[:, :, ::self.stride] = delta
            delta = dilated_delta 
        #padding    
        pad_delta = np.pad(delta, ((0, 0),(0, 0), (self.kernel_size - 1, self.kernel_size - 1)), 
                            'constant', constant_values=0)
        #dW
        for i in range(self.kernel_size):
            self.dW[:, :, i] = np.tensordot(delta, self.x[:, :, i : i + delta.shape[2]],([0, 2], [0, 2]))
        #db
        self.db = np.sum(delta, axis = (0,2))
        #dx        
        flip_W = self.W[:, :, ::-1]
        backprop_W = np.transpose(flip_W, axes=(1, 0, 2))
        backprp = pad_delta.shape[2] - self.kernel_size + 1
        for i in range(backprp):
            self.dx[:, :, i] = np.tensordot( pad_delta[:, :, i : i + self.kernel_size], backprop_W,([1, 2],[1, 2])) 

        return self.dx              
#%%

class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            # Kaiming init (fan-in) (good init strategy) 
            bound = np.sqrt(1 / (in_channel * kernel_size * kernel_size)) 
            # w is tensor(weight, requires_grad=True, is_parameter=True) 
            # self.w = np.random.uniform(-bound, bound, size=(out_channel, in_channel, kernel_size, kernel_size))            
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        if bias_init_fn is None:
            #bias is aTensor(bias, requires_grad=True, is_parameter=True)
            #self.b = np.random.uniform(-bound, bound, size=(out_channel,))             
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        self.in_width = x.shape[2]
        self.in_height = x.shape[-1]
        self.out_width = round(self.in_width - self.kernel_size) // self.stride + 1
        self.out_height = round(self.in_height - self.kernel_size) // self.stride + 1
        self.out = np.zeros((x.shape[0], self.out_channel, self.out_width, self.out_height))
        #W = (out_channel, in_channel, kernel_size, kernel_size)
                
        for w in range(self.out_width):
            for h in range(self.out_height):
                self.out[:, :, w, h] = np.tensordot(self.x[:, : , (w * self.stride): (w * self.stride) + self.kernel_size,
                                                            (h * self.stride): (h * self.stride) + self.kernel_size], self.W, 
                                                            ([1, 2, 3], [1, 2, 3])) + self.b       
        return self.out  
#%%
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        self.dW, self.db, self.dx = np.zeros_like(self.W), np.zeros_like(self.b), np.zeros_like(self.x)
        #check stride size -> dilation
        if self.stride > 1:
            dilated_factor = self.stride - 1
            dilated_delta = np.zeros((delta.shape[0], 
                                      self.out_channel, 
                                      delta.shape[2] + (delta.shape[2] - 1) * dilated_factor, 
                                      delta.shape[3] + (delta.shape[3] - 1) * dilated_factor))
            dilated_delta[:, :, ::self.stride, ::self.stride] = delta
            delta = dilated_delta 
        #padding    
        pad_delta = np.pad(delta, ((0, 0),(0, 0), (self.kernel_size - 1, self.kernel_size - 1), 
                                                   (self.kernel_size - 1, self.kernel_size - 1)), 
                                                                   'constant', constant_values=0)
        #dW
        #W = (out_channel, in_channel, kernel_size, kernel_size)
        for k_w in range(self.W.shape[2]):
            for k_h in range(self.W.shape[-1]):
                self.dW[:, :, k_w, k_h] = np.tensordot(delta, self.x[:, :, k_w: k_w + delta.shape[2], 
                                                                            k_h: k_h + delta.shape[-1]], 
                                                                               ([0, 2, 3],[0, 2, 3]))           
        #db
        self.db = np.sum(delta, axis = (0, 2, 3))
        #dx
        flip_W = self.W[:, :, ::-1, ::-1]
        backprop_W = np.transpose(flip_W, axes=(1, 0, 2, 3))
        backprp_width = pad_delta.shape[2] - self.kernel_size + 1
        backprp_height = pad_delta.shape[3] - self.kernel_size + 1
        for i in range(backprp_width):
            for j in range(backprp_height):
                self.dx[:, :, i, j] = np.tensordot(pad_delta[:, : , i: i + self.kernel_size,
                                                            j: j + self.kernel_size], backprop_W, 
                                                            ([1, 2, 3], [1, 2, 3]))         
        return self.dx
#%%
class Conv2D_dilation():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride, padding=0, dilation=1,
                 weight_init_fn=None, bias_init_fn=None):
        """
        Much like Conv2D, but take two attributes into consideration: padding and dilation.
        Make sure you have read the relative part in writeup and understand what we need to do here.
        HINT: the only difference are the padded input and dilated kernel.
        """

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # After doing the dilation， the kernel size will be: (refer to writeup if you don't know)
        self.dilated_factor = self.kernel_size - 1
        self.kernel_dilated = self.dilated_factor * (self.dilation - 1) + self.kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

        self.W_dilated = np.zeros((self.out_channel, self.in_channel, self.kernel_dilated, self.kernel_dilated))

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        self.in_width = x.shape[2]
        self.in_height = x.shape[-1]
        # TODO: padding x with self.padding parameter (HINT: use np.pad())
        self.pad_x = np.pad(self.x, ((0, 0),(0, 0), (self.padding, self.padding), 
                                                   (self.padding, self.padding)), 
                                                                   'constant', constant_values=0)
        # TODO: do dilation -> first upsample the W -> computation: k_new = (k-1) * (dilation-1) + k = (k-1) * d + 1
        #       HINT: for loop to get self.W_dilated
        #output size = [(input size + 2 * padding - dilation * (kernel size - 1) - 1)//stride] + 1
        self.out_width = round(self.in_width + 2 * self.padding - self.dilation * self.dilated_factor - 1) // self.stride + 1
        self.out_height = round(self.in_width + 2 * self.padding - self.dilation * self.dilated_factor - 1) // self.stride + 1
        self.out = np.zeros((x.shape[0], self.out_channel, self.out_width, self.out_height))
        # TODO: regular forward, just like Conv2d().forward()
        self.W_dilated[:, :, ::self.dilation, ::self.dilation] = self.W
        
        forprp_width = round(self.pad_x.shape[2] - self.kernel_dilated) // self.stride + 1
        forprp_height = round(self.pad_x.shape[3] - self.kernel_dilated) //self.stride + 1  
        for w in range(forprp_width):
            for h in range(forprp_height):
                self.out[:, :, w, h] = np.tensordot(self.pad_x[:, : , (w * self.stride): (w * self.stride) + self.kernel_dilated,
                                                            (h * self.stride): (h * self.stride) + self.kernel_dilated], self.W_dilated, 
                                                            ([1, 2, 3], [1, 2, 3])) + self.b    
        return self.out        


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # TODO: main part is like Conv2d().backward(). The only difference are: we get padded input and dilated kernel
        #       for whole process while we only need original part of input and kernel for backpropagation.
        #       Please refer to writeup for more details.
        self.db, self.dx = np.zeros_like(self.b), np.zeros_like(self.x)
        temp_x = np.zeros_like(self.pad_x)
        temp_W = np.zeros_like(self.W_dilated)
        #dilation
        if self.stride > 1:
            dilated_factor = self.stride - 1
            dilated_delta = np.zeros((delta.shape[0], self.out_channel, 
                                      delta.shape[2] + (delta.shape[2] - 1) * dilated_factor, 
                                      delta.shape[3] + (delta.shape[3] - 1) * dilated_factor))
            dilated_delta[:, :, ::self.stride, ::self.stride] = delta
            delta = dilated_delta
        pad_delta = np.pad(delta, ((0, 0),(0, 0), (self.kernel_dilated  - 1, self.kernel_dilated  - 1), 
                                                   (self.kernel_dilated  - 1, self.kernel_dilated  - 1)), 
                                                                   'constant', constant_values=0)            
        
        backprp_width = pad_delta.shape[2] - self.kernel_dilated + 1
        backrp_height = pad_delta.shape[3] - self.kernel_dilated + 1
        flip_W = self.W_dilated[:,:,::-1,::-1]
        backprop_W = np.transpose(flip_W, axes=(1, 0, 2, 3))
        for w in range(backprp_width):
            for h in range(backrp_height):
                temp_x[:, :, w, h] = np.tensordot(pad_delta[:, :, w : w + self.kernel_dilated,
                                                        h : h + self.kernel_dilated], backprop_W ,([1,2,3],[1,2,3])) 

        for i in range(self.W_dilated.shape[2]):
            for j in range(self.W_dilated.shape[3]):
                temp_W[:, :, i, j] = np.tensordot(delta, self.pad_x[:, :, i : i + delta.shape[2], 
                                                                    j : j + delta.shape[3]], ([0, 2, 3],[0, 2, 3]))
        self.dx = temp_x[:, :, self.padding: -self.padding, self.padding: -self.padding]
        #db
        self.db = np.sum(delta, axis = (0, 2, 3))
        #dW
        self.dW = temp_W[:, :, ::self.dilation, ::self.dilation]
        return self.dx
   




class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        (self.b, self.c, self.w) = x.shape
        out = x.reshape(x.shape[0], -1)
        return out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        dx = np.reshape(delta, (self.b, self.c, self.w))
        return dx

