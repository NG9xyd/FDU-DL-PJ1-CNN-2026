from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        output = self.W @ X + self.b
        return output
        # pass

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = self.input.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        output = grad @ self.W.T
        return output
        # pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.outchannels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels, 1, 1))

        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.input_padded = None

        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        # pass

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        # 这padding都给了还是把这padding写一下
        self.input = X
        if self.padding > 0:
            X_pad = np.pad(X,((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),mode='constant')
        else:
            X_pad = X
        self.input_padded = X_pad

        batch_size,__,original_h,original_w = X.shape
        k = self.kernel_size 
        out_h = (original_h - k) // self.stride +1
        out_w = (original_w - k) // self.stride +1
        output = np.zeros((batch_size,self.outchannels,out_w,out_h))
        for i in range(batch_size):
            for j in range(self.outchannels):
                for m in range(out_h):
                    h_start = m*self.stride 
                    for n in range(out_w):
                        w_start = n*self.stride
                        window = X_pad[i, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        output[i,j,m,n] = np.sum(window * self.W[j]) + self.b[0, j, 0, 0] #三维的卷积
        return output            
        # pass

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_h, out_w = grads.shape
        dX_pad = np.zeros_like(self.input_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for n in range(batch_size):
            for oc in range(self.outchannels):
                for i in range(out_h):
                    h_start = i * self.stride
                    for j in range(out_w):
                        w_start = j * self.stride
                        grad_value = grads[n, oc, i, j]
                        window = self.input_padded[n, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        dW[oc] += window * grad_value
                        dX_pad[n, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += self.W[oc] * grad_value
                        db[0, oc, 0, 0] += grad_value

        self.grads['W'] = dW
        self.grads['b'] = db
        if self.padding > 0:
            return dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_pad
        #pass
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

# CNN要有最基本的池化层和flatten层
class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        return grads.reshape(self.input_shape)

class maxpool(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass
    def backward(self):
        pass

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model 
        self.max_classes = max_classes
        self.label_smooth_para = 1e-2
        self.has_softmax = True
        self.predicts = None
        # 说不定实验报告可以看一下这个概率分布
        self.labels = None
        self.grads = None
        self.optimizable = True 
        # 用于给test？
        # pass

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.labels = labels
        batch_size = predicts.shape[0]
        ep = 1e-10
        if self.has_softmax:
            self.predicts = softmax(predicts)
        else:
            self.predicts = predicts
        correct_prob_log = np.log(predicts[np.arange(batch_size), self.labels]+ep)
        if self.label_smooth_para:
            # assert self.max_classes>1 , '类别为1种，无需MLP帮助分类'   
            log_sum_prob = np.sum(np.log(self.predicts+ep),axis=1)
            # 严格推导后 labelsmoothing得到的KL散度实际表达式应为有一个系数，不过他都是超参了，可以随便调节
            # label_smoothing_para = self.label_smooth_para * (self.max_classes-1) / self.max_classes
            return -np.mean( (1-self.label_smooth_para)*correct_prob_log + self.label_smooth_para/self.max_classes * log_sum_prob )
        else:
            return -np.mean(correct_prob_log+ep)
        # pass
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # target:self.grads
        batch_size, num_classes = self.predicts.shape
        eps = 1e-10
        s = self.label_smooth_para if self.label_smooth_para else 0
        q = np.full_like(self.predicts, s / num_classes)
        q[np.arange(batch_size), self.labels] = 1 - s + s / num_classes
        if self.has_softmax:
            self.grads = (self.predicts - q) / batch_size
        else:
            self.grads = -q / (self.predicts + eps) / batch_size

        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition