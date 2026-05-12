from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.params['W'] = param_list[i + 2]['W']
                layer.params['b'] = param_list[i + 2]['b']
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, channels_list=None, kernel_size_list=None, stride_list=None, num_classes=10, image_size=28, act_func='ReLU', weight_decay_lambda=1e-4):
        super().__init__()
        self.channels_list = channels_list
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.num_classes = num_classes
        self.image_size = image_size
        self.act_func = act_func
        self.weight_decay_lambda = weight_decay_lambda
        self.layers = []
        if channels_list is None or kernel_size_list is None:
            return

        if stride_list == None:
            stride_list = [1] * len(channels_list)
        assert len(channels_list)==len(kernel_size_list) and len(kernel_size_list) == len(stride_list),'第一层channel为1不算入channellist中或者长度不一致，请注意修改'
        if act_func != 'ReLU':
            raise NotImplementedError
        self.channels_list = channels_list
        self.num_classes = num_classes
        self.image_size = image_size
        self.kernel_size_list = kernel_size_list
        self.stride_list = stride_list
        self.act_func = act_func
        self.weight_decay_lambda = weight_decay_lambda
        self.CNN_layer_num = len(channels_list)      
        self.layers.append(conv2D(in_channels=1,out_channels=channels_list[0],kernel_size=kernel_size_list[0],stride=stride_list[0],weight_decay_lambda=weight_decay_lambda))
        self.layers.append(ReLU())
        for i in range(1,self.CNN_layer_num):
            self.layers.append(conv2D(in_channels=channels_list[i-1],out_channels=channels_list[i],kernel_size=kernel_size_list[i],stride=stride_list[i],weight_decay_lambda=weight_decay_lambda))
            self.layers.append(ReLU())
        self.layers.append(Flatten())
        feature_size = image_size
        for kernel_size, stride in zip(kernel_size_list, stride_list):
            feature_size = (feature_size - kernel_size) // stride + 1
            assert feature_size > 0, '卷积后的特征图尺寸小于等于0，请检查 kernel_size_list 和 stride_list'
        conv_out = channels_list[-1] * feature_size * feature_size
        self.layers.append(Linear(conv_out,num_classes,weight_decay_lambda=weight_decay_lambda))
        #pass

    def __call__(self, X):
        return self.forward(X)
        #pass

    def forward(self, X):
        assert self.channels_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with channels_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs
        #pass

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads
        # pass
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        config = param_list[0]
        self.__init__(
            channels_list=config['channels_list'],
            kernel_size_list=config['kernel_size_list'],
            stride_list=config['stride_list'],
            num_classes=config['num_classes'],
            image_size=config['image_size'],
            act_func=config['act_func'],
            weight_decay_lambda=config['weight_decay_lambda'],
        )
        optimizable_layers = [layer for layer in self.layers if layer.optimizable]
        for layer, saved_params in zip(optimizable_layers, param_list[1:]):
            layer.params['W'] = saved_params['W']
            layer.params['b'] = saved_params['b']
            layer.weight_decay = saved_params['weight_decay']
            layer.weight_decay_lambda = saved_params['lambda']
        
    def save_model(self, save_path):
        param_list = [{
            'channels_list': self.channels_list,
            'kernel_size_list': self.kernel_size_list,
            'stride_list': self.stride_list,
            'num_classes': self.num_classes,
            'image_size': self.image_size,
            'act_func': self.act_func,
            'weight_decay_lambda': self.weight_decay_lambda,
        }]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
