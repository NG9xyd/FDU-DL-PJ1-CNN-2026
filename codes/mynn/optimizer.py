from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]

# 训练的太差了，还是得把这个写一下方便训练
class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu):
        super().__init__(init_lr, model)
        self.mu = mu
        self.velocity = dict()

        for layer in self.model.layers:
            if layer.optimizable:
                self.velocity[id(layer)] = {key: 0 for key in layer.params.keys()}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                layer_v = self.velocity[id(layer)]
                for key, param in layer.params.items():
                    if layer.weight_decay:
                        param *= (1 - self.init_lr * layer.weight_decay_lambda)

                    grad = layer.grads[key]
                    layer_v[key] = self.mu * layer_v[key] - self.init_lr * grad
                    layer.params[key] = param + layer_v[key]
  
        # pass
