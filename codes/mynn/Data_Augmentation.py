from abc import abstractmethod
import numpy as np

class DataAugmentation4img():
    def __init__(self) -> None:
        pass

    def __call__(self):
        self.DataAugmentation4img_generate()

    def rotation(self):
        pass

    def resize(self):
        pass
    
    def tiny_noise(self,noise_norm_relative=1e-4):
        pass

    def mixture(self):
        
        pass

    def DataAugmentation4img_generate(self):
        # 对上述方法采取一定比例进行混合
        return 