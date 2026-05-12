from abc import abstractmethod
import numpy as np

class report_visualization():
    def __init__(self,model_path,dataset_path,num_classes=10) -> None:
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.num_classes = num_classes

    def __call__(self):
        self.confusion_matrix()
        self.show()

    def confusion_matrix(self):
        # 展示混淆矩阵
        pass

    def show(self):
        # 先随机展示6个正确的示例图片和模型计算出来的prob条形图

        # 再选取loss最高的6个错误图片和模型计算出来的prob条形图
        pass