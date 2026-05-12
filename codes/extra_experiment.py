# Part C的额外实验部分
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# 可视化部分
# 生成混淆矩阵和正确或者错误的视频

# 数据增广

# without rotation & mixture

# 生成增广过后的数据

# 利用增广后的数据进行CNN的训练

# 对比模型的表现

# with rotation without mixture
# 由于2,5和6,9这两个的极端情况，所以想微微探索一下如果极端旋转180度的模型表现

# 生成增广过后的数据

# 利用增广后的数据进行CNN的训练

# 我们看混淆矩阵，如果2,5或者6,9有好的混淆，那么可以说明一些事情

# 完全noise
# 我们生成一个noise的比例很大的noise进行训练

