import numpy as np
import matplotlib.pyplot as plt
from emgf import emgf

# 测试数据
# fs = 1000  # 采样频率
# Ts = 1 / fs  # 周期
# t = np.arange(0, 0.25, Ts)
# X = 0.01 * (np.cos(2 * np.pi * fs * t) + np.random.randn(len(t)))
data = np.load(r"./data/win_data.npy")


# 2--特征提取
# rms mav tdpsd mpf mdf dasdv
# def getFeat(X):
#     f1 = emgf("rms", abs(X))
#     f2 = emgf("mav", abs(X))
#     f3 = emgf("tdpsd", abs(X))
#     f4 = emgf("mpf", X)
#     f5 = emgf("mdf", X)
#     f6 = emgf("dasdv", abs(X))
#     feat = [f1, f2, f3, f4, f5, f6]
#     return feat


# X = data
# feat = np.zeros((8, 320, 12 * 6))
# for i in range(8):
#     for j in range(320):
#         for k in range(12):
#             sta = 6 * k
#             end = sta + 6
#             feat[i, j, sta:end] = getFeat(X[i, j, :, k])


# 3--模型训练


# 4--模型评估（特征评估）

x = 1
# 示例 2: 提取 3 个特征并带参数
