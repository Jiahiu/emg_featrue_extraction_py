import scipy.io
import os
import numpy as np


def load_mat_files_from_folder(folder_path):
    # 获取文件夹中所有.mat文件的列表
    mat_files = [f for f in os.listdir(folder_path) if f.endswith(".mat")]

    # 创建一个列表来存储所有数据
    numpy_data_list = []

    # 循环读取每个.mat文件
    for mat_file in mat_files:
        mat_file_path = os.path.join(folder_path, mat_file)

        # 加载.mat文件
        mat_data = scipy.io.loadmat(mat_file_path)

        # 假设每个.mat文件中的数据存储在名为'data'的键中
        # 你需要根据实际情况调整键名
        if "acData" in mat_data:
            data = mat_data["acData"]
            numpy_data_list.append(data)
        else:
            print(f"'data' key not found in {mat_file}")

    return numpy_data_list


# 示例用法
folder_path = r"G:\data\svm_dem\01\uf"  # 替换为你的.mat文件所在的文件夹路径
numpy_data_list = load_mat_files_from_folder(folder_path)
np.save("./data/numpy_data_list.npy", numpy_data_list)


# 0--被试数据
npy_file_path = r"./data/s01_uf.npy"
case_data = np.load(npy_file_path, allow_pickle=True)
# 1--分帧
# 对data (8,64000,12)的数据进行分帧
window_size = 200
num_windows = case_data.shape[1] // window_size
win_data = np.zeros((8, num_windows, 200, 12))  # 200*12更好处理


def vorbis_window(N):
    """
    生成一个长度为N的Vorbis窗函数。
    """
    n = np.arange(N)
    window = np.sin((np.pi / 2) * np.sin(np.pi * n / N) ** 2)
    return window


window = vorbis_window(window_size)
window_expanded = window.reshape(200, 1)
window_expanded = np.tile(window_expanded, (1, 12))
for i in range(8):
    for j in range(num_windows):
        start_index = j * 200
        end_index = start_index + 200
        win_data[i, j, :, :] = case_data[i, start_index:end_index, :] * window_expanded

np.save("./data/win_data.npy", win_data)
