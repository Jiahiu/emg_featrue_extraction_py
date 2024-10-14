import numpy as np


def vorbis_window(N):
    """
    生成一个长度为N的Vorbis窗函数。
    """
    n = np.arange(N)
    window = np.sin((np.pi / 2) * np.sin(np.pi * n / N) ** 2)
    return window


def apply_vorbis_window(signal, window_size):
    """
    将Vorbis窗函数应用于输入信号。
    参数：
    - signal: 输入信号，一维数组。
    - window_size: 窗口的大小。
    返回：
    - 加窗后的信号片段列表。
    """
    # 创建Vorbis窗
    window = vorbis_window(window_size)
    # 计算每个窗口的偏移（50%重叠）
    hop_size = window_size // 2

    # 存储加窗后的信号片段
    windowed_segments = []

    # 对信号进行分帧和加窗
    for start in range(0, len(signal) - window_size + 1, hop_size):
        segment = signal[start : start + window_size]
        windowed_segment = segment * window
        windowed_segments.append(windowed_segment)

    return np.array(windowed_segments)
