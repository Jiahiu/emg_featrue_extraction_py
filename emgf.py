import numpy as np
from scipy.signal import levinson_durbin
from scipy.stats import kurtosis, skew

# 定义 EMG 特征提取函数


def emgf(feature_type, X, opts=None):
    if feature_type == "fzc":
        return j_new_zero_crossing(X)
    elif feature_type == "ewl":
        return j_enhanced_wave_length(X)
    elif feature_type == "emav":
        return j_enhanced_mean_absolute_value(X)
    elif feature_type == "asm":
        return j_absolute_value_of_the_summation_of_exp_root(X)
    elif feature_type == "ass":
        return j_absolute_value_of_the_summation_of_square_root(X)
    elif feature_type == "msr":
        return j_mean_value_of_the_square_root(X)
    elif feature_type == "ltkeo":
        return j_log_teager_kaiser_energy_operator(X)
    elif feature_type == "lcov":
        return j_log_coefficient_of_variation(X)
    elif feature_type == "card":
        return j_cardinality(X, opts=None)
    elif feature_type == "ldasdv":
        return j_log_difference_absolute_standard_deviation_value(X)
    elif feature_type == "dasdv":
        return j_difference_absolute_standard_deviation_value(X)
    elif feature_type == "rms":
        return j_root_mean_square(X)
    elif feature_type == "mav":
        return j_mean_absolute_value(X)
    elif feature_type == "tdpsd":
        return tdpsd(X)
    elif feature_type == "mdf":
        return median_frequency(X)
    elif feature_type == "mpf":
        return mean_power_frequency(X)
    elif feature_type == "ldamv":
        return mean_power_frequency(X)
    elif feature_type == "dvarv":
        return mean_power_frequency(X)
    if feature_type == "mfl":
        return jMaximumFractalLength(X)
    elif feature_type == "myop":
        return jMyopulsePercentageRate(X, opts=None)
    elif feature_type == "ssi":
        return jSimpleSquareIntegral(X)
    elif feature_type == "vo":
        return jVOrder(X, opts=None)
    elif feature_type == "tm":
        return jTemporalMoment(X, opts=None)
    elif feature_type == "aac":
        return jAverageAmplitudeChange(X)
    elif feature_type == "mmav":
        return jModifiedMeanAbsoluteValue(X)
    elif feature_type == "mmav2":
        return jModifiedMeanAbsoluteValue2(X)
    elif feature_type == "iemg":
        return jIntegratedEMG(X)
    elif feature_type == "damv":
        return jDifferenceAbsoluteMeanValue(X)
    elif feature_type == "vare":
        return jVarianceOfEMG(X)
    elif feature_type == "wa":
        return jWillisonAmplitude(X, opts=None)
    elif feature_type == "ld":
        return jLogDetector(X)
    elif feature_type == "ar":
        return jAutoRegressiveModel(X, opts=None)
    elif feature_type == "zc":
        return jZeroCrossing(X, opts=None)
    elif feature_type == "ssc":
        return jSlopeSignChange(X, opts=None)
    elif feature_type == "wl":
        return jWaveformLength(X)
    elif feature_type == "mad":
        return jMeanAbsoluteDeviation(X)
    elif feature_type == "iqr":
        return jInterquartileRange(X)
    elif feature_type == "kurt":
        return jKurtosis(X)
    elif feature_type == "skew":
        return jSkewness(X)
    elif feature_type == "cov":
        return jCoefficientOfVariation(X)
    elif feature_type == "sd":
        return jStandardDeviation(X)
    elif feature_type == "var":
        return jVariance(X)
    elif feature_type == "ae":
        return jAverageEnergy(X)


"""
特征
num:40

"""


def j_new_zero_crossing(X):
    L = len(X)
    FZC = 0
    # Compute T (21)
    T = 4 * (1 / 10) * np.sum(X[:10])
    # Compute proposed zero crossing (20)
    for i in range(L - 1):
        if (X[i] > T and X[i + 1] < T) or (X[i] < T and X[i + 1] > T):
            FZC += 1
    return FZC


def j_enhanced_wave_length(X):
    L = len(X)
    EWL = 0
    for i in range(1, L):
        if 0.2 * L <= i <= 0.8 * L:
            p = 0.75
        else:
            p = 0.5
        EWL += abs((X[i] - X[i - 1]) ** p)
    return EWL


def j_enhanced_mean_absolute_value(X):
    L = len(X)
    Y = 0
    for i in range(L):
        if 0.2 * L <= i <= 0.8 * L:
            p = 0.75
        else:
            p = 0.5
        Y += abs(X[i] ** p)
    EMAV = Y / L
    return EMAV


def j_absolute_value_of_the_summation_of_exp_root(X):
    K = len(X)
    Y = 0
    for n in range(K):
        if 0.25 * K <= n <= 0.75 * K:
            exp = 0.5
        else:
            exp = 0.75
        Y += abs(X[n] ** exp)
    ASM = Y / K
    return ASM


def j_absolute_value_of_the_summation_of_square_root(X):
    temp = np.sum(np.sqrt(np.abs(X)))
    ASS = np.abs(temp)
    return ASS


def j_mean_value_of_the_square_root(X):
    K = len(X)
    MSR = (1 / K) * np.sum(np.sqrt(np.abs(X)))
    return MSR


def j_log_teager_kaiser_energy_operator(X):
    N = len(X)
    Y = 0
    for j in range(1, N - 1):
        Y += (X[j] ** 2) - X[j - 1] * X[j + 1]
    LTKEO = np.log(Y)
    return LTKEO


def j_log_coefficient_of_variation(X):
    mu = np.mean(X)
    sd = np.std(X)
    LCOV = np.log(sd / mu)
    return LCOV


def j_cardinality(X, opts=None):
    # 参数
    thres = 0.01  # 阈值

    if opts is not None and "thres" in opts:
        thres = opts["thres"]

    N = len(X)
    # 排序数据
    Y = np.sort(X)
    Z = np.abs(Y[:-1] - Y[1:]) > thres
    CARD = np.sum(Z)
    return CARD


def j_log_difference_absolute_standard_deviation_value(X):
    N = len(X)
    Y = 0
    for t in range(N - 1):
        Y += (X[t + 1] - X[t]) ** 2
    LDASDV = np.log(np.sqrt(Y / (N - 1)))
    return LDASDV


def j_difference_absolute_standard_deviation_value(X):
    N = len(X)
    Y = 0
    for i in range(N - 1):
        Y += (X[i + 1] - X[i]) ** 2
    DASDV = np.sqrt(Y / (N - 1))
    return DASDV


def j_root_mean_square(X):
    RMS = np.sqrt(np.mean(X**2))
    return RMS


def j_mean_absolute_value(X):
    MAV = np.mean(np.abs(X))
    return MAV


def tdpsd(X):
    # 计算均方根
    avm0 = np.sqrt(np.sum(X**2))

    # 计算一阶差分和二阶差分
    diff1 = np.diff(X)
    diff2 = np.diff(diff1)

    # 计算一阶差分和二阶差分的均方根
    avm2 = np.sqrt(np.sum(diff1**2))
    avm4 = np.sqrt(np.sum(diff2**2))

    # 归一化
    m0 = avm0**0.1 / 0.1
    m2 = avm2**0.1 / 0.1
    m4 = avm4**0.1 / 0.1

    # 计算 f1
    f1 = np.log(m0)

    return f1


def median_frequency(X):
    # 计算频谱
    f, P = np.fft.fftfreq(len(X)), np.abs(np.fft.fft(X)) ** 2

    # 计算累积分布
    cumulative_P = np.cumsum(P)

    # 找到累积分布超过一半的频率索引
    for i in range(len(f)):
        if cumulative_P[i] > np.sum(P) / 2:
            break

    # 计算中值频率
    mdf = f[i]
    return mdf


def mean_power_frequency(X):
    # 计算频谱
    f, P = np.fft.fftfreq(len(X)), np.abs(np.fft.fft(X)) ** 2

    # 计算频率加权的能量和
    numerator = np.sum(f * P)

    # 计算总能量
    denominator = np.sum(P)

    # 防止除以零
    if denominator == 0:
        return 0
    else:
        mpf = numerator / denominator
        return mpf


def jMaximumFractalLength(X):
    N = len(X)
    Y = 0
    for n in range(N - 1):
        Y += (X[n + 1] - X[n]) ** 2
    MFL = np.log10(np.sqrt(Y))
    return MFL


def jMyopulsePercentageRate(X, opts=None):
    # 参数
    thres = 0.016  # 阈值

    # 如果提供了opts并且包含'thres'，则使用opts中的值
    if opts is not None and "thres" in opts:
        thres = opts["thres"]

    N = len(X)
    Y = 0
    for i in range(N):
        if abs(X[i]) >= thres:
            Y += 1
    MYOP = Y / N
    return MYOP


def jSimpleSquareIntegral(X):
    SSI = np.sum(X**2)
    return SSI


def jVOrder(X, opts=None):
    # 参数
    order = 2  # 阶数

    # 如果提供了opts并且包含'order'，则使用opts中的值
    if opts is not None and "order" in opts:
        order = opts["order"]

    N = len(X)
    Y = (1 / N) * np.sum(X**order)
    VO = Y ** (1 / order)
    return VO


def jTemporalMoment(X, opts=None):
    # 参数
    order = 3  # 阶数

    # 如果提供了opts并且包含'order'，则使用opts中的值
    if opts is not None and "order" in opts:
        order = opts["order"]

    N = len(X)
    TM = abs((1 / N) * np.sum(X**order))
    return TM


def jAverageAmplitudeChange(X):
    N = len(X)
    Y = 0
    for i in range(N - 1):
        Y += abs(X[i + 1] - X[i])
    AAC = Y / N
    return AAC


def jModifiedMeanAbsoluteValue(X):
    N = len(X)
    Y = 0
    for i in range(N):
        if i >= 0.25 * N and i <= 0.75 * N:
            w = 1
        else:
            w = 0.5
        Y += w * abs(X[i])
    MMAV = (1 / N) * Y
    return MMAV


def jModifiedMeanAbsoluteValue2(X):
    N = len(X)
    Y = 0
    for i in range(N):
        if i >= 0.25 * N and i <= 0.75 * N:
            w = 1
        elif i < 0.25 * N:
            w = (4 * i) / N
        else:
            w = 4 * (i - N) / N
        Y += w * abs(X[i])
    MMAV2 = (1 / N) * Y
    return MMAV2


def jIntegratedEMG(X):
    IEMG = np.sum(np.abs(X))
    return IEMG


def jDifferenceAbsoluteMeanValue(X):
    N = len(X)
    Y = 0
    for i in range(N - 1):
        Y += abs(X[i + 1] - X[i])
    DAMV = Y / (N - 1)
    return DAMV


def jVarianceOfEMG(X):
    N = len(X)
    VAR = (1 / (N - 1)) * np.sum(X**2)
    return VAR


def jWillisonAmplitude(X, opts=None):
    # 参数
    thres = 0.01  # 阈值

    # 如果提供了opts并且包含'thres'，则使用opts中的值
    if opts is not None and "thres" in opts:
        thres = opts["thres"]

    N = len(X)
    WA = 0
    for k in range(N - 1):
        if abs(X[k] - X[k + 1]) > thres:
            WA += 1
    return WA


def jLogDetector(X):
    N = len(X)
    Y = 0
    for k in range(N):
        Y += np.log(np.abs(X[k]))
    LD = np.exp(Y / N)
    return LD


def jAutoRegressiveModel(X, opts=None):
    # 参数
    order = 4  # 阶数

    # 如果提供了opts并且包含'order'，则使用opts中的值
    if opts is not None and "order" in opts:
        order = opts["order"]

    # 使用Levinson-Durbin算法估计AR模型的参数
    Y, _, _ = levinson_durbin(X, order)

    # 第一个索引没有意义，所以我们取从第二个到最后一个索引的参数
    AR = Y[1 : order + 1]
    return AR


def jZeroCrossing(X, opts=None):
    # 参数
    thres = 0.01  # 阈值

    # 如果提供了opts并且包含'thres'，则使用opts中的值
    if opts is not None and "thres" in opts:
        thres = opts["thres"]

    N = len(X)
    ZC = 0
    for k in range(N - 1):
        if ((X[k] > 0 and X[k + 1] < 0) or (X[k] < 0 and X[k + 1] > 0)) and abs(
            X[k] - X[k + 1]
        ) >= thres:
            ZC += 1
    return ZC


def jSlopeSignChange(X, opts=None):
    # 参数
    thres = 0.01  # 阈值

    # 如果提供了opts并且包含'thres'，则使用opts中的值
    if opts is not None and "thres" in opts:
        thres = opts["thres"]

    N = len(X)
    SSC = 0
    for k in range(1, N - 1):
        if (
            (X[k] > X[k - 1] and X[k] > X[k + 1])
            or (X[k] < X[k - 1] and X[k] < X[k + 1])
        ) and ((abs(X[k] - X[k + 1]) >= thres) or (abs(X[k] - X[k - 1]) >= thres)):
            SSC += 1
    return SSC


def jWaveformLength(X):
    N = len(X)
    WL = 0
    for k in range(1, N):
        WL += abs(X[k] - X[k - 1])
    return WL


def jMeanAbsoluteDeviation(X):
    N = len(X)
    # 均值
    mu = np.mean(X)
    # 平均绝对偏差
    MAD = (1 / N) * np.sum(np.abs(X - mu))
    return MAD


def jInterquartileRange(X):
    # 计算四分位距
    IQR = np.subtract(*np.percentile(X, [75, 25]))
    return IQR


def jKurtosis(X):
    # 计算峰度
    KURT = kurtosis(X)
    return KURT


def jSkewness(X):
    # 计算偏度
    SKEW = skew(X)
    return SKEW


def jCoefficientOfVariation(X):
    # 计算标准差和均值
    std_dev = np.std(X)
    mean_val = np.mean(X)
    # 计算变异系数
    COV = std_dev / mean_val
    return COV


def jStandardDeviation(X):
    N = len(X)
    mu = np.mean(X)
    SD = np.sqrt((1 / (N - 1)) * np.sum((X - mu) ** 2))
    return SD


def jVariance(X):
    N = len(X)
    mu = np.mean(X)
    VAR = (1 / (N - 1)) * np.sum((X - mu) ** 2)
    return VAR


def jAverageEnergy(X):
    ME = np.mean(X**2)
    return ME
