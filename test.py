from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def timestamp_to_time(timestamp):
    # 将时间戳转换为datetime对象
    dt_object = datetime.fromtimestamp(timestamp)

    # 将datetime对象格式化为可读的时间字符串
    time_string = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    return time_string


# 原始矩阵
data = np.genfromtxt('data\LTCUSDT5m2023-06-162023-07-17.csv', delimiter=',', skip_header=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[:, 6:])

# 定义函数进行点乘


def multiply_rows(matrix):
    matrix = np.array([matrix])
    return np.multiply(matrix, matrix.T)


input_data = []
input_y = []
window_size = 23
for i in range(len(scaled_data) - (window_size+1)):
    templatebox = []
    for j in range(window_size):
        templatebox.append(multiply_rows(scaled_data[i+j, :]))

    # 添加templatebox到input_data里面
    input_data.append(templatebox)

    if scaled_data[i+window_size+1, 0] - scaled_data[i+window_size, 0] > 0:
        input_y.append(0)
    else:
        input_y.append(1)

input_data = np.array(input_data)
input_y = np.array(input_y)
loaded_model = load_model('cnn_model.h5')
predictions = loaded_model.predict(input_data)
for i in range(len(predictions)):
    weighted_prob_1 = predictions[i][0]
    weighted_prob_2 = predictions[i][1]

    # 计算总和
    total = abs(weighted_prob_1) + abs(weighted_prob_2)

    # 计算情况1的概率
    prob_1 = abs(weighted_prob_1) / total

    # 计算情况2的概率
    prob_2 = abs(weighted_prob_2) / total


    t = data[:, 0][i + window_size] / 1000
    timestamp = timestamp_to_time(t)
    result = "上涨" if predictions[i][0] > predictions[i][1] else "下跌" if predictions[i][1] > predictions[i][0] else "不明显"

    print(f"时间戳：{timestamp}")
    print(f"预测值：{predictions[i]}")
    print(f"趋势：{result}")
    print(f"情况1的概率：{prob_1:.2f}，情况2的概率：{prob_2:.2f}\n")
