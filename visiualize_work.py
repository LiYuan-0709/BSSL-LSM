# -*- encoding: utf-8 -*-
'''
file       :visiualize_work.py
Description:histogram_plot
Date       :2023/03/29 22:18:50
Author     :li-yuan
Email      :ly1837372873@163.com
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

'''
9分类,分辨率是10°的情况下:
'''
time_work = 0.6
input_nueron = 200

# 输入层神经元的index & time 单位是毫秒
# index_input = np.load('spikes_input\\3second_9direction\\train_index.npy')
# time_input = np.load('spikes_input\\3second_9direction\\train_time.npy')

# 输入数据的label
label = np.load('spikes_input\\3second_9direction\\k5\\train_label.npy')

# 液体层兴奋性神经元的索引和发放时间 单位是秒
index_e = np.load('spike_record\\3second_9direction\\index_array1_train.npy')
time_e = np.load('spike_record\\3second_9direction\\time_array1_train.npy')

# 液体层抑制性神经元的索引和发放时间
index_i = np.load('spike_record\\3second_9direction\\index_array2_train.npy')
time_i = np.load('spike_record\\3second_9direction\\time_array2_train.npy')

def plot_scatter(index_file, time_file, xlabel, ylabel, name):
    x = time_file[time_e <= 0.6]
    y = index_file[:len(x)]
    plt.scatter(x, y, marker='.')
    plt.title(name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()

# 对输入层各类别做一个统计
label = label.tolist()
label = list(enumerate(label))


# 对 0° 方向的输入层神经元做统计
class_0_list = [i for i,x in label if x==40]
print(class_0_list)
# print(time_input)

# plt.figure()
# for i in range(5):
#     plt.subplot(5,3, i*3+1)
#     vector = np.zeros(input_nueron)
#     x_before = time_input[time_input <= class_0_list[i]*600]
#     x_after = time_input[time_input <= (class_0_list[i] + 1)*600]
#     x_label = time_input[len(x_before):len(x_after)]
#     y_label = index_input[len(x_before):len(x_after)]
#     print(y_label)
#     plt.scatter(x_label,y_label,marker='.',color = 'black')

#     # 对 0° 方向的输入层神经元做直方图统计
#     plt.subplot(5,3,i*3+2)
#     for k in y_label:
#         vector[k] += 1
#     # plt.plot(vector,linestyle='--', color='y')
#     plt.bar(range(200), vector)

#     # 对0° 方向神经元的放电次数的总体统计
#     plt.subplot(5,3,i*3+3)
#     plt.hist(vector)
# plt.show()

# 对0° 方向的液体层神经元做统计
# 将液体层的兴奋性神经元和抑制性神经元画在同一个直方图中做输出
plt.figure()
for i in range(5):
    vector_1 = np.zeros(int(1290*0.8654354949391924))
    # vector_2 = np.zeros(int(1290*(1-0.8654354949391924)))
    vector_2 = np.zeros(1290)
    # 直接绘图
    plt.subplot(5,3, i*3+1)
    x_before = time_e[time_e <= class_0_list[i]*0.6]
    x_after = time_e[time_e <= (class_0_list[i] + 1)*0.6]
    x_label_1 = time_e[len(x_before):len(x_after)]
    y_label_1 = index_e[len(x_before):len(x_after)]
    plt.scatter(x_label_1,y_label_1,marker='.',color = 'black')
    x_before_i = time_i[time_i <= class_0_list[i]*0.6]
    x_after_i = time_i[time_i <= (class_0_list[i] + 1)*0.6]
    x_label_2 = time_i[len(x_before_i):len(x_after_i)]
    y_label_2 = index_i[len(x_before_i):len(x_after_i)]+len(vector_1)
    plt.scatter(x_label_2,y_label_2,marker='.',color= 'r')

    # 对 0° 方向的液体层神经元做直方图统计
    plt.subplot(5,3, i*3+2)
    for m in y_label_1:
        vector_1[m] += 1
    plt.bar(range(len(vector_1)), vector_1, color='b')
    for n in y_label_2:
        vector_2[n] += 1
    plt.bar(range(len(vector_2)), vector_2, color='r')

    # 对 0° 方向液体层神经元的放电次数的总体统计
    plt.subplot(5,3,i*3+3)
    plt.hist(vector_1, color='b')
    plt.hist(vector_2[len(vector_1):], color='r')

plt.show()