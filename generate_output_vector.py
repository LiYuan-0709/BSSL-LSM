# -*- encoding: utf-8 -*-
'''
file       :generate_output_vector.py
Description:生成状态向量
Date       :2023/03/20 10:46:02
Author     :li-yuan
Email      :ly1837372873@163.com
'''

# here put the import lib

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
      
print('generate_output_vector')

num_neuron = int(sys.argv[2])
time_work = int(sys.argv[3])
time_rest = int(sys.argv[4])
train_number = int(sys.argv[5])
test_number = int(sys.argv[6])
ex_rate = float(sys.argv[7])
ex_num = int(ex_rate*num_neuron)

time_index_path = './spike_record/3second_9direction/'
train_vector_savepath = './vector/3second_9direction/RC_state_vector_training'
test_vector_savepath = './vector/3second_9direction/RC_state_vector_testing'



train_vector = np.zeros((train_number, num_neuron))
index_array1 = np.load(time_index_path +'index_array1_train.npy')
time_array1 = np.load(time_index_path +'time_array1_train.npy')
index_array2 = np.load(time_index_path +'index_array2_train.npy')
time_array2 = np.load(time_index_path +'time_array2_train.npy')

for i in range(len(index_array1)):
    j = int(time_array1[i]*1000/(time_work+time_rest))
    train_vector[j][index_array1[i]] += 1
for i in range(len(index_array2)):
    j = int(time_array2[i]*1000/(time_work+time_rest))
    train_vector[j][index_array2[i]+ex_num] += 1
# 有800个兴奋性神经元和200个抑制神经元，这一步统计每一个样本输入时，、
# 各个兴奋性和抑制性神经元发放脉冲的总数量来作为输出向量。

# # 归一化操作
# for i in range(train_number):
#     if train_vector[i].max() != 0:
#         train_vector[i][:] = (train_vector[i][:]*1.0/train_vector[i].max())
# 分类归一化操作
for i in range(train_number):
    if train_vector[i].max() != 0:
        train_vector[i][:ex_num] = (train_vector[i][:ex_num]*1.0/train_vector[i][:ex_num].max())
        train_vector[i][ex_num:] = (train_vector[i][ex_num:]*1.0/train_vector[i][ex_num:].max())

np.save(train_vector_savepath, train_vector)



test_vector = np.zeros((test_number, num_neuron), float)
index_array1 = np.load(time_index_path +'index_array1_test.npy')
time_array1 = np.load(time_index_path +'time_array1_test.npy')
index_array2 = np.load(time_index_path +'index_array2_test.npy')
time_array2 = np.load(time_index_path +'time_array2_test.npy')
for i in range(len(index_array1)):
    j = int(time_array1[i]*1000/(time_work+time_rest))
    test_vector[j][index_array1[i]] += 1.0
for i in range(len(index_array2)):
    j = int(time_array2[i]*1000/(time_work+time_rest))
    test_vector[j][index_array2[i]+ex_num] += 1.0

for i in range(test_number):
    if test_vector[i].max() != 0:
        test_vector[i][:ex_num] = (test_vector[i][:ex_num]*1.0/test_vector[i][:ex_num].max())
        test_vector[i][ex_num:] = (test_vector[i][ex_num:]*1.0/test_vector[i][ex_num:].max())

np.save(test_vector_savepath, test_vector)



# 移除实验过程中产生的中间文件
print('delete file')
del_list = os.listdir('./output/results')
for f in del_list:
    file_path = os.path.join('./output/results', f)
    if os.path.isfile(file_path):
        os.remove(file_path)
