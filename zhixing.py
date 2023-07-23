# -*- encoding: utf-8 -*-
'''
file       :zhixing.py
Description:执行脚本
Date       :2023/03/20 10:35:53
Author     :li-yuan
Email      :ly1837372873@163.com
'''

import os
import sys
from time import time
import numpy as np
import random
from numpy.lib.function_base import piecewise

# # PSO搜索算法参数设置
'''
九分类搜索所得最佳参数
Parameters: 1290 0.8654354949391924 1.0566543699178226 0.41445214574975314 0.30796531268035343 1.251532833155424 0.5667298094206488 
1.1332479573659142 0.466417183455394 1.0566543699178226 0.314627986758363 0.6369516076696071
Acc: [0.82222222] 

852 0.38789709770174075 0.2451269419449072 0.7412090206349539 0.7655599141925946
'''

num_neuron = int(sys.argv[1])
excite_rate = float(0.8)
inhibite_rate = 1.0 - excite_rate


# 输入神经元到水库兴奋神经元的权重及连接概率
weight_ie = float(1.0)
w_ee = float(1.0)
w_ei = float(1.0)
w_ie = float(1.0)
w_ii = float(1.0)
p_inpute = float(sys.argv[2])
p_ee = float(sys.argv[3])
p_ei = float(sys.argv[4])
p_ie = float(sys.argv[5])
p_ii = float(sys.argv[6])

# 分类器超参数设置
hidden_size = float(sys.argv[7])
weight_decay = float(sys.argv[8])
step_size = int(sys.argv[9])
gamma = float(sys.argv[10])
t = float(sys.argv[11])


# 输入数据的持续时间和中间的空白窗口的持续时间
time_work = 600
time_rest = 0

# 训练和测试数据的数量
train_number = 500*9
test_number = 100*9

# 输入神经元的个数
channel = 200

acc_all = []

k_fold = os.listdir(".\\spikes_input\\3second_9direction")[-1:]
print(k_fold)
for k in k_fold:
        print(num_neuron)
        print(hidden_size)
        os.system('python RC_generater.py %d %d %d %d %f %f %f %f %f %f %f %f %f %f %d %f %f %s' % (num_neuron, time_work,
                time_rest, train_number, weight_ie, p_inpute, p_ee, w_ee, p_ei, w_ei, p_ie, w_ie, p_ii, w_ii, channel, excite_rate, inhibite_rate, k))
        os.system('python RC_test.py %d %d %d %d %d %f %f %s' % (num_neuron, time_work, time_rest, test_number, channel, excite_rate, inhibite_rate, k))
        os.system('python generate_output_vector.py %d %d %d %d %d %d %f' %
                (channel, num_neuron, time_work, time_rest, train_number, test_number, excite_rate))
        os.system('python classifier_new_copy_copy.py %d %f %d %f %f %s %d' % (hidden_size, weight_decay, step_size, gamma, t, k, num_neuron))

        acc_all.append(np.load('test_auc.npy'))
        print(acc_all)
        os.remove('test_auc.npy')

print(acc_all)
mean_acc = np.sum(acc_all)/len(acc_all)
print(mean_acc)
np.save("./mean_test_auc",mean_acc)


