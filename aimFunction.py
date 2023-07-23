# -*- encoding: utf-8 -*-
'''
file       :aimFunction.py
Description:
Date       :2023/03/23 16:56:52
Author     :li-yuan
Email      :ly1837372873@163.com
'''
import os
import numpy as np
import re

def function_wsy(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11): 
    num_neuron = int(x1)
    p_inpute = float(x2)
    p_ee = float(x3)
    p_ei = float(x4)
    p_ie = float(x5)
    p_ii = float(x6)
    hidden_size = int(x7)
    weight_decay = float(x8)
    step_size = int(x9)
    gamma = float(x10)
    t = float(x11)


    # 如果log不存在，则重新产生RC结构并运行,产生log,否则跳过此步直接读取；
    # if not os.path.exists("./test_auc.npy"):
    #     os.system("python zhixing.py  "+str(p_ie))
    if not os.path.exists("./mean_test_auc.npy"):
        os.system("python zhixing.py  %d %f %f %f %f %f %d %f %d %f %f" % 
                  (num_neuron, p_inpute, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, step_size,
                   gamma, t))
  
    # 从log中提取准确率：
    accu = np.load("./mean_test_auc.npy")
    print("准确率为：%s"%(accu))
    
    print('当前的参数为：%s %s %s %s %s %s %s %s %s %s %s \n'%(num_neuron, p_inpute, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, step_size,
                   gamma, t))
    
    file_handle=open('recorder_PSO.txt',mode='a')
    file_handle.write('Parameters: %s %s %s %s %s %s %s %s %s %s %s\n'%(num_neuron, p_inpute, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, step_size,
                   gamma, t))
    file_handle.write("Acc: %s \n"%(accu))
    file_handle.close()

    if os.path.exists("./mean_test_auc.npy"):
        os.remove("./mean_test_auc.npy")
    return accu
