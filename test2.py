#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test2.py
@Time    :   2023/05/06 20:43:07
@Contact :   liyuan22@nudt.edu.cn
@Desc    :   BO-opt 代码
'''

import optuna

import hyperopt

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import KFold,cross_validate
from bayes_opt import BayesianOptimization
from hyperopt import hp,fmin,tpe,Trials,partial,STATUS_OK
from hyperopt.early_stop import no_progress_loss
import numpy as np
import time
from aimFunction import function_wsy
import os

def GetTime(func_name):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func_name(*args, **kwargs)
        end_time = time.time()
        run_time = end_time - start_time
        print("pso算法运行时间为%f秒" % run_time)
        return ret
    return wrapper


# 设定参数空间：
param_space = {'num_neuron':hp.quniform('num_neuron', 200,1000,1)
,'p_input':hp.uniform('p_input', 0.1,0.9)
,'p_ee':hp.uniform('p_ee', 0.1, 0.9)
,'p_ei':hp.uniform('p_ei', 0.1, 0.9)
,'p_ie':hp.uniform('p_ie', 0.1, 0.9)
,'p_ii':hp.uniform('p_ii', 0.1, 0.9)
,'hidden_size':hp.quniform('hidden_size',32, 500,1)
,'weight_decay':hp.uniform('weight_decay',0.001, 0.01)
,'gamma':hp.uniform('gamma',0.05, 0.5)}
 

# 设置目标函数：
def hyperopt_objfunction(params):
    num_neuron = int(params['num_neuron'])
    p_input = float(params['p_input'])
    p_ee = float(params['p_ee'])
    p_ei = float(params['p_ei'])
    p_ie = float(params['p_ie'])
    p_ii = float(params['p_ii'])
    hidden_size = int(params['hidden_size'])
    weight_decay = float(params['weight_decay'])
    gamma = float(params['gamma'])

    if not os.path.exists("./test_auc.npy"):
        os.system("python zhixing.py  %d %f %f %f %f %f %d %f %f" % (num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    # 从log中提取准确率：
    accu = np.load("test_auc.npy")
    print('当前的参数为：%s %s %s %s %s %s %s %s %s \n'%(num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    print("准确率为：%s"%(accu))
    
    file_handle=open('recorder_hyperopt.txt',mode='a')
    file_handle.write('Parameters: %s %s %s %s %s %s %s %s %s \n'%(num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    file_handle.write("Acc: %s \n"%(accu))

    if os.path.exists("./test_auc.npy"):
        os.remove("./test_auc.npy")
    return {"loss":1-accu, "status": STATUS_OK}

# 设定优化过程
def param_hyperopt(max_evals):
    # 记录迭代过程
    trials=Trials()
    
    # 提前停止
    early_stop_fn=no_progress_loss(30) # 当损失函数的连续迭代100次都没有下降时，则停止；正常10-50即可
    
    # 定义代理模型
    # algo=partial(tpe.suggest # 设置代理模型的算法
		#	,n_sratup_jobs=20 # 设置初始样本量
		#	,n_EI_candidates=50) # 设置使用多少样本点来计算采集函数
    params_best=fmin(hyperopt_objfunction # 设定目标函数
                     ,space=param_space # 设定参数空间
                     ,algo=tpe.suggest # 设定代理模型，如果需要自定义代理模型，使用前面algo=……的代码
                     ,max_evals=max_evals # 设定迭代次数
                     ,trials=trials 
                     ,early_stop_fn=early_stop_fn # 控制提前停止
                    )
    
    # print('best parmas:',params_best)
    # params_best = eval(params_best)
    return params_best,trials

# 设定验证函数(和设定目标函数的代码一致)
def hyperopt_validation(params):
    num_neuron = int(params['num_neuron'])
    p_input = float(params['p_input'])
    p_ee = float(params['p_ee'])
    p_ei = float(params['p_ei'])
    p_ie = float(params['p_ie'])
    p_ii = float(params['p_ii'])
    hidden_size = int(params['hidden_size'])
    weight_decay = float(params['weight_decay'])
    gamma = float(params['gamma'])

    if not os.path.exists("./test_auc.npy"):
        os.system("python zhixing.py  %d %f %f %f %f %f %d %f %f" % (num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    # 从log中提取准确率：
    accu = np.load("test_auc.npy")
    print('当前的参数为：%s %s %s %s %s %s %s %s %s \n'%(num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    print("准确率为：%s"%(accu))
    
    file_handle=open('recorder_hyperopt.txt',mode='a')
    file_handle.write('__________________________validation__________________')
    file_handle.write('Parameters: %s %s %s %s %s %s %s %s %s \n'%(num_neuron, p_input, p_ee, p_ei, p_ie, p_ii, hidden_size, weight_decay, gamma))
    file_handle.write("Acc: %s \n"%(accu))

    if os.path.exists("./test_auc.npy"):
        os.remove("./test_auc.npy")
    return 1.0-accu


# 执行实际优化流程

# # 1. 计算1%空间时的优化过程，返回最佳参数组合和迭代过程
# params_best,trials=param_hyperopt(30) 

# #2. 计算3%空间时的优化过程，返回最佳参数组合和迭代过程
# params_best, trials = param_hyperopt(100)

#3. 计算10%空间时的优化过程，返回最佳参数组合和迭代过程
params_best, trials = param_hyperopt(10)

# 根据最佳参数组合验证模型，返回RMSE
hyperopt_validation(params_best)
# best_value = hyperopt_validation(params_best)
# print(best_value)

#打印所有搜索相关的记录
trials.trials[0]
trials.losses()[:10]
# print()

# #打印全部搜索的目标函数值
# print()













# # 设定参数空间
# param_grid_simple={'n_estimators':hp.quniform('n_estimators',80,100,1)
#                    ,'max_depth':hp.quniform('max_depth',10,25,1)
#                    ,'max_features':hp.quniform('max_features',10,20,1)
#                    ,'min_impurity_decrease':hp.quniform('min_impurity_decrease',20,25,1)
#                   }

# # 计算参数空间的大小
# len([*range(80,100,1)])*len([*range(10,25,1)])*\
# len([*range(10,20,1)])*len([range(20,25,1)])
# # 20*15*10*5


# # 设定目标函数_基评估器选择随机森林
# def hyperopt_objective(params):
#     model=RFR(n_estimators=int(params['n_estimators'])
#               ,max_depth=int(params['max_depth'])
#               ,max_features=int(params['max_features'])
#               ,min_impurity_decrease=params['min_impurity_decrease']
#               ,random_state=7
#               ,n_jobs=4)
    
#     cv=KFold(n_splits=5,shuffle=True,random_state=7)
#     validate_loss=cross_validate(model,X,y
#                                    ,cv=cv
#                                    ,scoring='neg_root_mean_squared_error'
#                                    ,n_jobs=-1
#                                    ,error_score='raise')
    
#     return np.mean(abs(validate_loss['test_score']))


# # 设定优化过程
# def param_hyperopt(max_evals=100):
#     # 记录迭代过程
#     trials=Trials()
    
#     # 提前停止
#     early_stop_fn=no_progress_loss(100) # 当损失函数的连续迭代100次都没有下降时，则停止；正常10-50即可
    
#     # 定义代理模型
#     # algo=partial(tpe.suggest # 设置代理模型的算法
# 		#	,n_sratup_jobs=20 # 设置初始样本量
# 		#	,n_EI_candidates=50) # 设置使用多少样本点来计算采集函数
#     params_best=fmin(hyperopt_objective # 设定目标函数
#                      ,space=param_grid_simple # 设定参数空间
#                      ,algo=tpe.suggest # 设定代理模型，如果需要自定义代理模型，使用前面algo=……的代码
#                      ,max_evals=max_evals # 设定迭代次数
#                      ,trials=trials 
#                      ,early_stop_fn=early_stop_fn # 控制提前停止
#                     )
    
#     print('best parmas:',params_best)
#     return params_best,trials

# # 设定验证函数(和设定目标函数的代码一致)
# def hyperopt_validation(params):
#     model=RFR(n_estimators=int(params['n_estimators'])
#               ,max_depth=int(params['max_depth'])
#               ,max_features=int(params['max_features'])
#               ,min_impurity_decrease=params['min_impurity_decrease']
#               ,random_state=7
#               ,n_jobs=4)
        
#     cv=KFold(n_splits=5,shuffle=True,random_state=7)
#     validate_loss=cross_validate(model,X,y
#                                    ,cv=cv
#                                    ,scoring='neg_root_mean_squared_error'
#                                    ,n_jobs=-1
#                                   )
    
#     return np.mean(abs(validate_loss['test_score']))


# # 执行实际优化流程

# # 1. 计算1%空间时的优化过程，返回最佳参数组合和迭代过程
# params_best,trials=param_hyperopt(30) 

# #2. 计算3%空间时的优化过程，返回最佳参数组合和迭代过程
# params_best, trials = param_hyperopt(100)

# #3. 计算10%空间时的优化过程，返回最佳参数组合和迭代过程
# params_best, trials = param_hyperopt(300)

# # 根据最佳参数组合验证模型，返回RMSE
# hyperopt_validation(params_best)

# #打印所有搜索相关的记录
# trials.trials[0]

# #打印全部搜索的目标函数值
# trials.losses()[:10]
