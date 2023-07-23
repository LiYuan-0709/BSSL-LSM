# -*- encoding: utf-8 -*-
'''
file       :classifier.py
Description:写一个简单的分类器
Date       :2023/03/22 14:11:33
Author     :li-yuan
Email      :ly1837372873@163.com
'''

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.preprocessing import LabelEncoder
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sys
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


torch.manual_seed(1)  # reproducible
print("CLASSIFIER START!!")
epochs = 150
classes = 9
first_layer_num = int(sys.argv[7])

hidden_size = int(sys.argv[1])
weight_decay = float(sys.argv[2])
step_size = int(sys.argv[3])
gamma = float(sys.argv[4])
t = float(sys.argv[5])
k = sys.argv[6]

train_data_path = "vector\\3second_"+str(9)+"direction\\RC_state_vector_training.npy"
train_label_path = "spikes_input\\3second_"+str(9)+"direction\\"+k+"\\train_label.npy"
test_data_path = "vector\\3second_"+str(9)+"direction\\RC_state_vector_testing.npy"
test_label_path = "spikes_input\\3second_"+str(9)+"direction\\"+k+"\\label_test.npy"

train_data = np.load(train_data_path)
train_data.reshape(-1,1)

label_training = np.load(train_label_path)
label_training.reshape(-1,1)
azimuth_list = []
for label in label_training:
    amzuith = label
    azimuth_list.append(amzuith)
azimuth_list = np.array(azimuth_list)
print(azimuth_list)
azimuth_list = LabelEncoder().fit_transform(azimuth_list)
print(azimuth_list)


test_data = np.load(test_data_path)
test_data.reshape(-1,1)

label_testing = np.load(test_label_path)
label_testing.reshape(-1,1)
azimuth_list_test = []
for label_test in label_testing:
    amzuith_test = label_test
    azimuth_list_test.append(amzuith_test)
azimuth_list_test = np.array(azimuth_list_test)
azimuth_list_test = LabelEncoder().fit_transform(azimuth_list_test)

print(train_data)

'''数据集预处理'''
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float() #加载npy数据
        self.label = torch.from_numpy(label)
    def __getitem__(self, index):
        single_data = self.data[index]  # 读取每一个npy的数据
        single_label = self.label[index]
        return single_data,single_label #返回数据还有标签
    def __len__(self):
        return self.data.shape[0] #返回数据的总个数
    
def plot_confusion_matrix(cm, labels_name, title,thresh=0.8):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行


'''网络结构定义 - 9/5分类最佳参数'''
if classes==9:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(first_layer_num, hidden_size)
            self.fc4 = nn.Linear(hidden_size,classes)
            self.relu = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(hidden_size)
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.bn2(x)

            x = self.fc4(x)
            return x
else:
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(first_layer_num, 512)
            self.fc3 = nn.Linear(512,5)
            self.dropout = nn.Dropout(0.2)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = F.relu(self.fc1(x))
            # x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x
        
 
if __name__ == '__main__':
    # 1、训练与测试数据集准备与加载
    train_data_set = MyDataset(train_data,azimuth_list)
    train_data= DataLoader(train_data_set, batch_size=64, shuffle=True, pin_memory=True)
    print(train_data_set.__len__())

    test_data_set = MyDataset(test_data,azimuth_list_test)
    test_data = DataLoader(test_data_set,batch_size=64, shuffle=True, pin_memory=True)

    labels_name = [-40, -30, -20, -10, 0, 10, 20, 30, 40]

    RESUME = 0


    # 2、网络模型的搭建
    net = Net()

    criterion = nn.CrossEntropyLoss()#交叉熵损失函数
    # optim里面只有L2正则化，同时这个正则化对权重和偏置同时进行正则化，但是对偏置进行正则化可能会导致欠拟合，需要进一步分析
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    '''
    L1正则化代码:
    regularization_loss = 0.0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))

    '''
    # optimizer = optim.SGD(net.parameters(), lr=0.01)

    start_epoch = -1

    if RESUME:
        path_checkpoint = "ckpt_best_159.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        # print(checkpoint)

        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch


    correct_rate_train = []
    correct_rate = [-0.01]
    acc = []
    # 3、网络训练
    run_loss = 0.0
    correct = 0
    run_loss_val = 0.0
    correct_val = 0

    # lam = 0.0001
    pi = 3.1415926535
    # t = 180.0
    t = t
    theta0 = torch.tensor([-40, -30, -20, -10, 0, 10, 20, 30, 40], dtype=torch.float32).unsqueeze(dim=0)  # [1, c]
    # print('theta0:', theta0)
    for epoch in range(start_epoch ,epochs+start_epoch):
        # print(optimizer)
        for i, data in enumerate(train_data):
            input_data, input_label = data
            input_label = input_label.long()
            optimizer.zero_grad()
            # print(input_data.shape)
            outputs = net(input_data)
            # print(outputs.shape)
            # outputs = nn.LogSoftmax(dim=1)(outputs)

            # class_loss = criterion(outputs, input_label)
            # class_loss = angle_softmax(labels_name, input_label, outputs)

            ###########################################
            theta = theta0[0, :][input_label].float().unsqueeze(dim=1)  # [n, 1]
            delta = theta0 - theta  # [n, c]
            # print(delta.shape)
            # print("delta:",input_label[0], delta[0])
            cos_delta = torch.cos(delta * pi / 180.0)
            # print("cos_delta:",input_label[0], cos_delta[0])
            p = F.softmax(cos_delta * t, dim=1)
            # print("p:",p)
            # if epoch == start_epoch and i == 0:
            #     print(max(p[0]).item())
            log_q = F.log_softmax(outputs, dim=1)
            # print("log_q:",log_q)
            loss = torch.sum(-p.mul(log_q), dim=1).mean()


            ###########################################
            
            # loss = class_loss
            # loss = torch.tensor(class_loss, requires_grad=True)
            loss.backward()
            optimizer.step()

            correct += (outputs.argmax(1) == input_label).type(torch.float).sum().item()
            run_loss += loss.item()
        if(epoch+1)%10==0:
            print('epoch:{} run_loss:{} correct_train_rate:{}'.format(epoch+1, run_loss/train_data_set.__len__(), correct/train_data_set.__len__()))
        correct_rate_train.append(correct/train_data_set.__len__())
        run_loss = 0.0
        correct = 0
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.65)
        scheduler.step()

        
        net.eval()

        with torch.no_grad():
            for i, data_test in enumerate(test_data):
                input_data_test, input_label_test = data_test
                input_label_test = input_label_test.long()
                # optimizer.zero_grad()

                outputs_test = net(input_data_test)

                loss_val = criterion(outputs_test, input_label_test)
                # loss_val.backward()
                # optimizer.step()

                correct_val += (outputs_test.argmax(1) == input_label_test).type(torch.float).sum().item()
                run_loss_val += loss_val.item()
            if epoch>0:
                if correct_val/(test_data_set.__len__()) > np.max(correct_rate):
                    checkpoint = {
                        "net": net.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(net.state_dict(), str(classes)+'_'+'best_model.pth')
                    torch.save(checkpoint, 'ckpt_best.pth')
            if(epoch+1)%10==0:
                print('epoch:{} run_loss_val:{} correct_val_rate:{}'.format(epoch+1, run_loss_val/test_data_set.__len__(), correct_val/test_data_set.__len__()))
            correct_rate.append(correct_val/test_data_set.__len__())
            run_loss_val = 0.0
            correct_val = 0

    acc.append(np.max(correct_rate))
    print(acc)
    np.save("test_auc",acc)
    # plt.plot(correct_rate_train, color='r')
    # plt.plot(correct_rate, color='b')
    # plt.show()
    print("MLP Done!!!!!!")


    # model_eval = Net()
    # if classes==25:
    #     labels_name = [-80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80]
    # elif classes==9:
    #     labels_name = [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    # else:
    #     labels_name = [-40,-20,0,20,40]
    # model_eval.load_state_dict(torch.load(str(classes)+'_'+'best_model.pth'))
    # model_eval.eval()

    # y_true = []
    # y_pred = []
    # for i, data_test in enumerate(test_data):
    #     input_data_test, input_label_test = data_test
    #     input_label_test = input_label_test.long()

    #     outputs_test = model_eval(input_data_test)

    #     y_true.append(input_label_test.tolist())
    #     y_pred.append(outputs_test.argmax(1).tolist())
    # y_pred = list(itertools.chain.from_iterable(y_pred))
    # y_true = list(itertools.chain.from_iterable(y_true))
    # print(y_pred)
    # print(y_true)

    # cm = confusion_matrix(y_true, y_pred)
    # plot_confusion_matrix(cm, labels_name, "Classifier Confusion Matrix")
    # plt.savefig(str(k)+'_fig.png')
    # plt.close()



# '''
# # ---------------------------------------------------
# MLP-Classifier -- liyuan 23/04/06
# # ---------------------------------------------------
# '''

# from markupsafe import string
# from sklearn.neural_network import MLPRegressor
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# from sklearn import metrics
# import sys
# import os
# import random

# path_RC_state = './RC_simulation/RC_state_archive/'
# #data = scio.loadmat('./traindata.mat')

# vector = np.load(path_RC_state + 'RC_state_vector_training.npy')

# labels = np.load('./dataset/' + 'label_training.npy')

# RC_state_vector_training =vector[0:900]
# label_training = labels[0:900]
# RC_state_vector_testing = vector[900:]
# label_testing = labels[900:]
# print(label_training)

# acc = []
# azimuth_list = np.array(azimuth_list)
# azimuth_list_test = np.array(azimuth_list_test)

# model_2 = MLPClassifier(hidden_layer_sizes=(800,1000,600), max_iter=500, early_stopping=False, solver="adam",random_state=True)
# model_2.fit(RC_state_vector_training, azimuth_list)
# # model_2 = MLPRegressor(hidden_layer_sizes=(500, 600, 300), max_iter=1000)
# # model_2.fit(RC_state_vector_training, elevation_list)


# print("MLP predict...")
# label_predict_training_2 = model_2.predict(RC_state_vector_training)
# train_score_2 = model_2.score(RC_state_vector_training, azimuth_list)

# label_predict_2 = model_2.predict(RC_state_vector_testing)
# test_score_2 = model_2.score(RC_state_vector_testing, azimuth_list_test)

# # print("训练集的预测输出：", label_predict_training_2)
# # print("训练集的真实标签：", azimuth_list)

# # print("测试集的预测输出：", label_predict_2)
# # print("测试集的真实标签: ", azimuth_list_test)

# print("训练集上的准确率: ", train_score_2)
# print("测试集上的准确率: ", test_score_2)

# acc.append(test_score_2)
# np.save("test_auc",acc)
# print("MLP Done!!!!!!")


