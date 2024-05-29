import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 读取数据集
train_all = pd.read_csv('churn-bigml-80.csv')
test_all = pd.read_csv('churn-bigml-20.csv')

# 预处理，首先先将其类型转化为矩阵
train_all = np.array(train_all)
test_all = np.array(test_all)

# 将标签值转化为0/1
train_all[:, 19] = LabelEncoder().fit_transform(train_all[:, 19])
test_all[:, 19] = LabelEncoder().fit_transform(test_all[:, 19])

# 将属性值中的第三列和第四列的YES和NO也转化为1和0
train_all[:, 3] = LabelEncoder().fit_transform(train_all[:, 3])
test_all[:, 3] = LabelEncoder().fit_transform(test_all[:, 3])
train_all[:, 4] = LabelEncoder().fit_transform(train_all[:, 4])
test_all[:, 4] = LabelEncoder().fit_transform(test_all[:, 4])

# 将国家名称转化为数字标签，便于网络计算
train_all[:, 0] = LabelEncoder().fit_transform(train_all[:, 0])
test_all[:, 0] = LabelEncoder().fit_transform(test_all[:, 0])

# 对数据切片，将属性和标签分割
feature_train = train_all[:, 0: -1].astype(np.float32)
feature_test = test_all[:, 0: -1].astype(np.float32)
label_train = train_all[:, -1].astype(np.float32)
label_test = test_all[:, -1].astype(np.float32)

# 数据归一化
scaler = StandardScaler()
feature_train = scaler.fit_transform(feature_train)
feature_test = scaler.fit_transform(feature_test)

# 将ndarray数据类型转化为tensor类型,以适用于神经网络的传播过程
feature_train = torch.from_numpy(feature_train)
label_train = torch.from_numpy(label_train)
label_train = label_train.view(-1, 1)
feature_test = torch.from_numpy(feature_test)
label_test = torch.from_numpy(label_test)


# 定义BP神经网络类，继承自nn.Module
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 调用父类的构造函数
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tan = nn.Tanh()
        self.fc0 = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tan(self.fc0(x))     # 进行前向传播
        x = self.relu(self.fc1(x))
        x = self.tan(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tan(self.fc4(x))
        x = self.tan(self.fc5(x))
        x = self.sigmoid(x)
        return x


net = Net(19, 60, 1)
accuracy_list = []
for n in range(10):
    # 定义优化器，加入正则项，定义超参数
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
    # 定义损失函数，因为是二分类问题，所以采用二分类交叉熵损失函数
    criterion = nn.BCELoss()

    # 训练
    epoch = 1000   # 训练次数epoch
    for i in range(epoch):
        optimizer.zero_grad()   # 把梯度清零
        output = net(feature_train)   # 进行一次前向传播
        loss = criterion(output, label_train)
        loss.backward()     # 后向传播
        optimizer.step()    # 进行一次参数更新

    # 测试
    test_outputs = net(feature_test)
    test_predictions = (test_outputs > 0.5).float().view(-1)
    accuracy = accuracy_score(label_test, test_predictions)
    precision = precision_score(label_test, test_predictions)
    recall = recall_score(label_test, test_predictions)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall:{recall:.4f}")
    accuracy_list.append(accuracy)

# 画出每次训练的模型其对应的准确率曲线，观察平均情况
x_ = [i for i in range(len(accuracy_list))]
plt.plot(x_, accuracy_list)
plt.show()
