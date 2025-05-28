from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

class iris_dataloader(Dataset):
    #要继承Dataset,我们必须要重新定义len和getitem,这两个我们自己定义好了之后才可以在后面建模的过程中直接调用
    def __init__(self, data_path):
        self.data_path = data_path

        assert os.path.exists(data_path), "data_set not exist"

        data = pd.read_csv(self.data_path, names = [0,1,2,3,4])
        dic = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}  #可以试试用pd的loop去改，这样的话即使有很多数据集也可以高效的收集数据
        data[4] = data[4].map(dic) #把最后一列的名字替换为数字

        #分别读取数据dt, 和最后要识别的label
        dt = data.iloc[:, :4]
        label = data.iloc[:, 4:]

        dt = (dt-np.mean(dt) / np.std(dt)) #standarization

        #把数据转化到torch可以识别的tensor格式
        self.dt = torch.from_numpy(np.array(dt, dtype='float32'))
        self.label = torch.from_numpy(np.array(label, dtype='int64'))

        self.data_number = len(label)
        print("current size of dataset: ", self.data_number)

    # 获取数据集长度
    def __len__(self):
        return self.data_number

    # 读取并返回一个数据样本
    def __getitem__(self, index):
        self.dt = list(self.dt)
        self.label = list(self.label)
        return self.dt[index], self.label[index]
