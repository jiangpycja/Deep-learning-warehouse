from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch

class iris_dataloader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

        assert os.path.exists(data_path), "data_set not exist"

        data = pd.read_csv(self.data_path, names = [0,1,2,3,4])
        dic = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2} 
        data[4] = data[4].map(dic) 


        dt = data.iloc[:, :4]
        label = data.iloc[:, 4:]

        dt = (dt-np.mean(dt)) / np.std(dt)


        self.dt = torch.from_numpy(np.array(dt, dtype='float32'))
        self.label = torch.from_numpy(np.array(label, dtype='int64'))

        self.data_number = len(label)
        print("current size of dataset: ", self.data_number)


    def __len__(self):
        return self.data_number


    def __getitem__(self, index):
        self.dt = list(self.dt)
        self.label = list(self.label)
        return self.dt[index], self.label[index]
