import os
import sys
from torch.utils.data import DataLoader

import torch
import tqdm as tq
import torch.nn as nn
import torch.optim as optim

from data_loader import iris_dataloader

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


custom_dataset = iris_dataloader("./Pytorch/Iris_data.txt")
train_size = int(len(custom_dataset) * 0.7)
val_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset)-train_size-val_size

# 根据刚才定义的比例把整个集合分成训练集，验证集，测试集
train_set, val_set, test_set = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size]) 
print(f"test set size: {train_size}; validation set size: {val_size}; test set size: {test_size}")

# 从训练集里头每次抽取一些数据喂给神经网络模型，一组抽取batch_size个，根据数据集大小而定
train_loader = DataLoader(train_set, batch_size = 50, shuffle = True) #shuffle = true证明每次抽取完之后把剩下的东西打乱
val_loader = DataLoader(val_set, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

# 定义推理函数返回准确率
def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            output = model(datas.to(device))
            # 我们把数据喂给模型之后，模型会通过每一行的数据给不同的label进行打分，比如这个花朵用例，因为有三个不同的花朵，所以每一行的数据都会针对不同的花朵给出各自的分数
            # 紧接着我们用max来找到每一行打分最高的那一个，那一个也就相当于时这一行的数据最有可能是属于这一朵花的类别
            # eg:[1.2, 0.3, 2.1]，那么我们会选取2.1，同时标明这是第三朵花，label=2
            # 所以从max我们可能会得到：[[2.1, 1.7, 2.2, 2.5], [2, 1, 0, 1]]
            # 所以我们最后index[1]就是为了找到我们的label这个list，而不是得分的list
            predict = torch.max(output, dim=1)[1]
            # 下面我们可以对比predict和实际label的数值，把符合的加到acc_num头上
            acc_num += torch.eq(predict, label.to(device)).sum().item() #.sum()把boolean值作为数值加起来，但是因为我们用了torch，所以要用.item()把tensor数字转化为正常数字
        accuracy_rate = acc_num / len(dataset)
        return accuracy_rate
    
def main(lr=0.005, learn_time=20):
    model = NN(4, 16, 32, 3).to(device) # hidden layer可以随便调，input和output是固定好的
    loss = nn.CrossEntropyLoss()

    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameter, lr=lr)

    # 权重文件存储路径
    save_path = os.path.join(os.getcwd(), "result/weight")
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    
    # 开始训练
    for i in range(learn_time):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        

        for datas in train_loader:
            data, label = datas
            # 这里的squeeze(-1)指的是把label[-1]位置的维度如果是1的话就缩减掉
            # eg: label.shape = [batch_size, 1], ie label = [[1], [2], [3]], then label.shape = [3,1]，这时候我用squeeze把维度是1的这部分缩掉，也就意味着[[1],[2],[3]]中的一维数组没了，变成了[1,2,3]
            # 这样的话更有利于我们后面直接进行对比，比如我们predict的list可以直接和label进行对比了，因为格式现在一样了
            label = label.squeeze(-1)
            sample_num = data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            predict_lst = torch.max(outputs, dim=1)[1]
            acc_num = torch.eq(predict_lst, label.to(device)).sum().item()

            loss_f = loss(outputs, label.to(device))
            div = loss_f.backward()
            if div == 0:
                break
            optimizer.step()

            acc_rate = acc_num / sample_num
            print("The accuracy of model now is: ", acc_rate)
        
        val_acc = infer(model, val_loader, device)
        print("The accuracy of validation set is: ", val_acc)
        torch.save(model.state_dict(), os.path.join(save_path, "NN.pth"))

        #清零初始化数据
        acc_rate=0
        val_acc=0
    
    print("job down!!")

    test_acc = infer(model, test_loader, device)
    print("The accuracy of test set is: ", test_acc)

if __name__ == "__main__":
    main()
