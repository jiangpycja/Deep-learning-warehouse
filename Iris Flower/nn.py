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


train_set, val_set, test_set = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size]) 
print(f"test set size: {train_size}; validation set size: {val_size}; test set size: {test_size}")


train_loader = DataLoader(train_set, batch_size = 50, shuffle = True) 
val_loader = DataLoader(val_set, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)


def infer(model, dataset, device):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for data in dataset:
            datas, label = data
            output = model(datas.to(device))
            
            predict = torch.max(output, dim=1)[1]
            
            acc_num += torch.eq(predict, label.to(device)).sum().item() 
        accuracy_rate = acc_num / len(dataset)
        return accuracy_rate
    
def main(lr=0.005, learn_time=20):
    model = NN(4, 16, 32, 3).to(device)
    loss = nn.CrossEntropyLoss()

    parameter = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameter, lr=lr)


    save_path = os.path.join(os.getcwd(), "result/weight")
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    

    for i in range(learn_time):
        model.train()
        acc_num = torch.zeros(1).to(device)
        sample_num = 0
        

        for datas in train_loader:
            data, label = datas
            
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


        acc_rate=0
        val_acc=0
    
    print("job down!!")

    test_acc = infer(model, test_loader, device)
    print("The accuracy of test set is: ", test_acc)

if __name__ == "__main__":
    main()
