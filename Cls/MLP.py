#!D:/Application/python/python.exe
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        # 定义第一个隐藏层
        self.fc1 = nn.Linear(input_size, hidden_size1)
        # 定义第二个隐藏层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        # 定义第三个隐藏层
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        # 定义输出层
        self.out = nn.Linear(hidden_size3, output_size)

    def forward(self, x):
        # 通过第一个隐藏层，并使用ReLU激活函数
        x = F.relu(self.fc1(x))
        # 通过第二个隐藏层，并使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过第三个隐藏层，并使用ReLU激活函数
        x = F.relu(self.fc2(x))
        # 通过输出层
        x = self.out(x)
        return x
    
if __name__ == '__main__':
    cls = MLP(
        input_size=54,
        hidden_size1=54,
        hidden_size2=108,
        hidden_size3=108,
        output_size=9
        )
    print(cls)