#!D:/Application/python/python.exe
import torch, os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 获取训练张量
def get_tensor(ohe,data_dir=r'D:/desk/github/Cell_Classification/data'):
    tensor_lst = []
    label_lst = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    for t_d in txt_files:
        pass
    return tensor_lst,label_lst  # 返回特征值与每组特征值对应的独热编码




# 训练方法
def model_train(model,device,one_hot_encode,learning_rate=0.001,data_dir=r'D:/desk/github/Cell_Classification/data'):
    model.to(device)  # 有 GPU 的情况下 将模型转移至 GPU
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于多分类问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ohe = one_hot_encode
    tensor_lst = get_tensor(ohe=ohe, data_dir=r'D:/desk/github/Cell_Classification/data')
    print(model)



    return model

if __name__ == '__main__':
    get_tensor()
