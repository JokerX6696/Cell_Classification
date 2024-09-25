#!D:/Application/python/python.exe
import torch, os, random, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
# 获取训练张量
def get_tensor(ohe,features,data_dir=r'D:/desk/github/Cell_Classification/data'):
    tensor_lst = []
    label_lst = []
    col_idx = [i.replace('-','.') for i in features]
    txt_files = [os.path.abspath(os.path.join(data_dir, f)) for f in os.listdir(data_dir) if f.endswith('.txt')]
    for t_d in txt_files:
        df = pd.read_csv(t_d,sep='\t',index_col=0) # 行索引作为 label
        df = df[col_idx].dropna(axis=1, how='all') # 按照特征值排序
        for index, row in df.iterrows(): # 每个细胞循环一次
            tensor_lst.append(row.to_list()) 
            label_lst.append(index)

    label_lst = [ohe[i] for i in label_lst]

    return tensor_lst,label_lst  # 返回特征值与每组特征值对应的独热编码

# 随机选择训练集与验证集
def lst_split(lst,jm_per=0.7):
    length = len(lst)
    train_num = math.ceil(length * jm_per)  # 70% 建模 30 % 验证
    all_num = range(0,length)
    train_idx = random.sample(all_num, train_num)
    val_idx = [i for i in (set(all_num) - set(train_idx))]
    
    train_set = [lst[i] for i in train_idx]
    val_set = [lst[i] for i in val_idx]
    return train_set,val_set
    


# 训练方法
def model_train(model,features,device,one_hot_encode,learning_rate=0.001,data_dir=r'D:/desk/github/Cell_Classification/data',num_epochs=100):
    model.to(device)  # 有 GPU 的情况下 将模型转移至 GPU
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 对于多分类问题
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ohe = one_hot_encode
    tensor_lst,label_lst = get_tensor(ohe=ohe, features=features,data_dir=r'D:/desk/github/Cell_Classification/data')
    x_train,x_val = lst_split(tensor_lst)
    y_train,y_val = lst_split(label_lst)
    x_train_tensor = torch.tensor(x_train,dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train,dtype=torch.float32).to(device)
    x_val_tensor = torch.tensor(x_val,dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val,dtype=torch.float32).to(device)


    # 训练
    for epoch in range(num_epochs):
        
        model.train()  # 切换到训练模式

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(x_train_tensor)

        # 计算损失
        loss = criterion(outputs, y_train_tensor)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 验证步骤
        model.eval()  # 切换到评估模式
        with torch.no_grad():  # 不计算梯度
            val_outputs = model(x_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_accuracy = (val_outputs.argmax(dim=1) == y_val_tensor.argmax(dim=1)).float().mean().item()  # 计算验证准确率


        # 每 10 轮打印一次损失和验证结果
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

    # 训练完成后，你可以使用 model.eval() 切换到评估模式进行测试。
    return model

if __name__ == '__main__':
    pass
