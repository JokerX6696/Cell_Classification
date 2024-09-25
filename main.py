#!D:/Application/python/python.exe
import torch, os, time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Cls import MLP
from Cfg import *
from wk_fun import *
# 初始化工作目录 将流程常用 celltype_marker 转化为 label 文件与特征值文件！
wkdir_init(
    fpth=r'D:/desk/github/Cell_Classification/Cfg/celltype_marker.txt',
    ct_output=r'D:/desk/github/Cell_Classification/Cfg/celltype_list.txt',
    mk_output=r'D:/desk/github/Cell_Classification/Cfg/feature_genelist.txt'
    )
# 定义运算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# 分类 label
ohe = one_hot_encoding()
# 特征值 list

features = get_feature_list()
# MLP 参数
model_path = '' # 如果模型存在 做迁移学习 可以填写该参数！
input_size = len(features)
hidden_size1 = input_size*2
hidden_size2 = input_size*2
hidden_size3 = input_size*2
output_size = len(ohe)
# 实例化新感知机 或进行迁移学习
if os.path.isfile(model_path):
    print("模型文件存在，加载模型做迁移学习...")
    model = torch.load(model_path, map_location=device)
else:
    print("不存在已有模型，新实例化模型...")
    model = MLP(input_size=input_size,hidden_size1=hidden_size1,hidden_size2=hidden_size2,hidden_size3=hidden_size3,output_size=output_size)

# 模型训练
tm_start = time.strftime("%Y-%m-%d-%H")

new_model = model_train(
    model=model,
    features=features,
    device=device,
    one_hot_encode=ohe,
    data_dir=r'D:/desk/github/Cell_Classification/data',
    learning_rate=0.0001,
    num_epochs=100
    )


tm_end = time.strftime("%Y-%m-%d-%H")

