#!D:/Application/python/python.exe
import torch, os, time
from Cls import MLP
from Cfg import *

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
output_size=len(ohe)
# 实例化新感知机 或进行迁移学习
if os.path.isfile(model_path):
    print("模型文件存在，加载模型做迁移学习...")
    model = torch.load(model_path, map_location=device)
else:
    print("不存在已有模型，新实例化模型...")
    model = MLP(input_size=input_size,hidden_size1=hidden_size1,hidden_size2=hidden_size2,hidden_size3=hidden_size3,output_size=output_size)

tm = time.strftime("%Y-%m-%d-%H")
print(model)