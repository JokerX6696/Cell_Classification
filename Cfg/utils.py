# 常用功能在此封装
def read_listfile(fpth):
    with open(fpth)as f:
        xxx_list = f.readlines()

    xxx_list = [i.replace("\n",'') for i in xxx_list] # 删除换行符
    xxx_list = [x for x in xxx_list if x is not None and x != ""] # 删除空值
    xxx_list = list(set(xxx_list)) # 删除重复值
    xxx_list.sort() # 保证顺序不变
    return(xxx_list)
if __name__ == '__main__':
    a = read_listfile(r'D:/desk/github/Cell_Classification/Cfg/celltype_list.txt')
    print(a)