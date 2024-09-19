#!D:/Application/python/python.exe

def one_hot_encoding():
    from .utils import read_listfile
    celltypes = read_listfile('Cfg/celltype_list.txt')

    celltype_num = len(celltypes)
    ohe_dict = {}
    counts = 0
    while counts < celltype_num:
        celltype = celltypes[counts]
        raw_list = [0 for _ in range(celltype_num)]
        raw_list[counts] = 1
        ohe_dict[celltype] = raw_list
        counts += 1
    
    with open('Log/one_hot_encoding.log','w')as f:
        for k in ohe_dict:
            print(k,': \t',end="",file=f,sep='')
            for j in ohe_dict[k]:
                print(j,end=", ",file=f)
            print('',file=f)
    return(ohe_dict) # 返回独热编码的字典

if __name__ == '__main__':
    ohe_dict = one_hot_encoding()
    print(ohe_dict)