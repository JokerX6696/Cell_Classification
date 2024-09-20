# 常用功能在此封装
def read_listfile(fpth):
    with open(fpth)as f:
        xxx_list = f.readlines()

    xxx_list = [i.replace("\n",'') for i in xxx_list] # 删除换行符
    xxx_list = [x for x in xxx_list if x is not None and x != ""] # 删除空值
    xxx_list = list(set(xxx_list)) # 删除重复值
    xxx_list.sort() # 保证顺序不变
    return(xxx_list)

# 基于 celltype_marker.txt 生成 celltype_list.txt feature_genelist.txt 两个文件！
def wkdir_init(fpth,ct_output,mk_output):
    import re
    with open(fpth,'r')as f:
        info = f.readlines()
    
    counts = 0
    marker = []
    for line in info:
        line = line.strip()
        line_lst = line.split('\t')
        if counts == 0:
            celltype = line_lst
            counts += 1
        else:
            for gene in line_lst:
                if bool(re.search('[a-zA-Z]', gene)):
                    marker.append(gene)
    with open(ct_output,'w')as f:
        for i in celltype:
            f.write('%s\n' %i)
    with open(mk_output,'w')as f:
        for i in marker:
            upper_i = str.upper(i)
            f.write('%s\n' %upper_i)        
    return 0



if __name__ == '__main__':
    con = wkdir_init(fpth=r'D:/desk/github/Cell_Classification/Cfg/celltype_marker.txt',ct_output=r'D:/desk/github/Cell_Classification/Cfg/celltype_list.txt',mk_output=r'D:/desk/github/Cell_Classification/Cfg/feature_genelist.txt')
