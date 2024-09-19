#!D:/Application/python/python.exe

def get_feature_list():
    from .utils import read_listfile
    featrue_list = read_listfile('Cfg/feature_genelist.txt')
    with open('Log/feature_genelist.log','w')as f:
        for k in featrue_list:
            print(k,file=f)
            
    return(featrue_list)

if __name__ == '__main__':
    featrue_list = get_feature_list()
    print(len(featrue_list))