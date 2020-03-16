import pandas as pd
import numpy as np
import os
import random
'''
from keras import Sequential
from keras.layers import LSTM,Dense, Activation
from keras import optimizers
'''
def read_data(folder_name='C:\Jason\gra_proj\dataset\out_3_5test'):
    x_per=[]
    x_lable=[]
    y_input=[]
    #读取数据
    for dirpath, dirnames, filenames in os.walk(r'C:\Jason\gra_proj\dataset\out_3_5test'):
        y_input=filenames
    for info in os.listdir('C:\Jason\gra_proj\dataset\out_3_5test'):
        domain = os.path.abspath(r'C:\Jason\gra_proj\dataset\out_3_5test')
        info = os.path.join(domain,info)
        #储存标签
        data = pd.read_csv(info)
        data=np.array(data)
        #删除第一行时间戳
        temp=np.delete(data,0,axis=1)
        #合并为一个矩阵，维数变为一维
        x_per=np.append(x_per,temp)
    #调整矩阵维数
    x_input=x_per.reshape(-1,5,25)
    #数据分割
    temp_1=[]#训练集数据
    temp_2=[]#测试集数据
    temp_3=[]#训练集标签
    temp_4=[]#测试集标签
    num=1
    a=random.randint(1,5)
    for k in range(x_input.shape[0]):
        if num==a:
            temp_2=np.append(temp_2,x_input[k,:,:])
            temp_4=np.append(temp_4,y_input[k])
        else:
            temp_1=np.append(temp_1,x_input[k,:,:])
            temp_3=np.append(temp_3,y_input[k])
        
        if num==5:
            num=0
            a=random.randint(1,5)
        num+=1

    temp_1=temp_1.reshape(-1,25)
    temp_2=temp_2.reshape(-1,25)
    
    temp_1 = pd.DataFrame(temp_1)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    temp_1.to_csv("x_train.csv",index=False,header=False)
    
    temp_2 = pd.DataFrame(temp_2)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    temp_2.to_csv("x_test.csv",index=False,header=False)

    temp_3 = pd.DataFrame(temp_3)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    temp_3.to_csv("y_train.csv",index=False,header=False)

    temp_4 = pd.DataFrame(temp_4)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    temp_4.to_csv("y_test.csv",index=False,header=False)
    '''
#归一化
def nomalization(x_input):
    for i in range(input.shape[2]):
        _range = np.max(x_input[:,:,i]) - np.min(x_output[:,:,i])
        minimum=np.min(x_output[:,:,i])
        for j in x_output[:,:,i]:
            j=(j-minimum)/_range
    return x_input
'''
def y_change(file_root,file_name):
    temp=[]
    temp_02=[]
    y_input=pd.read_csv(file_root,header=None)
    print(y_input.shape)
    y_input = np.array(y_input)
    for elements in y_input[:,0]:
        temp=np.append(temp,elements[0:3])
    print(temp.shape)
    temp_02= pd.get_dummies(temp)
    temp_02= pd.DataFrame(temp_02)
    temp_02.to_csv(file_name,index=False,header=False)
def main():
    y_change('C:/Users/Zhangzzs/source/repos/gra_project/gra_project/y_train.csv','y_train_OHC.csv')
    y_change('C:/Users/Zhangzzs/source/repos/gra_project/gra_project/y_test.csv','y_test_OHC.csv')

if __name__ == '__main__':
    main()
    # print(__name__)

