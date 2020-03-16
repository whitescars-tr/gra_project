import pandas as pd
import numpy as np
import os
from keras import Sequential
from keras.layers import LSTM,Dense, Activation,Dropout
from keras import optimizers
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def read_data():
    x_train=pd.read_csv('x_train_z_score.csv',header=None)#训练集数据
    x_test=pd.read_csv('x_test_z_score.csv',header=None)#测试集数据
    y_train=pd.read_csv('y_train_OHC.csv',header=None)#训练集标签
    y_test=pd.read_csv('y_test_OHC.csv',header=None)#测试集标签
    return x_train.values,x_test.values,y_train.values,y_test.values

def normalization_train(x_input,name):
    x_input = x_input.reshape(-1,25)
    x_input = x_input.astype('float32')
    _peremeter=[]
    _peremeter = np.array(_peremeter)
    for k in range(25):
        _range=np.max(x_input[:,k])-np.min(x_input[:,k])
        _peremeter=np.append(_peremeter,_range)
        _min=np.min(x_input[:,k])
        _peremeter=np.append(_peremeter,_min)
        for j in range(x_input.shape[0]):
            x_input[j,k]=(x_input[j,k]-_min)/_range
    x_input = pd.DataFrame(x_input)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    x_input.to_csv(name,index=False,header=False)
    _peremeter=_peremeter.reshape(-1,2)
    _peremeter=pd.DataFrame(_peremeter)
    _peremeter.to_csv('peremeter_normalization.csv',index=False,header=False)

def normalization_test(x_input,name):
    _peremeter=pd.read_csv('peremeter_normalization.csv',header=None)
    _peremeter=_peremeter.values.reshape(-1,2)
    for k in range(25):
        _range=_peremeter[k,0]
        _min=_peremeter[k,1]
        for j in range(x_input.shape[0]):
            x_input[j,k]=(x_input[j,k]-_min)/_range
    x_input = pd.DataFrame(x_input)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    x_input.to_csv(name,index=False,header=False)

def z_score_for_train(input_array,name):
    _peremeter=[]#参数储存
    _peremeter = np.array(_peremeter) 
    for k in range(input_array.shape[1]):
        _mean=np.mean(input_array[:,k])#平均数
        _peremeter=np.append(_peremeter,_mean)
        _std=np.std(input_array[:,k])#标准差
        _peremeter=np.append(_peremeter,_std)
        for i in range(input_array.shape[0]):
            input_array[i,k]=(input_array[i,k]-_mean)/_std
    input_array = pd.DataFrame(input_array)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    input_array.to_csv(name,index=False,header=False)
    _peremeter=_peremeter.reshape(-1,2)
    _peremeter=pd.DataFrame(_peremeter)
    _peremeter.to_csv('peremeter_z_score.csv',index=False,header=False)

def z_score_for_test(input_array,name):
    _peremeter=pd.read_csv('peremeter_z_score.csv',header=None)
    _peremeter=_peremeter.values.reshape(-1,2)
    for k in range(input_array.shape[1]):
        _mean=_peremeter[k,0]
        _std=_peremeter[k,1]
        for i in range(input_array.shape[0]):
            input_array[i,k]=(input_array[i,k]-_mean)/_std
    input_array = pd.DataFrame(input_array)
    #index参数设置为False表示不保存行索引,header设置为False表示不保存列索引
    input_array.to_csv(name,index=False,header=False)


def train_01(n_step,n_input,n_classes,x_train,x_test,y_train,y_test):
    model = Sequential()

#这个网络中，我们采用LSTM+Dense 层+激活层，优化函数采用Adam，
#损失函数采用交叉熵，评估采用正确率。

#学习率
    learning_rate = 0.001
#每次处理的数量
    batch_size = 25
#循环次数
    epochs = 20
#神经元的数量
    n_lstm_out = 64

#分割验证集测试集
    x_train,x_val, y_train, y_val =train_test_split(x_train,y_train)
    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    '''
#LSTM层
    model.add(LSTM(
        units = n_lstm_out,
        input_shape = (n_step, n_input)))
    #全连接层          
    model.add(Dense(units = n_classes))
#激活层
    model.add(Activation('softmax'))

#查看各层的基本信息
    model.summary()

# 编译
    model.compile(
        optimizer = optimizers.Adam(lr = learning_rate),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

#训练
    history=model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_val,y_val))
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left') 
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
#评估
    score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
    print('loss:',score[0])
    print('acc:',score[1])
    '''
def train_02(n_step,n_input,n_classes,x_train,x_test,y_train,y_test):
    model = Sequential()

#这个网络中，我们采用LSTM+Dense 层+激活层，优化函数采用Adam，
#损失函数采用交叉熵，评估采用正确率。

#学习率
    learning_rate = 0.001
#每次处理的数量
    batch_size = 25
#循环次数
    epochs = 30
#神经元的数量
    n_lstm_out = 32

#分割验证集测试集
    x_train,x_val, y_train, y_val =train_test_split(x_train,y_train,test_size=0.25)

#LSTM层
    model.add(LSTM(units= n_lstm_out,
                   input_shape=(n_step, n_input),
                   activation='relu',
                   return_sequences=True))
    for i in range(lstm_layers - 1):
        model.add(LSTM(output_dim=32 * (i+1),
                       activation='relu',
                       return_sequences=True))
    model.add(LSTM(units=n_lstm_out*2,
                   activation='relu',
                   return_sequences=False))
    model.add(Dense(units = n_classes))
#激活层
    model.add(Activation('softmax'))
#查看各层的基本信息
    model.summary()
# 编译
    model.compile(
        optimizer = optimizers.Adam(lr = learning_rate),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
#训练
    history=model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_val,y_val))

    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left') 
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
#评估
    score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
    print('loss:',score[0])
    print('acc:',score[1])
    
def main():
    x_train,x_test,y_train,y_test=read_data()
    n_step=5
    n_input=25
    n_classes=16
    #z_score_for_train(x_train,'x_train_z_score.csv')
    #z_score_for_test(x_test,'x_test_z_score.csv')
    #normalization_train(x_train,'x_train_nor.csv')
    #normalization_test(x_test,'x_test_nor.csv')
    x_train=x_train.reshape(-1,5,25)
    x_test=x_test.reshape(-1,5,25)
    train_02(n_step,n_input,n_classes,x_train,x_test,y_train,y_test)



if __name__ == '__main__':
    main()

