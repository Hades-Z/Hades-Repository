# 环境初始化
import numpy as np
import pandas as pd

# 加载数据
from sklearn.datasets import load_boston
data = load_boston()
# data

# 数据探索
df = pd.DataFrame(data['data'])
df.columns = data['feature_names']
df.info();df.head()
print(data['DESCR'])

# Pre-processing
df_cat = df[['CHAS','RAD']].astype('int').astype('category')
df_num = df.drop(columns=['CHAS','RAD'])
# print(df_cat.shape); print(df_num.shape)

#
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()
nar_cat = onehot.fit_transform(df_cat).toarray()

#
from sklearn.preprocessing import StandardScaler
standardized = StandardScaler()
nar_num = standardized.fit_transform(df_num)

#
X = np.concatenate((nar_num, nar_cat), axis=1)
y = data['target'].reshape(len(data['target']),1)

#
def data_split(data, test_ratio=0.2, val_ratio=0):
    index = np.random.choice(range(len(data)), size=len(data), replace=False)
    train_index = index[:int(len(data)*(1-val_ratio-test_ratio))]
    val_index = index[int(len(data)*(1-val_ratio-test_ratio)):int(len(data)*(1-test_ratio))]
    test_index = index[int(len(data)*(1-test_ratio)):]
    return data[train_index], data[test_index], data[val_index]

#
X_train,X_test,_ = data_split(X,test_ratio=0.2)
y_train,y_test,_ = data_split(y,test_ratio=0.2)

#
def LinReg_train(X,y,num_epochs,lr):
    # 初始化
    loss=[]
    W = np.random.normal(0,1,(1,X.shape[1]))
    b = 0
    # 训练
    for i in range(num_epochs):
        y_hat = np.dot(X,W.T)+b
        # MSE
        ls = np.sum((y-y_hat)**2)/2
        # ls = np.dot((y_train-y_hat).T,(y_train-y_hat))/2
        loss.append(ls)
        # 优化（模型参数迭代）
        W = W-lr*(-np.dot((y-y_hat).T,X)/X.shape[0])
        b = b-lr*np.mean(y-y_hat)
    return loss, W, b

#
def LinReg_price(X,y,W,b):
    y_hat = np.dot(X,W.T)+b
    # MSE
    ls = np.sum((y-y_hat)**2)/2
    # ls = np.dot((y_train-y_hat).T,(y_train-y_hat))/2
    return y_hat, ls

#
loss_CV,W,b = LinReg_train(X_train,y_train,100,0.03)
y_hat,loss = LinReg_price(X_test,y_test,W,b)
