#coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import scipy
from sklearn.model_selection import KFold
# from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix, hstack


def get_data():
    train = pd.read_csv('../data/train_first.csv')
    test = pd.read_csv('../data/predict_first.csv')
    data = pd.concat([train, test])
    print('train %s test %s'%(train.shape,test.shape))
    print('train columns',train.columns)
    return data,train.shape[0],train['Score'],test['Id']

# 分词处理
def split_discuss(data):
    data['length'] = data['Discuss'].apply(lambda x:len(x))
    data['Discuss'] = data['Discuss'].apply(lambda x:' '.join(jieba.cut(x)))
    return data

# 预处理
def pre_process():
    data,nrw_train,y,test_id = get_data()
    data = split_discuss(data)
    cv = CountVectorizer(ngram_range=(1,2))
    discuss = cv.fit_transform(data['Discuss'])
    tf = TfidfVectorizer(max_df=10000,ngram_range=(1,2))
    discuss_tf = tf.fit_transform(data['Discuss'])
    # length = csr_matrix(pd.get_dummies(data['length'],sparse=True).values)
    data = hstack((discuss,discuss_tf)).tocsr()
    return data[:nrw_train],data[nrw_train:],y,test_id

def xx_mse_s(y_true,y_pre):
    y_true = y_true
    y_pre = pd.DataFrame({'res':list(y_pre)})

    y_pre['res'] = y_pre['res'].astype(int)
    return 1 / ( 1 + mean_squared_error(y_true,y_pre['res'].values)**0.5)


X,test,y,test_id = pre_process()
kf = KFold(n_splits=3,shuffle=True,random_state=42)
cv_pred = []
kf = kf.split(X)
xx_mse = []
model_1 = Ridge(solver='auto', fit_intercept=True, alpha=0.4, max_iter=250, normalize=False, tol=0.01)
for i ,(train_fold,test_fold) in enumerate(kf):
    X_train, X_validate, label_train, label_validate = X[train_fold, :], X[test_fold, :], y[train_fold], y[test_fold]
    model_1.fit(X_train, label_train)

    val_ = model_1.predict(X=X_validate)
    print(xx_mse_s(label_validate, val_))

    cv_pred.append(model_1.predict(test))
    xx_mse.append(xx_mse_s(label_validate, val_))

import numpy as np
print('xx_result',np.mean(xx_mse))

s = 0
for i in cv_pred:
    s = s + i

s = s/3
res = pd.DataFrame()
res['Id'] = list(test_id)
res['pre'] = list(s)

res.to_csv('../result/t_20180215_1.csv',index=False,header=False)



# 0.581334990703  0.528227538116  0.48691