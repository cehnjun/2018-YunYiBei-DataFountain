import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../data/train_first.csv')

cw = lambda x: list(jieba.cut(x)) #定义分词函数
train['words'] = train['Discuss'].apply(cw)
# print(train['Score'])
y = np_utils.to_categorical(train['Score'])
# print(y)

predict = pd.read_csv('../data/predict_first.csv')

predict['words'] = predict['Discuss'].apply(cw)

# print(train.head())
# print(predict.head())

# 统计词频
d2v_train = pd.concat([train['words'], predict['words']], ignore_index = True)
#将所有词语整合在一起
w = []
for i in d2v_train:
  w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train

dict['id']=list(range(1, len(dict)+1))
# dict = dict[(dict[0]>=10) & (dict[0] < 72410)]
# print(dict)

get_sent = lambda x: list(dict['id'][x])
train['sent'] = train['words'].apply(get_sent) #速度太慢
predict['sent'] = predict['words'].apply(get_sent) #速度太慢

maxlen = 10

print("Pad sequences (samples x time)")
train['sent'] = list(sequence.pad_sequences(train['sent'], maxlen=maxlen))
predict['sent'] = list(sequence.pad_sequences(predict['sent'], maxlen=maxlen))


x = np.array(list(train['sent']))[::2] #训练集
y_ = np.array(y)[::2]

xt = np.array(list(train['sent']))[1::2] #测试集
yt = np.array(y)[1::2]

x_sub = np.array(list(predict['sent'])) #提交集

print('Build model...')
model = Sequential()
model.add(Embedding(len(dict) + 1, 256))
model.add(LSTM(256)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x, y_, batch_size=32, nb_epoch=2,validation_data=(xt,yt)) #训练时间为若干个小时

y_test = [np.argmax(i) for i in list(yt)]
v = pd.DataFrame()
v['result'] = list(y_test)
v['true'] = list(model.predict_classes(xt,batch_size=32))
print(v)

re = pd.DataFrame()
sub = model.predict_classes(x_sub,batch_size=32)
re['id'] = list(predict['Id'].values)
re['score'] = list(sub)
re.to_csv('../tmp/lstm_epoh_2.csv',index=False,header=False)

# 54 46
