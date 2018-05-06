import pandas as pd
import numpy as np
import jieba
import re

trainFirst = pd.read_csv('E:/AIcompetition/DataFountainScore/data/train_first.csv')
testFirst = pd.read_csv('E:/AIcompetition/DataFountainScore/data/predict_first.csv')

cw = lambda x: jieba.lcut(x, cut_all=False)
zw = lambda x: jieba.cut(x, cut_all=False)
r = '[a-zA-Z0-9’!"#$%&\'()（）「」ಡω*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~～]+'  # 特殊字符过滤正则
m = '[\u4e00-\u9fa5]+'  # 中文匹配正则
rw = lambda x: re.sub(r, '', x)  # 特殊字符去除正则
# mw = lambda x: re.match(m, x).group()
# mw = lambda x: None if re.findall(m, x) is None else re.findall(m, x)

mw = lambda x: [''.join(re.findall(m, x))][0]  # 去除标点等特殊非中文字符，并合并单元格里的列表元素
# 训练集和测试集中的评论去除除中文外的字符
trainFirst['words'] = trainFirst['Discuss'].apply(mw)
testFirst['words'] = testFirst['Discuss'].apply(mw)

trainFirst['Discuss_split'] = trainFirst['Discuss'].apply(cw)  # 未去除非中文字符时的分词
testFirst['Discuss_split'] = testFirst['Discuss'].apply(cw)

trainFirst['words_split'] = trainFirst['words'].apply(cw)  # 去除非中文字符后的分词
testFirst['words_split'] = testFirst['words'].apply(cw)

# 统计词频
word2vec_train = pd.concat([trainFirst['words_split'], testFirst['words_split']], ignore_index=True)
# 将所有词语整合在一起
word2train = []
for word in word2vec_train:
    word2train.extend(word)


# 停用词加载
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


stopwords = stopwordslist('E:/AIcompetition/DataFountainScore/data/stopwords/stopwords.txt')
Word_train = []
for word in word2train:
    if word not in stopwords:
        Word_train.extend(word)


dict = pd.DataFrame(pd.Series(word2train).value_counts())  # 统计词频，找停用词


# 按空格分隔符分隔词写入txt文件
sep = ' '
with open('E:/AIcompetition/DataFountainScore/data/word2train.txt', 'w', encoding='utf-8') as file:
    file.write(sep.join(str(word) for word in word2train))

# 按行写入每句分词后的结果
sep = ' '
with open('E:/AIcompetition/DataFountainScore/data/word2train1.txt', 'w', encoding='utf-8') as file:
    for sentence in word2vec_train:
        file.write(sep.join(str(word) for word in sentence) + '\n')

# 利用gensim训练词向量

from gensim.models import word2vec

sentences = word2vec.Text8Corpus('E:/AIcompetition/DataFountainScore/data/word2train.txt')
model = word2vec.Word2Vec(
    sg=1, sentences=sentences,
    size=256, window=5, min_count=3, workers=4, iter=40)

model.wv.save_word2vec_format(out_path, binary=True)
model = word2vec.Word2Vec(sentences, size=50)
model.save_word2vec_format('E:/AIcompetition/DataFountainScore/data/word2vec.bin', binary=True)

sentences = word2vec.LineSentence('E:/AIcompetition/DataFountainScore/data/word2train1.txt')
model = word2vec.Word2Vec(
    sg=1, sentences=sentences,
    size=256, window=5, min_count=5, iter=40)
model.save('E:/AIcompetition/DataFountainScore/data/word2vec_model')

model.save_word2vec_format('E:/AIcompetition/DataFountainScore/data/word2vec_model.bin', binary=True)


# 将不存在于model词典中的词去除
cleanWords = lambda x: [m for m in x if m in model.vocab]
trainFirst['WSP_clean'] = trainFirst['WSP'].apply(cleanWords)
testFirst['WSP_clean'] = testFirst['WSP'].apply(cleanWords)


# 建立评论的词向量，对每个词的词向量求平均
def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    if len(text) != 0:
        for words in text:
            vec += model[words].reshape((1, size))
        vec /= len(text)
    return vec


# 建立评论向量
train_vectors = np.concatenate([buildWordVector(z, 256) for z in trainFirst['WSP']])
test_vectors = np.concatenate([buildWordVector(z, 256) for z in testFirst['WSP']])


train_vectors = np.load('E:/AIcompetition/DataFountainScore/data/train_vectors.npy')
test_vectors = np.load('E:/AIcompetition/DataFountainScore/data/test_vectors.npy')

TrainData = pd.DataFrame(train_vectors)
PreData = pd.DataFrame(test_vectors)
TrainData['Score'] = trainFirst['Score']

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

params = {
    'learning_rate': 0.04,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    #'sub_feature': 0.7,
    'num_leaves': 40,
    'colsample_bytree': 0.7,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
}

test_preds = np.zeros((PreData.shape[0], 5))
cv_pre = np.zeros((20000, 5))
print('开始CV 5折训练...')
kf = KFold(n_splits=5, shuffle=True, random_state=2018)
for i, (train_index, test_index) in enumerate(kf.split(TrainData)):
    print('第{}次训练...'.format(i))
    train_feat1 = TrainData.iloc[train_index]
    train_feat2 = TrainData.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1.drop('Score', axis=1), train_feat1['Score'])
    lgb_train2 = lgb.Dataset(train_feat2.drop('Score', axis=1), train_feat2['Score'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    #feval=evalerror,
                    early_stopping_rounds=50)
    y_pred=gbm.predict(train_feat2.drop('Score', axis=1))
    cv_pre[:, i] = y_pred
    print('Score: %.4f' % (1/(1+np.sqrt(mean_squared_error(train_feat2['Score'], y_pred.round())))))
    test_preds[:, i] = gbm.predict(PreData)

submission = pd.DataFrame({'Score': test_preds.mean(axis=1)})
submission = submission.round()
testFirst['Score'] = submission['Score']
testFirst.drop('Discuss', axis=1, inplace=True)
testFirst['Score'] = testFirst['Score']
testFirst.to_csv('E:/AIcompetition/DataFountainScore/data/Lgb01.csv', header=None, index=False)

submission.loc[submission[submission['Score']<1.7].index, 'Score'] = 1
submission.loc[submission[submission['Score']>4.7].index, 'Score'] = 5
submission[submission['Score'] < 2.7][submission['Score']>1.7]['Score'] = 2
