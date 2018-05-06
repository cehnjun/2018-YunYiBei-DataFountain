import pandas as pd
import numpy as np
import jieba
import re
from gensim.models import word2vec
from gensim.models import keyedvectors


def get_stopwords_dict(file_path='E:/AIcompetition/DataFountainScore/data/stopwords.txt'):
    stopwords_dict = {}
    with open(file_path, 'r', encoding='utf-8') as stopwords:
        for line in stopwords.readlines():
            stopwords_dict[line.strip()] = line.strip()
    return stopwords_dict


def split_words(input_data):
    chinese_characters = '[\u4e00-\u9fa5]+'  # 中文匹配正则
    # 去除标点等特殊非中文字符，并合并单元格里的列表元素,训练集和测试集中的评论去除除中文外的字符
    input_data['Discuss_NoSym'] = input_data['Discuss'].apply(lambda x: [''.join(re.findall(chinese_characters, x))][0])
    # 去除非中文字符后的分词
    input_data['Discuss_NoSym_split'] = input_data['Discuss_NoSym'].apply(lambda x: jieba.lcut(x, cut_all=False))
    # 未去除非中文字符时的分词
    input_data['Discuss_split'] = input_data['Discuss'].apply(lambda x: jieba.lcut(x, cut_all=False))
    return input_data


def create_w2c_file(splited_word, stop_words, output_filename):
    file_path = 'E:/AIcompetition/DataFountainScore/data/'
    with open(file_path+output_filename, 'w', encoding='utf-8') as file:
        for sentence in splited_word:
            file.write(' '.join(str(word) for word in sentence if word not in stop_words) + '\n')


def train_word2vec(input_data='discuss_w2v.txt', model_name='model.bin'):
    file_path = 'E:/AIcompetition/DataFountainScore/data/'
    sentences = word2vec.LineSentence(file_path+input_data)
    model = word2vec.Word2Vec(
        sg=1, sentences=sentences,
        size=256, window=5, min_count=5, workers=4, iter=40)
    model.wv.save_word2vec_format(file_path+model_name, binary=True)


def load_w2c_model(model_name='model.bin'):
    file_path = 'E:/AIcompetition/DataFountainScore/data/'
    model = keyedvectors.KeyedVectors.load_word2vec_format(file_path+model_name, binary=True)
    return model


def build_sentence_vector(model, text, size=256):
    vec = np.zeros(size).reshape((1, size))
    if len(text) != 0:
        for words in text:
            vec += model[words].reshape((1, size))
        vec /= len(text)
    return vec


def create_discuss_feature(input_data, model):
    input_data = input_data.apply(lambda x: [word for word in x if word in model.vocab])
    vectors = np.concatenate([build_sentence_vector(model, sentence, size=256) for sentence in input_data])
    feature_vec = pd.DataFrame(vectors)
    return feature_vec


if __name__ == '__main__':
    train_First = pd.read_csv('E:/AIcompetition/DataFountainScore/data/train_first.csv')
    predict_First = pd.read_csv('E:/AIcompetition/DataFountainScore/data/predict_first.csv')
    train_First = split_words(train_First)
    predict_First = split_words(predict_First)
    train_w2v_data = pd.concat([train_First['Discuss_NoSym_split'], predict_First['Discuss_NoSym_split']], ignore_index=True)
    stopwords = get_stopwords_dict(file_path='E:/AIcompetition/DataFountainScore/data/stopwords.txt')
    create_w2c_file(train_w2v_data, stopwords, 'discuss_w2v.txt')
    train_word2vec(input_data='discuss_w2v.txt', model_name='model.bin')
    model = load_w2c_model(model_name='model.bin')
    TrainData = create_discuss_feature(train_First['Discuss_NoSym_split'], model)
    PreData = create_discuss_feature(predict_First['Discuss_NoSym_split'], model)
    TrainData['Score'] = train_First['Score']
    TrainData['Id'] = train_First['Id']
    PreData['Id'] = predict_First['Id']
    TrainData.to_csv('E:/AIcompetition/DataFountainScore/data/TrainData.csv', index=False, encoding='utf-8')
    PreData.to_csv('E:/AIcompetition/DataFountainScore/data/PreData.csv', index=False, encoding='utf-8')
