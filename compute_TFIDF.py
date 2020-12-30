# -*- coding: utf-8 -*-
import json
import operator
import time
import jieba
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# 创建停用词列表
def get_stopwords_list():
    stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


# 去除停用词
def move_stopwords(sentence_list, stopwords_list):
    out_list = []
    for index, word in enumerate(sentence_list):
        if word == '新' and sentence_list[index + 1] == '冠':
            word = '新冠'
            sentence_list[index + 1] = ''
        elif word not in stopwords_list:
            if word != '\t':
                out_list.append(word)
    return out_list


# 将文章转化为词组
def get_paticle_words(t):
    base = move_stopwords(jieba.lcut(t), get_stopwords_list())
    pattern = re.compile('[\u4e00-\u9fa5]')
    ans = []
    for index, i in enumerate(base):
        m = pattern.match(str(i))
        if m != None:
            ans.append(str(i))
        else:
            pass
    return ans


# 判断文章是否和疫情相关
def judge_relativeness(words):
    relative_words = [line.strip() for line in open('relative_words.txt', encoding='UTF-8').readlines()]
    for i in relative_words:
        if (i in words):
            return True
    return False


def compute_TFIDF(start_date, days):
    for i in range(days):
        date = int_to_date(date_to_int(start_date) + 86400 * i)
        save_IDF(date)
    print("从" + str(start_date) + "开始，共" + str(days) + "天的TF*IDF记录完成")


def save_IDF(date):
    corpus = []
    dic = {}
    filename = 'source' + '/' + date + ".json"
    with open(filename, 'r', encoding='utf-8') as File:
        data = json.load(File).get("news")
    for part in data:
        new = ""
        word_list = get_paticle_words(part.get('text'))
        if (judge_relativeness(word_list)):
            new = " ".join(word_list)
            corpus.append(new)

    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    # 写入 完成输出 'frequency of words written'
    tmp_dic = {}
    dic["date"] = date

    for i in range(len(weight)):
        IDF = {}
        for j in range(len(word)):
            if weight[i][j] != 0:
                IDF[word[j]] = weight[i][j]
        IDF = dict(sorted(IDF.items(), key=operator.itemgetter(1), reverse=True))
        tmp_dic["第" + str(i + 1) + "篇文章"] = IDF
    dic["TF*IDF"] = tmp_dic
    filename = 'tfidf_source' + '/' + date + ".json"
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)
    print(date + ": 写入TF*IDF完成。")


def date_to_int(date):
    timeArray = time.strptime(date, "%Y-%m-%d")
    return int(time.mktime(timeArray))


def int_to_date(integer):
    return time.strftime("%Y-%m-%d", time.localtime(integer))


if __name__ == '__main__':
    # start_date: 开始日期      格式：YYYY-MM-DD
    # days      : 天数          int
    start_date = '2019-12-08'
    days = 1
    compute_TFIDF(start_date, days)
