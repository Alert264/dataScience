# -*- coding: utf-8 -*-
import json
import operator
import time
import jieba
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys


# 创建停用词列表
def get_stopwords_list(stop_words_file_name):
    stopwords = [line.strip() for line in open(stop_words_file_name, encoding='UTF-8').readlines()]
    return stopwords

# 去除停用词
def move_stopwords(sentence_list, stopwords_list):
    out_list = []
    for index, word in enumerate(sentence_list):

        if word == '新' and index+1 < len(sentence_list) and sentence_list[index + 1] == '冠':
            word = '新冠'
            sentence_list[index + 1] = ''
        elif word not in stopwords_list:
            if word != '\t':
                out_list.append(word)
    return out_list



# 将文章转化为词组
def get_paticle_words(t,stop_words_file_name='cn_stopwords.txt'):
    base = move_stopwords(jieba.lcut(t), get_stopwords_list(stop_words_file_name))
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
def judge_relativeness(words,relative_words_file='relative_words.txt'):
    relative_words = [line.strip() for line in open(relative_words_file, encoding='UTF-8').readlines()]
    for i in relative_words:
        if (i in words):
            return True
    return False


def compute_TFIDF(start_date, days,relative_words_file='relative_words.txt',stop_words_file_name='cn_stopwords.txt',mode='sina_news',file_title='tfidf_source',n = 20):
    for i in range(days):
        date = int_to_date(date_to_int(start_date) + 86400 * i)
        save_IDF(date,relative_words_file,stop_words_file_name,mode,file_title,n)
    print("从" + str(start_date) + "开始，共" + str(days) + "天的关键词记录完成."+"保存在："+file_title)


def save_IDF(date,relative_words_file='relative_words.txt',stop_words_file_name='cn_stopwords',mode='sina_news',file_title='tfidf_source',n = 20):
    current = sys.path[0].replace('\\','/') + '/'
    corpus = []
    dic = {}
    filename = 'source/sina_news' + '/' + date + ".json"
    fileTitle = file_title
    if mode == 'sina_news':

        filename = current + 'source/sina_news' + '/' + date + ".json"
        with open(filename, 'r', encoding='utf-8') as File:
            data = json.load(File).get("news")
        File.close()
        for part in data:
            new = ""
            word_list = get_paticle_words(part.get('text'))
            if (judge_relativeness(word_list,relative_words_file)):
                new = " ".join(word_list)
                corpus.append(new)

    elif mode == 'weibo_comments':
        filename = current + 'source/weibo_comments/'+date + '.json'
        #current + 'source/weibo_comments/' + date + ".json"
        with open(filename, 'r', encoding='utf-8') as File:
            data = json.load(File)
            for element in data:
                # part = element.get("微博内容")
                # new = ""
                comments = ",".join(element.get('评论'))
                word_list = get_paticle_words(comments, stop_words_file_name)

                new = " ".join(word_list)
                corpus.append(new)
        File.close()

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

        tmp_dic["第" + str(i + 1) + "篇文章"] = order_dict(IDF,20)
    dic["前"+str(n)+"个关键词"] = tmp_dic
    filename = fileTitle + '/' + date + ".json"

    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)

    f.close()
    print(date + ":"+ "写入关键词完成。")

def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=True)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1

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
    # n为每篇文章保留的关键词数量
    compute_TFIDF(start_date, days,file_title="tfidf_comments",mode='weibo_comments',n = 20)
