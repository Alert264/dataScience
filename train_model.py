import json
import os
import warnings
import time
# import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')


def read_text(filepath):
    """
    :param filepath: 文件路径
    :return: 返回值
        features: 文本(特征)数据，以列表形式返回;
          labels: 分类标签，以列表形式返回
    """
    features, labels = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for news in data:
        features.append(news.get("word_list"))
        labels.append(news.get("label"))
    return features, labels


def merge_text(train_or_test_path):
    """
    :param train_or_test_path: train 训练数据集  test 测试数据集 的根目录
    :return: 返回值
        merge_features: 合并好的所有特征数据，以列表形式返回;
          merge_labels: 合并好的所有分类标签数据，以列表形式返回
    """
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '正在合并的数据位于:' + train_or_test_path)
    merge_features, merge_labels = [], []
    file_list = os.listdir(train_or_test_path)
    for file in file_list:
        features, labels = read_text(train_or_test_path + "/" + file)
        merge_features += features
        merge_labels += labels
    print("[info]: 样本数量：" + str(len(merge_labels)))
    return merge_features, merge_labels


def convert_to_matrix(train_path, test_path):
    """
    :param train_path: 训练数据集路径
    :param test_path: 测试数据集路径
    :return
        x_train_count: 训练数据的特征矩阵
         x_test_count: 测试数据的特征矩阵
           y_train_le: 训练数据的标签矩阵
            y_test_le: 测试数据的标签矩阵
    """
    x_train, y_train = merge_text(train_path)
    x_test, y_test = merge_text(test_path)
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)
    # print(y_train_le)
    # print(y_test_le)
    count = CountVectorizer()
    count.fit(list(x_train) + list(x_test))
    x_train_count = count.transform(x_train).toarray()
    x_test_count = count.transform(x_test).toarray()
    # print(x_train_count.shape, x_test_count.shape)
    # print(x_train_count)
    # print(x_test_count)
    return x_train_count, x_test_count, y_train_le, y_test_le


# 用于存储所有算法的名字，准确率和所消耗的时间
estimator_list, score_list, time_list = [], [], []


def get_text_classification(estimator, X, y, X_test, y_test):
    """
    :param estimator: 分类器，必选参数
    :param X: 特征训练数据，必选参数
    :param y: 标签训练数据，必选参数
    :param X_test: 特征测试数据，必选参数
    :param y_test: 标签测试数据，必选参数
    :return
       y_pred_model: 预测值
         classifier: 分类器名字
              score: 准确率
                  t: 消耗的时间
              matrix: 混淆矩阵
              report: 分类评价函数
    """
    start = time.time()

    print('\n>>>算法正在启动，请稍候...')
    model = estimator

    print('\n>>>算法正在进行训练，请稍候...')
    model.fit(X, y)
    print(model)

    print('\n>>>算法正在进行预测，请稍候...')
    y_pred_model = model.predict(X_test)
    print(y_pred_model)

    print('\n>>>算法正在进行性能评估，请稍候...')
    score = metrics.accuracy_score(y_test, y_pred_model)
    matrix = metrics.confusion_matrix(y_test, y_pred_model)
    report = metrics.classification_report(y_test, y_pred_model)

    print('>>>准确率\n', score)
    print('\n>>>混淆矩阵\n', matrix)
    print('\n>>>召回率\n', report)
    print('>>>算法程序已经结束...')

    end = time.time()
    t = end - start
    print('\n>>>算法消耗时间为：', t, '秒\n')
    classifier = str(model).split('(')[0]

    return y_pred_model, classifier, score, round(t, 2), matrix, report


def decision_tree(train_path, test_path):
    knc = DecisionTreeClassifier()
    x_train_count, x_test_count, y_train_le, y_test_le = convert_to_matrix(train_path, test_path)
    result = get_text_classification(knc, x_train_count, y_train_le, x_test_count, y_test_le)
    estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


if __name__ == '__main__':
    train_path = "train/sina_news"
    test_path = "test/sina_news"
    decision_tree(train_path, test_path)


