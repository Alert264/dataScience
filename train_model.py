import json
import os
import warnings
import sys
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import mean_squared_error
import lightgbm as lightgbm
import xgboost
import data_visualization

warnings.filterwarnings('ignore')

num_to_label = {0: '自豪', 1: '平静', 2: '高兴', 3: '恐惧', 4: '忧愁', 5: '疑惑', 6: '同情', 7: '愤怒',
                8: '喜爱', 9: '悲哀', 10: '感动', 11: '期望', 12: "着急", 13: '满意', 14: '羡慕', 15: '惊讶', -1: ""}

label_to_num = {'自豪': 0, '平静': 1, '高兴': 2, '恐惧': 3, '忧愁': 4, '疑惑': 5, '同情': 6, '愤怒': 7,
                '喜爱': 8, '悲哀': 9, '感动': 10, '期望': 11, "着急": 12, '满意': 13, '羡慕': 14, '惊讶': 15, "": -1}


def read_text(filepath):
    """
    :param filepath: 文件路径
    :return: 返回值
        features: 文本(特征)数据，以列表形式返回;
          labels: 分类标签，以列表形式返回
    """
    features, labels = [], []
    # print(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            print(filepath)
    for news in data["前20个关键词"]:
        features.append(" ".join(news["keywords"]))
        try:
            # if news["label"] == "":
            #     print(filepath)
            labels.append(label_to_num[news["label"]])
        except:
            print(filepath)
            print(news["label"])
            sys.exit(0)
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
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 样本数量：" + str(len(merge_labels)))
    return merge_features, merge_labels


def convert_to_matrix(train_path, test_path, predict_path):
    """
    :param train_path: 训练数据集路径
    :param test_path: 测试数据集路径
    :param predict_path: 预测数据路径
    :return
        x_train_count: 训练数据的特征矩阵
         x_test_count: 测试数据的特征矩阵
           y_train_le: 训练数据的标签矩阵
            y_test_le: 测试数据的标签矩阵
    """
    x_train, y_train = merge_text(train_path)
    x_test, y_test = merge_text(test_path)
    x_predict, y_predict = merge_text(predict_path)
    le = LabelEncoder()
    y_train_le = le.fit_transform(y_train)
    y_test_le = le.fit_transform(y_test)
    # print(y_train_le)
    # print(y_test_le)
    count = CountVectorizer()
    count.fit(list(x_train) + list(x_test) + list(x_predict))
    x_train_count = count.transform(x_train).toarray()
    x_test_count = count.transform(x_test).toarray()
    # print(x_train_count.shape, x_test_count.shape)
    # print(x_train_count)
    # print(x_test_count)
    return x_train_count, x_test_count, y_train_le, y_test_le


# 用于存储所有算法的名字，准确率和所消耗的时间
# estimator_list, score_list, time_list = [], [], []


def get_text_classification(estimator, X, y, X_test, y_test, saveModel=True):
    """
    :param estimator: 分类器，必选参数
    :param X: 特征训练数据，必选参数
    :param y: 标签训练数据，必选参数
    :param X_test: 特征测试数据，必选参数
    :param y_test: 标签测试数据，必选参数
    :return
       y_pred_model: 预测值
         classifier: 分类器名字
              test_score: 测试集得分
              train_score: 训练集得分
                  t: 消耗的时间
              matrix: 混淆矩阵
              report: 分类评价函数
    """
    start = time.time()
    model = estimator

    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '正在进行训练模型，请稍候...')
    model.fit(X, y)
    print(model)

    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '正在进行预测，请稍候...')
    y_pred_model = model.predict(X_test)
    y_train_pred = model.predict(X)
    print(y_pred_model)

    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '正在进行性能评估，请稍候...')
    test_score = metrics.accuracy_score(y_test, y_pred_model)
    train_score = metrics.accuracy_score(y, y_train_pred)

    matrix = metrics.confusion_matrix(y_test, y_pred_model)
    report = metrics.classification_report(y_test, y_pred_model)

    if saveModel:
        save_model(model, test_score)

    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '准确率: ', test_score)
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '训练集得分：: ', train_score)
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '混淆矩阵\n', matrix)
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '召回率\n', report)
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '算法程序已经结束...')

    end = time.time()
    t = end - start
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: " + '算法消耗时间为：', t, '秒\n')
    classifier = str(model).split('(')[0]

    return y_pred_model, classifier, test_score, train_score, round(t, 2), matrix, report


def plot_learning_curve(algorithm, train_path, test_path, predict_path):
    train_score = []
    test_score = []
    X = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
    x_train_count, x_test_count, y_train_le, y_test_le = convert_to_matrix(train_path, test_path, predict_path)
    for i in range(10):
        if i == 9:
            result = get_text_classification(algorithm, x_train_count[::10 - i], y_train_le[::10 - i],
                                             x_test_count[::10 - i], y_test_le[::10 - i], saveModel=True)
        else:
            result = get_text_classification(algorithm, x_train_count[::10 - i], y_train_le[::10 - i],
                                             x_test_count[::10 - i], y_test_le[::10 - i], saveModel=False)
        train_score.append(result[3])
        test_score.append(result[2])
    print(train_score)
    print(test_score)
    data_visualization.plot_learning_curves(train_score, test_score, X, str(algorithm).split('(')[0])


def train_model(algorithm, train_path, test_path, predict_path):
    """
    :param algorithm: 算法
    :param train_path: 训练集路径
    :param test_path: 测试集路径
    :param predict_path 预测数据路径
    :return:
    """
    x_train_count, x_test_count, y_train_le, y_test_le = convert_to_matrix(train_path, test_path, predict_path)
    result = get_text_classification(algorithm, x_train_count, y_train_le, x_test_count, y_test_le)
    # plot_learning_curve(algorithm, x_train_count, x_test_count, y_train_le, y_test_le)
    # estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


def save_model(model, accuracy):
    """
    :param model: 训练好的模型
    :param accuracy: 模型准确率
    :return:
    """
    file_path = "model/" + str(round(time.time() * 1000)) + "-" + str(model).split('(')[0] + "-" + str(
        int(accuracy * 100)) + "%.m"
    # 文件名加时间戳以区分不同阶段训练结果
    # print(file_path)
    joblib.dump(model, file_path)


def predict(model_paths, train_path, test_path, predict_path, store_path):
    """
    :param model_paths: 模型路径,list, 按准确率从大到小排练
    :param train_path: 训练集路径
    :param test_path: 测试集路径
    :param predict_path: 要预测的数据路径
    :param store_path: 预测结果存储路径
    :return:
    """
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在加载第 1 个模型...")
    model_0 = joblib.load(model_paths[0])
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在加载第 2 个模型...")
    # model_1 = joblib.load(model_paths[1])
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在加载第 3 个模型...")
    # model_2 = joblib.load(model_paths[2])
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 模型加载完成，正在预测...")
    x_train, y_train = merge_text(train_path)
    x_test, y_test = merge_text(test_path)
    # x_predict_0, y_predicted_0 = merge_text("temp/rmrb")
    x_predict, y_predict = merge_text(predict_path)
    count = CountVectorizer()
    count.fit(list(x_train) + list(x_test) + list(x_predict))
    x_predict_count = count.transform(x_predict).toarray()
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在使用第 1 个模型预测...")
    y_predicted = model_0.predict(x_predict_count)
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在使用第 2 个模型预测...")
    # y_predicted_1 = model_1.predict(x_predict_count)
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在使用第 3 个模型预测...")
    # y_predicted_2 = model_2.predict(x_predict_count)
    # y_predicted = []
    # print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 合并预测结果。")
    # for i in range(len(y_predicted_0)):
    #     if y_predicted_1[i] == y_predicted_2[i]:
    #         y_predicted.append(y_predicted_1[i])
    #     else:
    #         y_predicted.append(y_predicted_0[i])
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 预测结束，开始写入文件...")

    file_list = os.listdir(predict_path)
    y_predicted_count = 0

    for file in file_list:
        write_data = {"date": file.replace(".json", "")}
        labels = []
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 正在写入: " + file.replace(".json", ""))
        with open(predict_path + "/" + file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                print(file)
        for news in data["前20个关键词"]:
            news["label"] = num_to_label[y_predicted[y_predicted_count]]
            y_predicted_count = y_predicted_count + 1
            labels.append(news)
        write_data["前20个关键词"] = labels
        with open(store_path + "/" + file, "w+", encoding='utf-8') as f:
            json.dump(write_data, f, ensure_ascii=False, indent=4)
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]: 写入完成: " + store_path + "/" + file)


def predict_one_news(news_keywords, model=None):
    # 此函数用于模型训练好以后随机测试一篇新闻评论
    """
    :param news_keywords: 一则新闻评论关键词，空格隔开，字符串
    :return: num_to_label[y_predicted[0]]: 关键词对应情绪
    """
    model_path = "model/1611320988907-LogisticRegression-52%.m"
    train_path = "train/rmrb"
    test_path = "test/rmrb"
    predict_path = "temp/rmrb"

    if model is None:
        model = joblib.load(model_path)
    x_train, y_train = merge_text(train_path)
    x_test, y_test = merge_text(test_path)
    x_predict, y_predict = merge_text(predict_path)
    count = CountVectorizer()
    count.fit(list(x_train) + list(x_test) + list(x_predict))
    x_predict_count = count.transform([news_keywords]).toarray()
    y_predicted = model.predict(x_predict_count)
    return num_to_label[y_predicted[0]]


if __name__ == '__main__':
    train_path = "train/xlxw"
    test_path = "test/xlxw"
    predict_path = "filtered/xlxw"
    store_path = "predicted/xlxw(4)"

    # features, labels = merge_text(train_path)
    # features1, labels1 = merge_text(test_path)
    # print(features)
    # # print(labels)
    # print(set(labels))
    # print("--" * 64)
    # print(features1)
    # # print(labels)
    # print(set(labels1))

    # k 近邻算法
    # algorithm = KNeighborsClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 决策树
    # algorithm = DecisionTreeClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 多层感知器
    # algorithm = MLPClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 伯努力贝叶斯算法
    # algorithm = BernoulliNB()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 高斯贝叶斯
    # algorithm = GaussianNB()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 多项式朴素贝叶斯
    # algorithm = MultinomialNB()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 逻辑回归算法
    algorithm = LogisticRegression()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 支持向量机算法
    # algorithm = svm.SVC(kernel='linear')
    # train_model(algorithm, train_path, test_path, predict_path)
    # 随机森林算法
    # algorithm = RandomForestClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # 自增强算法
    # algorithm = AdaBoostClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # lightgbm算法
    # algorithm = lightgbm.LGBMClassifier()
    # train_model(algorithm, train_path, test_path, predict_path)
    # xgboost算法
    # algorithm = xgboost.XGBClassifier(objective="multi：softmax  num_class=16")
    # train_model(algorithm, train_path, test_path, predict_path)
    # plot_learning_curve(algorithm, train_path, test_path, predict_path)
    # predict(["model/1611464257925-LogisticRegression-75%.m"], train_path, test_path, predict_path, store_path)
    predict(["model/1611583839453-LogisticRegression-72%.m"], train_path, test_path, predict_path, store_path)

