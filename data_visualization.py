import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
import train_model

labels = ['满意', '自豪', '平静', '高兴', '恐惧', '忧愁', '疑惑', '同情', '羡慕', '惊讶', '愤怒', '喜爱', '悲哀', '感动', '期望', "着急"]


def plot_data(data_path):
    # 多维尺度分析
    x, y = train_model.merge_text(data_path)
    count = CountVectorizer()
    count.fit(list(x))
    x_array = count.transform(x).toarray()
    print(x_array)
    news_labels = pd.DataFrame(x_array)
    news_labels = pd.merge(news_labels, pd.DataFrame(labels, columns=["mood"]), left_index=True, right_index=True)
    news_labels.head()
    similarities = euclidean_distances(x_array)
    mds = manifold.MDS(n_components=2, max_iter=500, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    X = mds.fit(similarities).embedding_
    print(X)
    pos = pd.DataFrame(X, columns=['X', 'Y'])
    pos['mood'] = y
    ax = pos[pos['mood'] == train_model.label_to_num["自豪"]].plot(kind='scatter', x='X', y='Y', color='blue', label='1')
    pos[pos['mood'] == train_model.label_to_num["期望"]].plot(kind='scatter', x='X', y='Y', color='green', label='2', ax=ax)
    pos[pos['mood'] == train_model.label_to_num["平静"]].plot(kind='scatter', x='X', y='Y', color='red', label='3', ax=ax)
    plt.show()


def date_to_int(date):
    timeArray = time.strptime(date, "%Y-%m-%d")
    return int(time.mktime(timeArray))


def int_to_date(integer):
    return time.strftime("%Y-%m-%d", time.localtime(integer))


def count_data(root_path):
    news_count = 0
    comments_count = 0
    start_date = "2019-12-08"
    for i in range(206):
        date = int_to_date(date_to_int(start_date) + 86400 * i)
        filename = root_path + '/' + date + ".json"
        with open(filename, 'r', encoding="utf-8") as f:
            data = json.load(f)
        if "source" in filename:
            for news in data:
                news_count += 1
                comments_count = comments_count + len(news["评论"])
        elif "filtered" in filename:
            news_count += len(data["前20个关键词"])
        elif "word_list" in filename:
            news_count += len(data)
    return news_count, comments_count


if __name__ == '__main__':
    # news_count_0, comments_count_0 = count_data("source/rmrb")
    # news_count_1, comments_count_1 = count_data("word_list/rmrb")
    # news_count_2, comments_count_2 = count_data("filtered/rmrb")
    #
    # print("原始新闻数量：", news_count_0)
    # print("原始评论总量：", comments_count_0)
    #
    # print("分词后新闻数量：", news_count_1)
    # print("筛选后新闻数量：", news_count_2)
    plot_data("test/rmrb")
