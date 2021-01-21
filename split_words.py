import json
import time
from compute_TFIDF import *


def date_to_int(date):
    timeArray = time.strptime(date, "%Y-%m-%d")
    return int(time.mktime(timeArray))


def int_to_date(integer):
    return time.strftime("%Y-%m-%d", time.localtime(integer))


def save_word_list(start_date, days, root_path):
    for i in range(days):
        date = int_to_date(date_to_int(start_date) + 86400 * i)
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]:" + "正在分词：" + date)
        corpus = []
        filename = 'source/' + root_path + '/' + date + ".json"
        with open(filename, 'r', encoding='utf-8') as File:
            data = json.load(File).get("news")
        for part in data:
            new = ""
            title = part.get('title')
            word_list = get_paticle_words(part.get('text'))
            if (judge_relativeness(word_list)):
                new = " ".join(word_list)
                corpus.append({"label": "", "title": title, "word_list": new})
        filename = 'word_list' + '/' + root_path + '/' + date + ".json"
        with open(filename, 'w+', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + date + ": 写入分词完成。")
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + "从" + str(
        start_date) + "开始，共" + str(days) + "天的分词完成")


def save_word_list_comment(start_date, days, root_path):
    for i in range(days):
        date = int_to_date(date_to_int(start_date) + 86400 * i)
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S") + "]:" + "正在分词：" + date)
        corpus = []
        filename = 'source/' + root_path + '/' + date + ".json"
        with open(filename, 'r', encoding='utf-8') as File:
            data = json.load(File)
        File.close()

        for part in data:
            word_list = []
            new = ""
            title = part.get('微博内容')
            count = part.get('评论数量')
            reviewList = part.get('评论')
            for review in reviewList:
                word_list += get_paticle_words(review)

            new = " ".join(word_list)
            corpus.append({"label": "", "title": title, "count": count, "word_list": new})
        filename = 'word_list' + '/' + root_path + '/' + date + ".json"
        with open(filename, 'w+', encoding='utf-8') as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + date + ": 写入分词完成。")
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + "从" + str(
        start_date) + "开始，共" + str(days) + "天的分词完成")


if __name__ == '__main__':
    # start_date: 开始日期      格式：YYYY-MM-DD
    # days      : 天数          int
    start_date = '2019-12-08'
    days = 206
    root_path = 'rmrb'
    save_word_list_comment(start_date, days, root_path)

# 2020年5月下次获取第 33 页
