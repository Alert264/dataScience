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
        print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + "正在分词：" + date)
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
    print("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + "从" + str(start_date) + "开始，共" + str(days) + "天的分词完成")


if __name__ == '__main__':
    # start_date: 开始日期      格式：YYYY-MM-DD
    # days      : 天数          int
    start_date = '2020-02-01'
    days = 5
    root_path = 'sina_news'
    save_word_list(start_date, days, root_path)
