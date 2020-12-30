import requests
from bs4 import BeautifulSoup
import re
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

conn = requests.Session()


def get_news_title_and_url(url):
    error = ""
    try:
        # html = requests.get(url)
        html = conn.get(url)
        html.encoding = html.apparent_encoding
        data = re.sub('try\{jQuery\d*_\d*\(', '', html.text)
        data = re.sub('\);\}catch\(e\)\{\};', '', data)
        data = json.loads(data)
        news_list_0 = data['result']['data']
        total = int(data['result']['total'])
        news_list = []
        for news in news_list_0:
            news_list.append({"title": news['title'], "url": news['url']})
        return total, news_list
    except:
        write_error("get_news_title_and_url error:" + " No news in this page." + "  url: " + url)
        return 0, ""


def get_one_day_news_list(time_stamp):
    today = time_stamp
    tomorrow = today + 86400
    url_list = []
    url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&etime=" + str(today) + "&stime=" + str(
        tomorrow) + "&ctime=" + str(tomorrow) + "&date=" + time.strftime("%Y-%m-%d", time.localtime(
        today)) + "&k=&num=50&page=1&r=0.688144202509144&callback=jQuery111205553092203077052_1606993792554&_=1606993792626"
    total, url_list = get_news_title_and_url(url)
    print(time.strftime("%Y-%m-%d", time.localtime(today)) + ": 第 1 页正常。")
    for i in range(2, total // 50 + 2):
        try:
            url = "https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid=2509&etime=" + str(today) + "&stime=" + str(
                tomorrow) + "&ctime=" + str(tomorrow) + "&date=" + time.strftime("%Y-%m-%d", time.localtime(
                today)) + "&k=&num=50&page=" + str(
                i) + "&r=0.688144202509144&callback=jQuery111205553092203077052_1606993792554&_=1606993792626"
            useless, temp = get_news_title_and_url(url)
            url_list.extend(temp)
            print(time.strftime("%Y-%m-%d", time.localtime(today)) + ": 第 " + str(i) + " 页正常。")
        except:
            print(time.strftime("%Y-%m-%d", time.localtime(today)) + ": 第 " + str(i) + " 页发生异常。")
    return url_list


def get_news(url):
    try:
        html = requests.get(url)
        html.encoding = html.apparent_encoding
        # p = re.findall("<p cms-style=\"font-L\">(.*?)</p>", html.text)
        soup = BeautifulSoup(html.text, "html.parser")
        p = soup.find_all('p')
        # return p
        return data_clean(p)
    except:
        write_error("get_news error:" + url)
        return ""


def data_clean(data_list):
    dirty = []
    dirty.append("|")
    dirty.append("24小时滚动播报最新的财经资讯和视频，更多粉丝福利扫描二维码关注（sinafinance）")
    dirty.append("新浪财经意见反馈留言板")
    dirty.append("电话：400-052-0066 欢迎批评指正")
    dirty.append("新浪简介")
    dirty.append("广告服务")
    dirty.append("About Sina")
    dirty.append("联系我们")
    dirty.append("招聘信息")
    dirty.append("通行证注册")
    dirty.append("产品答疑")
    dirty.append("网站律师")
    dirty.append("SINA English")
    dirty.append("Copyright © 1996-2019 SINA Corporation")
    dirty.append("All Rights Reserved")
    dirty.append("新浪公司 ")
    dirty.append("版权所有")
    dirty.append("违法和不良信息举报电话：")
    dirty.append("010-62675637")
    dirty.append("更多猛料！欢迎扫描左方二维码关注新浪新闻官方微信（xinlang-xinwen）")
    dirty.append("感知中国经济的真实温度，见证逐梦时代的前行脚步。")
    dirty.append("谁能代表2019年度商业最强驱动力？")
    dirty.append("点击投票，评选你心中的“2019十大经济年度人物”")
    dirty.append("【我要投票】")
    dirty.append("安装新浪财经客户端第一时间接收最全面的市场资讯→【下载地址】安装新浪财经客户端第一时间接收最全面的市场资讯→【下载地址】")
    # dirty.append("")
    dirty.append(', ')

    data = str(data_list).replace("[", "").replace("]", "")
    data = re.findall(">(.*?)<", data)
    data = "".join(data)

    for i in range(0, len(dirty)):
        data = data.replace(dirty[i], "")
    return data


def save_news_by_day(today, url_list):
    # print('%s is running:' % os.getpid(), end=": ")
    # url_list = get_one_day_news_list(today)
    # print(url_list)
    try:
        data = {"date": time.strftime("%Y-%m-%d", time.localtime(today))}
        news_list = []
        for i in range(0, len(url_list)):
        # for i in range(0, 10):
            print(time.strftime("%Y-%m-%d", time.localtime(today)) + "： 正在获取第" + str(i) + "条新闻。")
            title = url_list[i]["title"]
            news = get_news(url_list[i]["url"])
            news_list.append({"title": title, "text": news})
        print(time.strftime("%Y-%m-%d", time.localtime(today)) + ": " + str(len(url_list)) + "条新闻获取完成，写入文件中。。。")
        data["news"] = news_list
        filename = "D:\\pythonproject\\dataScience\\source\\sina_news\\" + time.strftime("%Y-%m-%d", time.localtime(today)) + ".json"
        with open(filename, "w+", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(time.strftime("%Y-%m-%d", time.localtime(today)) + ": 写入文件完成。")
    except:
        write_error("save_news_by_day error" + time.strftime("%Y-%m-%d", time.localtime(today)))


def write_error(error):
    log_file = "D:\\pythonproject\\dataScience\\log\\log.txt"
    with open(log_file, "a+", ) as log:
        log.write("[" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())) + "]:" + error + "\n")


if __name__ == '__main__':
    # date = 1575734400     # 2020-12-08

    executor = ProcessPoolExecutor(max_workers=3)

    tss1 = '2020-04-28 00:00:00'
    timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    date = int(time.mktime(timeArray))

    for j in range(0, 3):
        url_lists = get_one_day_news_list(date)
        future = executor.submit(save_news_by_day, date, url_lists)
        date = date + 86400
    executor.shutdown(True)

    # url_lists = get_one_day_news_list(date)
    # save_news_by_day(date, url_lists)
