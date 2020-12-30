import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
from lxml import html
etree = html.etree
import random_proxy_pool


class Weibospider:
    def __init__(self, date):
        # 获取首页的相关信息：
        # self.start_url = 'https://weibo.com/2803301701/Is5xeqvl1?from=page_1002062803301701_profile&wvr=6&mod=weibotime&type=comment#_rnd1608714031543'
        # self.start_url = 'https://weibo.com/rmrb?is_all=1&stat_date=201912&page={0}#1608738638854'
        self.start_url = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=2020' + date + '&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page=0'
        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "max-age=0",
            "cookie": '_ga=GA1.2.251401487.1603990360; _s_tentry=-; Apache=6459632317051.233.1608711051353; SINAGLOBAL=6459632317051.233.1608711051353; ULV=1608711051359:1:1:1:6459632317051.233.1608711051353:; login_sid_t=836768862393a5ea300038c4609f06ba; cross_origin_proto=SSL; SSOLoginState=1608712142; SUB=_2A25y5o-BDeRhGeNI71MT9SnOzD2IHXVuKBHJrDV8PUJbkNAfLRemkW1NSGU_PxQfFB3h56wBChPxrQ09qdGb_SO4; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhSxjHssiu67T8-eFN2ZgH95NHD95QfSoBpeo-NeoMpWs4DqcjGHcL.9-v79Btt; wvr=6; wb_view_log_5641257271=1440*9001.600000023841858; UOR=,,graph.qq.com; webim_unReadCount=%7B%22time%22%3A1608713178777%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A37%2C%22msgbox%22%3A0%7D',
            "referer": "https://www.weibo.com/u/5644764907?topnav=1&wvr=6&topsug=1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36",
        }
        self.date = date
        # self.proxy = {
        #     'HTTP': 'HTTP://180.125.70.78:9999',
        #     'HTTP': 'HTTP://117.90.4.230:9999',
        #     'HTTP': 'HTTP://111.77.196.229:9999',
        #     'HTTP': 'HTTP://111.177.183.57:9999',
        #     'HTTP': 'HTTP://123.55.98.146:9999',
        # }

    def parse_home_url(self, url):  # 处理解析首页面的详细信息
        proxy = random_proxy_pool.get_quick_proxy()
        start = time.time()
        res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        while time.time() - start > 5:
            print("响应超时，重新获取")
            proxy = random_proxy_pool.get_quick_proxy()
            start = time.time()
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        response = res.content.decode().replace("\\", "")
        # every_url = re.compile('target="_blank" href="(/\d+/\w+\?from=\w+&wvr=6&mod=weibotime)" rel="external nofollow" ', re.S).findall(response)
        every_id = re.compile('name=(\d+)', re.S).findall(response)  # 获取次级页面需要的id
        home_url = []
        for id in every_id:
            base_url = 'https://weibo.com/aj/v6/comment/big?ajwvr=6&id={}&from=singleWeiBo'
            url = base_url.format(id)
            home_url.append(url)

        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(json.loads(res.text)["data"], "html.parser")
        titles = soup.find_all("div", class_="WB_text W_f14")  # 获取每条微博的内容
        title_list = []
        for title in titles:
            title_list.append(title.text.replace("\n", "").replace(" ", ""))

        times = re.compile('fromprofile"> (.*?)</a>').findall(response)  # 获取每条微博的时间
        dates = []
        for time_ in times:
            month = time_.split("u6708")[0].zfill(2)
            day = time_.split("u6708")[1].split("u65e5")[0].zfill(2)
            dates.append("2020-" + month + "-" + day)
        return dates, title_list, home_url

    def parse_comment_info(self, url):  # 爬取直接发表评论的人的相关信息(name,info,time,info_url)
        proxy = random_proxy_pool.get_quick_proxy()
        res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        response = res.json()
        count = response['data']['count']
        while count is None:
            print("[error]   获取评论出错!!!重新获取。")
            proxy = random_proxy_pool.get_quick_proxy()
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
            response = res.json()
            count = response['data']['count']
        html = etree.HTML(response['data']['html'])
        name = html.xpath("//div[@class='list_li S_line1 clearfix']/div[@class='WB_face W_fl']/a/img/@alt")  # 评论人的姓名
        info = html.xpath("//div[@node-type='replywrap']/div[@class='WB_text']/text()")  # 评论信息
        info = "".join(info).replace(" ", "").split("\n")
        info.pop(0)
        comment_time = html.xpath("//div[@class='WB_from S_txt2']/text()")  # 评论时间
        name_url = html.xpath("//div[@class='WB_face W_fl']/a/@href")  # 评论人的url
        name_url = ["https:" + i for i in name_url]
        comment_info_list = []
        for i in range(len(name)):
            item = {}
            # item["name"] = name[i]  # 存储评论人的网名
            item["comment_info"] = info[i]  # 存储评论的信息
            # item["comment_time"] = comment_time[i]  # 存储评论时间
            # item["comment_url"] = name_url[i]  # 存储评论人的相关主页
            comment_info_list.append(info[i].replace("：", ""))
        return count, comment_info_list

    def write_file(self, path_name, data):
        if os.path.exists(path_name):
            with open(path_name, "r", encoding='utf-8') as f:
                data_0 = json.load(f)
                data_0.append(data)
            with open(path_name, "w+", encoding='utf-8') as f:
                json.dump(data_0, f, ensure_ascii=False, indent=4)
        else:
            with open(path_name, "w+", encoding='utf-8') as f:
                data_0 = [data]
                json.dump(data_0, f, ensure_ascii=False, indent=4)

    def run(self):
        start_url = self.start_url
        start_ajax_url1 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=2020{1}&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page={0}'
        start_ajax_url2 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=2020{1}&page={0}&pagebar=1&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page={0}'
        i = 33
        while True:
            i += 1
            print("[info]    获取第 " + str(i) + " 页微博。")
            dates_1, titles_1, home_url = self.parse_home_url(start_url.format(i, self.date))  # 获取每一页的微博
            if len(home_url) == 0:
                print("[waring] 最后一页是：第 " + str(i-1) + " 页")
                break
            dates_2, titles_2, ajax_url1 = self.parse_home_url(start_ajax_url1.format(i, self.date))  # ajax加载页面的微博
            dates_3, titles_3, ajax_url2 = self.parse_home_url(start_ajax_url2.format(i, self.date))  # ajax第二页加载页面的微博
            all_url = home_url + ajax_url1 + ajax_url2
            all_dates = dates_1 + dates_2 + dates_3
            # print(all_dates)
            all_titles = titles_1 + titles_2 + titles_3

            for j in range(len(all_url)):
                print("[info]    获取第 " + str(i) + " 页第 " + str((i-1) * 45 + j + 1) + " 条微博的评论。")
                # print(all_url[j])
                path_name = "source\\weibo_comments\\{0}.json".format(all_dates[j])
                print("[info]    获取第 " + str(i) + " 页第" + str(j + 1) + "条，总第 " + str((i - 1) * 45 + j + 1) + " 条微博第 1 页的评论。")
                all_count, comment_info_list = self.parse_comment_info(all_url[j])
                # self.write_file(path_name, comment_info_list)
                data = {"微博内容": all_titles[j], "评论数量": all_count, "评论": comment_info_list}
                # print(data)
                for num in range(1, 20):
                    print("[info]    获取第 " + str(i) + " 页第" + str(j + 1) + "条，总第 " + str((i - 1) * 45 + j + 1) + " 条微博第 " + str(num + 1) + " 页的评论。")
                    if num * 15 < int(all_count) + 15:
                        comment_url = all_url[j] + "&page={}".format(num + 1)
                        # print(comment_url)
                        try:
                            count, comment_info_list = self.parse_comment_info(comment_url)
                            # self.write_file(path_name, comment_info_list)
                            # data["评论数量"] = data["评论数量"] + count
                            data["评论"] = data["评论"] + comment_info_list
                        except Exception as e:
                            print("Error:", e)
                            time.sleep(20)
                            count, comment_info_list = self.parse_comment_info(comment_url)
                            # self.write_file(path_name, comment_info_list)
                            # data["评论数量"] = data["评论数量"] + count
                            data["评论"] = data["评论"] + comment_info_list
                        del count
                        time.sleep(0.2)
                        # print(data)
                self.write_file(path_name, data)
                print("[info]    第{}条微博信息获取完成！写入文件:".format((i-1) * 45 + j + 1) + path_name)
                # print("第{}条微博信息获取完成！".format((i-1) * 45 + j + 1))
            # break


if __name__ == '__main__':
    # for i in range(1, 3):
    weibo = Weibospider("04")
    weibo.run()
