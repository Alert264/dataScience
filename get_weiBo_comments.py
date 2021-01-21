import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
from lxml import html
etree = html.etree
import random_proxy_pool
import winsound


class Weibospider:
    def __init__(self, date):
        # 获取首页的相关信息：
        # self.start_url = 'https://weibo.com/2803301701/Is5xeqvl1?from=page_1002062803301701_profile&wvr=6&mod=weibotime&type=comment#_rnd1608714031543'
        # self.start_url = 'https://weibo.com/rmrb?is_all=1&stat_date=201912&page={0}#1608738638854'
        # 人民日报
        # self.start_url = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=' + date + '&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page=0'
        # 央视新闻
        # self.start_url = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=' + date + '&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__26&id=1002062656274875&script_uri=/cctvxinwen&feed_type=0&pre_page=0'
        # 新浪新闻
        self.start_url = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=' + date + '&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__26&id=1002062028810631&script_uri=/sinapapers&feed_type=0&pre_page=0'

        self.headers = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
            "cache-control": "max-age=0",
            # "cookie": '_ga=GA1.2.251401487.1603990360; _s_tentry=-; Apache=6459632317051.233.1608711051353; SINAGLOBAL=6459632317051.233.1608711051353; ULV=1608711051359:1:1:1:6459632317051.233.1608711051353:; login_sid_t=836768862393a5ea300038c4609f06ba; cross_origin_proto=SSL; SSOLoginState=1608712142; SUB=_2A25y5o-BDeRhGeNI71MT9SnOzD2IHXVuKBHJrDV8PUJbkNAfLRemkW1NSGU_PxQfFB3h56wBChPxrQ09qdGb_SO4; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhSxjHssiu67T8-eFN2ZgH95NHD95QfSoBpeo-NeoMpWs4DqcjGHcL.9-v79Btt; wvr=6; wb_view_log_5641257271=1440*9001.600000023841858; UOR=,,graph.qq.com; webim_unReadCount=%7B%22time%22%3A1608713178777%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A37%2C%22msgbox%22%3A0%7D',
            "cookie": 'SINAGLOBAL=6231556531733.056.1588128770990; _ga=GA1.2.958850115.1607182656; UM_distinctid=1765c20c5e96aa-09b5b8fb4e915d-930346c-144000-1765c20c5ea518; _s_tentry=login.sina.com.cn; Apache=4789115122888.512.1611039342080; ULV=1611039342321:15:1:1:4789115122888.512.1611039342080:1609424214640; login_sid_t=61cc70fad967187ac88cb6720aa5a2a2; cross_origin_proto=SSL; UOR=,,login.sina.com.cn; SSOLoginState=1611112413; wvr=6; wb_view_log_5715604584=1536*8641.25; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhzI8YFvNo.8GUaj2GrDIlj5JpX5KMhUgL.Fo-NeK-cehBf1hB2dJLoIXzLxKnLBo-LBoMLxKBLBonLB-2LxK-L1K5L12BLxK-LB-BL1KMLxK-LBK-LB.BLxK.L1-2LB.-LxK-L1K-L122ESoBt; ALF=1642735195; SCF=AlB5cnq7q8EgwK56rx6GguAFhcbJSfuZSFx80QoUcPRXzhrfo_jR9YkhLAJ-nFb5fzl7QEsBITd6IzQuvPSt3Qo.; SUB=_2A25NDIKNDeRhGeNJ6lcX8CrJwziIHXVue_NFrDV8PUNbmtAKLRfykW9NS-Qk9xtMUcxLWh6I6y-BG91Q4IAz3oi3; webim_unReadCount=%7B%22time%22%3A1611199487322%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A0%2C%22msgbox%22%3A2%7D',
            "referer": "https://www.weibo.com/u/5644764907?topnav=1&wvr=6&topsug=1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36",
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
            print("[warning] 响应超时，重新获取")
            proxy = random_proxy_pool.get_quick_proxy()
            start = time.time()
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        # res.encoding = res.apparent_encoding
        # print(res.text)
        response = res.content.decode().replace("\\", "")

        # every_url = re.compile('target="_blank" href="(/\d+/\w+\?from=\w+&wvr=6&mod=weibotime)" rel="external nofollow" ', re.S).findall(response)
        every_id = re.compile('name=(\d{16})', re.S).findall(response)  # 获取次级页面需要的id
        home_url = []
        for id in every_id:
            base_url = 'https://weibo.com/aj/v6/comment/big?ajwvr=6&id={}&from=singleWeiBo'
            url = base_url.format(id)
            home_url.append(url)

        count = 0
        while len(home_url) == 0 and count < 5:
            print("[warning] 第{0}次获取失败，重新获取".format(count + 1))
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
            count = count + 1
            response = res.content.decode().replace("\\", "")
            every_id = re.compile(r'name=(\d{16})', re.S).findall(response)  # 获取次级页面需要的id
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
        # for time_ in times:
        #     month = time_.split("u6708")[0].zfill(2)
        #     day = time_.split("u6708")[1].split("u65e5")[0].zfill(2)
        #     dates.append("2020-" + month + "-" + day)
        for time_ in times:
            year = time_.split("-")[0]
            month = time_.split("-")[1].zfill(2)
            day = time_.split("-")[2].split(" ")[0].zfill(2)
            dates.append(year + "-" + month + "-" + day)
        return dates, title_list, home_url

    def parse_comment_info(self, url):  # 爬取直接发表评论的人的相关信息(name,info,time,info_url)
        start = time.time()
        proxy = random_proxy_pool.get_quick_proxy()
        res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        if '"count":null' in res.text:
            print(url)
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        while time.time() - start > 5:
            print("[warning] 获取评论超时，重新获取")
            start = time.time()
            proxy = random_proxy_pool.get_quick_proxy()
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})
            if '"count":null' in res.text:
                print(url)
                res = requests.get(url, headers=self.headers, proxies={'http': proxy})
            if '"count":null' in res.text:
                print(url)
                res = requests.get(url, headers=self.headers, proxies={'http': proxy})
        get_count = 0
        while res is None:
            print("[error]   获取评论出错!!!重新获取。", end="")
            print("\r第{}次尝试...".format(get_count + 1), end="")

            proxy = random_proxy_pool.get_quick_proxy()
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})

        while '"count":null' in res.text and get_count < 10:
            get_count = get_count + 1
            print(url)
            res = requests.get(url, headers=self.headers, proxies={'http': proxy})

        response = res.json()
        if '"count":null' in res.text:
            print("[warning] 请检查该新闻是否有评论。")
            print(url)
            print(res.text)
            beep(2)
            return 0, []
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
        # 人民日报
        # start_ajax_url1 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page={0}'
        # start_ajax_url2 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=1&pl_name=Pl_Official_MyProfileFeed__27&id=1002062803301701&script_uri=/rmrb&feed_type=0&pre_page={0}'
        # 央视新闻
        # start_ajax_url1 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__26&id=1002062656274875&script_uri=/cctvxinwen&feed_type=0&pre_page={0}'
        # start_ajax_url2 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=1&pl_name=Pl_Official_MyProfileFeed__26&id=1002062656274875&script_uri=/cctvxinwen&feed_type=0&pre_page={0}'
        # 新浪新闻
        start_ajax_url1 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=0&pl_name=Pl_Official_MyProfileFeed__26&id=1002062028810631&script_uri=/sinapapers&feed_type=0&pre_page={0}'
        start_ajax_url2 = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date={1}&page={0}&pagebar=1&pl_name=Pl_Official_MyProfileFeed__26&id=1002062028810631&script_uri=/sinapapers&feed_type=0&pre_page={0}'

        i = 0  # 页数-1
        while True:
            # if i == 4:
            #     break
            i += 1
            print("[info]    获取第 " + str(i) + " 页微博。")
            dates_1, titles_1, home_url = self.parse_home_url(start_url.format(i, self.date))  # 获取每一页的微博
            if len(home_url) == 0:
                print("[warning] 最后一页是：第 " + str(i-1) + " 页")
                break
            dates_2, titles_2, ajax_url1 = self.parse_home_url(start_ajax_url1.format(i, self.date))  # ajax加载页面的微博
            dates_3, titles_3, ajax_url2 = self.parse_home_url(start_ajax_url2.format(i, self.date))  # ajax第二页加载页面的微博
            all_url = home_url + ajax_url1 + ajax_url2
            all_dates = dates_1 + dates_2 + dates_3
            # print(all_dates)
            all_titles = titles_1 + titles_2 + titles_3

            if len(all_url) != len(all_dates) or len(all_titles) != len(all_dates) or len(all_url) != len(all_titles):
                print("[error]   各数组长度不一致。")
                print(len(all_url))
                print(len(all_titles))
                print(len(all_dates))
                beep(3)
                return
            #
            # print(all_url)
            # print(all_titles)
            # print(all_dates)

            for j in range(0, len(all_url)):
                print("[info]    获取第 " + str(i) + " 页第" + str(j + 1) + "条，总第 " + str((i-1) * 45 + j + 1) + " 条微博的评论。")
                # print(all_url[j])
                path_name = "source/xlxw/{0}.json".format(all_dates[j])
                # path_name = "temp/{0}.json".format(all_dates[j])
                print("[info]    获取第 " + str(i) + " 页第" + str(j + 1) + "条，总第 " + str((i - 1) * 45 + j + 1) + " 条微博第 1 页的评论。", end="")
                all_count, comment_info_list = self.parse_comment_info(all_url[j])
                # self.write_file(path_name, comment_info_list)
                data = {"微博内容": all_titles[j], "评论数量": all_count, "评论": comment_info_list}
                # print(data)
                for num in range(1, 20):
                    print("\r[info]    获取第 " + str(i) + " 页第" + str(j + 1) + "条，总第 " + str((i - 1) * 45 + j + 1) + " 条微博第 " + str(num + 1) + " 页的评论。", end="")
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
                print("\n[info]    第{}条微博信息获取完成！写入文件:".format((i-1) * 45 + j + 1) + path_name)
                # print("第{}条微博信息获取完成！".format((i-1) * 45 + j + 1))
            # break


def beep(times):
    for i in range(times):
        winsound.Beep(1000, 700)
        time.sleep(0.5)


if __name__ == '__main__':
        weibo = Weibospider("202006")
        weibo.run()
        beep(2)



