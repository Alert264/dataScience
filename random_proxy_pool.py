import requests
import random
import time

# 随机ip代理获取
PROXY_POOL_URL = 'http://localhost:5555/random'

proxies_0 = {
    '0': 'HTTP://118.212.105.90:9999',
    '1': 'HTTP://136.243.133.76:3128',
    '2': 'HTTP://113.121.67.178:9999',
    '3': 'HTTP://121.226.215.198:9999',
    '4': 'HTTP://121.226.214.202:9999',
    '5': 'HTTP://118.212.105.90:9999',
    '6': 'HTTP://35.229.214.209:3128',
    '7': 'HTTP://51.75.147.40:3128',
    '8': 'HTTP://123.169.121.75:9999',
    '9': 'HTTP://170.0.54.253:8080',  # 1.48
    '10': 'HTTP://14.207.176.4:8080',
    '11': 'HTTP://157.7.196.84:1080',  # 1.41
    '12': 'HTTP://62.171.177.80:3129',  # 1.39
    '13': 'HTTP://144.217.101.245:3129',
    '14': 'HTTP://61.145.49.187:9999',
    '15': 'HTTP://113.194.143.235:9999',
    '16': 'HTTP://110.243.21.55:9999',
    '17': 'HTTP://59.38.60.45:9797',
    '18': 'HTTP://115.211.186.220:9999',
    '19': 'HTTP://62.171.177.80:3129',
    '20': 'HTTP://59.38.60.45:9797',
    '21': 'HTTP://59.38.60.45:808',
    '22': 'HTTP://118.212.106.140:9999',
    '23': 'HTTP://35.229.214.209:3128',
}

proxies_1 = {
    '0': 'HTTP://110.228.118.147:8118',
    '1': 'HTTP://123.169.121.75:9999',
    '2': 'HTTP://118.212.105.90:9999',
    '3': 'HTTP://60.246.7.4:8080',
    '4': 'HTTP://157.7.197.239:1080',
    '5': 'HTTP://157.7.199.41:1080',
    '6': 'HTTP://110.243.16.147:9999',
    '7': 'HTTP://138.255.4.4:3128',
    '8': 'HTTP://139.59.251.156:8888',
    '9': 'HTTP://171.35.173.6:9999',
    '10': 'HTTP://113.194.48.85:9999',
    '11': 'HTTP://113.121.95.58:9999',
    '12': 'HTTP://115.221.243.219:9999',
    '13': 'HTTP://115.221.243.219:9999',
    '14': 'HTTP://62.171.177.80:3129',
    '15': 'HTTP://113.194.148.11:9999',
    '16': 'HTTP://142.93.163.56:8080',
    '18': 'HTTP://49.75.59.242:3128',
    '19': 'HTTP://52.56.100.107:3128',
    '20': 'HTTP://139.162.11.25:8888',
    '21': 'HTTP://113.194.48.85:9999',
    '22': 'HTTP://171.35.173.6:9999',
    '23': 'HTTP://209.126.4.134:3128',
    '24': 'HTTP://173.82.240.78:3128',
    '26': 'HTTP://118.212.107.40:9999',
    '27': 'HTTP://144.217.101.245:3129',
    '28': 'HTTP://118.212.105.90:9999',
    '29': 'HTTP://103.153.66.27:8080',
}


def re_quick_proxy():
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "accept-encoding": "gzip, deflate, br",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cache-control": "max-age=0",
        "cookie": '_ga=GA1.2.251401487.1603990360; _s_tentry=-; Apache=6459632317051.233.1608711051353; SINAGLOBAL=6459632317051.233.1608711051353; ULV=1608711051359:1:1:1:6459632317051.233.1608711051353:; login_sid_t=836768862393a5ea300038c4609f06ba; cross_origin_proto=SSL; SSOLoginState=1608712142; SUB=_2A25y5o-BDeRhGeNI71MT9SnOzD2IHXVuKBHJrDV8PUJbkNAfLRemkW1NSGU_PxQfFB3h56wBChPxrQ09qdGb_SO4; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WhSxjHssiu67T8-eFN2ZgH95NHD95QfSoBpeo-NeoMpWs4DqcjGHcL.9-v79Btt; wvr=6; wb_view_log_5641257271=1440*9001.600000023841858; UOR=,,graph.qq.com; webim_unReadCount=%7B%22time%22%3A1608713178777%2C%22dm_pub_total%22%3A0%2C%22chat_group_client%22%3A0%2C%22chat_group_notice%22%3A0%2C%22allcountNum%22%3A37%2C%22msgbox%22%3A0%7D',
        "referer": "https://www.weibo.com/u/5644764907?topnav=1&wvr=6&topsug=1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36",
    }
    start_url = 'https://weibo.com/p/aj/v6/mblog/mbloglist?ajwvr=6&domain=100206&is_all=1&stat_date=202001&page=1&pagebar=0&pl_name=Pl_Official_MyProfileFeed__26&id=1002062028810631&script_uri=/sinapapers&feed_type=0&pre_page=0'
    count = 0
    for i in range(30):
        proxy = get_proxy()
        start = time.time()
        res = requests.get(start_url, headers=headers, proxies={'http': proxy})
        if time.time() - start < 1.5 and res.status_code == 200:
            print("'" + str(count) + "': '" + proxy + "',")
            count = count + 1


def get_proxy():
    try:
        response = requests.get(PROXY_POOL_URL)
        if response.status_code == 200:
            return "HTTP://" + response.text
    except ConnectionError:
        return None


def get_quick_proxy():
    return random.choice(list(proxies_1.values()))


if __name__ == '__main__':
    # url = 'http://httpbin.org/ip'
    # # url = "http://114.212.100.158:98/new/gpa"
    # proxy = get_proxy()
    # print(proxy)
    # try:
    #     response = requests.get(url, proxies={'http': "http://" + proxy})  # 使用代理
    #     # response = requests.get(url)  # 不使用代理
    #     print(response.status_code)
    #     if response.status_code == 200:
    #         print(response.text)
    # except requests.ConnectionError as e:
    #     print(e.args)
    # print(get_quick_proxy())
    # print(get_proxy())
    re_quick_proxy()
