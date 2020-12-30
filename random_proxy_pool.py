import requests
import random

# 随机ip代理获取
PROXY_POOL_URL = 'http://localhost:5555/random'

proxies = {
    '0': 'HTTP://118.212.105.90:9999',
    '1': 'HTTP://136.243.133.76:3128',
    '2': 'HTTP://113.121.67.178:9999',
    '3': 'HTTP://121.226.215.198:9999',
    '4': 'HTTP://121.226.214.202:9999',
    '5': 'HTTP://118.212.105.90:9999',
    '6': 'HTTP://35.229.214.209:3128',
    '7': 'HTTP://51.75.147.40:3128',
    '8': 'HTTP://123.169.121.75:9999',
    '9': 'HTTP://170.0.54.253:8080', # 1.48
    '10': 'HTTP://14.207.176.4:8080',
    '11': 'HTTP://157.7.196.84:1080', # 1.41
    '12': 'HTTP://62.171.177.80:3129', # 1.39
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


def get_proxy():
    try:
        response = requests.get(PROXY_POOL_URL)
        if response.status_code == 200:
            return "HTTP://" + response.text
    except ConnectionError:
        return None


def get_quick_proxy():
    return random.choice(list(proxies.values()))

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
    print(get_quick_proxy())
    print(get_proxy())
