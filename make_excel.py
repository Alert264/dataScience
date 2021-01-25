import pandas
import os
import numpy as np
import json

def make_excel_for_pic(file_path='predicted/rmrb'):
    filepaths = os.listdir(file_path)
    filepaths.sort(key=lambda x: int(x.split('-')[2][1]),reverse=False)
    filepaths.sort(key=lambda x: int(x.split('-')[2][0]),reverse=False)
    filepaths.sort(key=lambda x: int(x.split('-')[1][-1]), reverse=False)
    filepaths.sort(key=lambda x: int(x.split('-')[1][0]), reverse=False)
    filepaths.sort(key=lambda x: int(x.split('-')[0]), reverse=False)
    ManYi = []
    ZiHao = []
    PingJing = []
    GaoXing = []
    FenNu = []
    GanDong = []
    QiWang = []
    Date = []
    for file in filepaths:
        string = file.split('.'[0]).pop(0)
        Date.append(string)
        with open(file_path + '/' + file,'r',encoding='utf-8') as f:
            fileData = {}
            pc = 0
            man_yi = 0
            zi_hao = 0
            ping_jing = 0
            gao_xing = 0
            fen_nu = 0
            gan_dong = 0
            qi_wang = 0

            fileData = json.load(f)
            filedata = fileData.get('前20个关键词')
            for element in filedata:
                label = element.get('label')
                if label == '满意':
                    man_yi +=1
                elif label == '自豪':
                    zi_hao +=1
                elif label == '平静':
                    ping_jing +=1
                elif label == '高兴':
                    gao_xing += 1
                elif label == '愤怒':
                    fen_nu += 1
                elif label == '感动':
                    gan_dong += 1
                elif label == '期望':
                    qi_wang += 1
        QiWang.append(qi_wang)
        GanDong.append(gan_dong)
        GaoXing.append(gao_xing)
        ManYi.append(man_yi)
        ZiHao.append(zi_hao)
        PingJing.append(ping_jing)
        FenNu.append(fen_nu)
    dic1 = {'日期':Date,'期望':QiWang,'感动':GanDong,'高兴':GaoXing,'满意':ManYi,'自豪':ZiHao,'平静':PingJing,'愤怒':FenNu}
    frame = pandas.DataFrame(dic1)
    frame.to_excel('predict_xlxw.xlsx',index=False)

if __name__ == '__main__':
    make_excel_for_pic('predicted/xlxw')