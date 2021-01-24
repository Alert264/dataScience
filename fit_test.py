import matplotlib.pyplot as plot
import numpy as np
import json
import os
import pandas
def divide_data(filesource='source/rmrb',gap=20,length=10000):
    filenames = os.listdir(filesource)
    limit = (int)(length/gap)
    res = []
    left = 0
    count = 0
    front = 0
    frontLimit = 1
    for element in range(int(length/gap)+1):
        res.append(0)
    for filepath in filenames:
        try:
            with open(filesource+'/'+filepath,encoding='utf-8') as f:
                jsonData = json.load(f)
                for dic in jsonData:
                    count +=1
                    num = dic.get('评论数量')
                    index = (int)(num/gap)
                    if index >limit:
                        left += 1
                    elif index <= limit:
                        front +=1
                        res[index] = res[index]+1
        except:
            print(filepath)


    frontNum = 0
    for element in range(0,frontLimit):
        frontNum += res[element]
    x=[]
    print(frontNum)
    for element in range(0,limit+1):
        x.append(element * gap)
    print(front)
    dic1 = {'num_gap':x,'num_of_each':res}
    df = pandas.DataFrame(dic1)
    df.to_excel('rmrb.xlsx',index=False)


    plot.plot(x,res)
    plot.show()
    print('more than {}: {}'.format(length,left))

    print("total:{}".format(count))
    print("front {} make up for {:.4f}%".format(frontLimit*gap,frontNum/count*100))




if __name__ == '__main__':
    divide_data('source/xlxw',100)