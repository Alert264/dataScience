import matplotlib.pyplot as plot
import numpy as np
import json
import os
def divide_data(filesource='source/rmrb',gap=50,length=10000):
    filenames = os.listdir(filesource)
    limit = (int)(length/gap)
    res = []
    left = 0
    count = 0
    front = 0
    frontLimit = 5
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

                    if index < frontLimit:
                        front+=1
                    res[index] = res[index]+1

        except:
            pass


    x=[]
    for element in range(0,limit+1):
        x.append(element * gap)
    print(sum(res))

    plot.plot(x,res)
    plot.show()
    print('more than {}: {}'.format(length,left))

    print("total:{}".format(count))
    print("front {} make up for {:.4f}%".format(frontLimit*gap,100*gap*frontLimit/count))




if __name__ == '__main__':
    divide_data()