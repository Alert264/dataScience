import os
import json


def filter_by_num_of_comments(standard=250, sourcePath='word_list/rmrb', targetPath='filter_word_list/rmrb'):
    fileNames = os.listdir(sourcePath)
    targetList = []
    frontline = 0
    lastline = 50
    count = 0

    for filename in fileNames:
        try:

            with open(sourcePath + '/' + filename, 'r', encoding='utf-8') as f:
                fileData = json.load(f)

                for news in fileData:
                    if news.get('count') <= standard:
                        count+=1
                        if count >=frontline and count <lastline:
                            targetList.append(news)


        except:
            print("open file error, path = {}".format(filename))
    with open(targetPath + '/' + 'front.json', 'w', encoding='utf-8') as f:
            json.dump(targetList, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    filter_by_num_of_comments(550,'split_words/xlxw','filter_word_list/xlxw')