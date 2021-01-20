import os
import json


def filter_by_num_of_comments(standard=250, sourcePath='word_list/rmrb', targetPath='filter_word_list/rmrb'):
    fileNames = os.listdir(sourcePath)
    for filename in fileNames:
        try:
            targetList = []
            with open(sourcePath + '/' + filename, 'r', encoding='utf-8') as f:
                fileData = json.load(f)

                for news in fileData:
                    if news.get('count') > standard:
                        targetList.append(news)
            with open(targetPath + '/' + filename, 'w', encoding='utf-8') as f:
                json.dump(targetList, f,ensure_ascii=False,indent=4)

        except:
            print("open file error, path = {}".format(filename))

if __name__ == '__main__':
    filter_by_num_of_comments()