#coding:utf-8
'''
@author: aitech
@email:
'''
import nltk
import multiprocessing
import os

#载入停用词
def load_stop_words():
    return [ele.encode('utf8') for ele in nltk.corpus.stopwords.words('english')]

#过滤
def filter_words(words):
    stop_words = load_stop_words()
    return [ele for ele in words if ele not in stop_words]

#过滤每个语料文件的停用词
def filter(filename):
    '''
    params:
        filename:训练语料的文件名
    '''
    file_out = "./filter/"+os.path.basename(filename)
    fh_out = open(file_out,"w")
    with open(filename) as fh:
        for line in fh:
            words = line.rstrip().split(" ")
            fh_out.write(" ".join(filter_words(words)))
            fh_out.write("\n")
    fh_out.close()

    #显示进度
    schedule = len(os.listdir("./filter/"))/1122.0*100.0
    print("finish {}".format(schedule))

if __name__=="__main__":
    #多进程过滤
    file_list = os.listdir("./split/")
    filename_list = []
    for ele in file_list:
        filename_list.append("./split/"+ele)
    cpus = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpus)
    pool.map(filter,filename_list)
    pool.close()
    pool.join()


