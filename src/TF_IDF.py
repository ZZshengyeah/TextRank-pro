# -*- coding: utf-8 -*-
'''
 * @Author: ZZshengYeah 
 * @Date: 2018-05-30 23:15:50 
 * @Last Modified by: ZZshengYeah
 * @Last Modified time: 2018-05-31 02:17:01
'''

from __future__ import division
from collections import Counter
import os
import nltk
import math 

theme_text_num={"C":25,"H":25,"I":25,"J":25}

def get_pure_words(text):
    '''
    只返回文本中的单词，不保留标点符号、数字等等
    '''
    words=nltk.word_tokenize(text)
    Words=[]
    for ele in words:
        if ele.isalpha():
            Words.append(ele)
        else:
            continue
    return Words


def calculate_tf_idf_score(word,in_text_name):
    f_in_text=open('../SemEval2010/process_text/Stm_final/' + in_text_name,'r')
    in_text=f_in_text.read().lower()
    words=get_pure_words(in_text)
    count=Counter(words)
    tf = count[word] / len(words)
    text_name=os.listdir('../SemEval2010/process_text/Stm_final/')
    for ele in text_name:
        num=0
        if ele.startswith('Stm'):
            continue
        elif ele[0] != in_text[0]:
            continue
        else:
            f=open('../SemEval2010/process_text/Stm_final/' + ele,'r')
            text=f.read().lower()
            if word in text: 
                num=num+1
            else:
                continue
    idf=math.log(theme_text_num[in_text_name[0]] / (num + 1))
    tf_idf_score=tf*idf
    return tf_idf_score

if __name__ == '__main__':
    word='servic'
    in_text_name="C-1.txt.stm"
    tf_idf_score=calculate_tf_idf_score(word,in_text_name)
    print(tf_idf_score)

