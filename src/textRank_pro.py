# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:06:06 2018

@author: zzshengyeah
"""

from __future__ import division
import networkx as nx
import nltk
from gensim.models import Word2Vec
import os
from PorterStemmer import PorterStemmer
from TF_IDF import calculate_tf_idf_score
from collections import defaultdict
#from textblob import TextBlob
import math
p = PorterStemmer()

def load_model():
    '''
    :return: word2vec模型
    '''
    model = Word2Vec.load("../src/word2vec/wiki_model")
    return model


# 释放word2vec模型
def release_model(model):
    del model


model = load_model()


def get_syn_dis(dic_name):
    f = open('../SemEval2010/ALL_dep_dis/' + dic_name)
    dic = eval(f.read())
    return dic


def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    """Apply syntactic filters based on POS tags."""
    return [
        item[0] for item in tagged
        if item[1] in tags and item[0].isalpha() and len(item[0]) > 1
    ]


def word2vec_distance(first, second, model):
    try:
        similarity = model.similarity(first, second)
    except:
        similarity = 0.0001
    distance = 11 - 10 * similarity
    return distance


def unique(List):
    l = []
    for ele in List:
        if ele not in l:
            l.append(ele)
    return l


def build_graph(dic, model):
    """Return a networkx graph instance.

    :param nodes: List of hashables that represent the nodes of a graph.
    """
    gr = nx.Graph()  # initialize an undirected graph
    for ele in dic:

        word2vec_dis = word2vec_distance(ele[0],ele[1], model)
        syntactic_dis = dic[ele]
        weight =1/(1+math.exp(-word2vec_dis)) * (11-syntactic_dis)
        print('Weight :{}'.format(weight))
        gr.add_edge(ele[0], ele[1], weight=weight)
    return gr
'''
def get_textblob_nounphrases(text):
    #获得textblob中的名词短语
    #已经过滤掉含有非法字符和无意义的单词
    #返回列表
    blob=TextBlob(text)
    noun_phrases=blob.noun_phrases
    noun_phrases=unique(noun_phrases)
    for i,noun in enumerate(noun_phrases):
        words=noun.split()
        for j,word in enumerate(words):
            words[j]=p.stem(word,0,len(word)-1)
        noun_phrases[i]=' '.join(words)
    noun_phrases=list(noun_phrases)
    new_noun_phrases=[]
    for ele in noun_phrases:
        l=ele.split()
        goal=1
        for ll in l:
            if ll.isalpha() and len(ll)>1:
                continue
            else:
                goal=0
                break
        if goal == 1:
            new_noun_phrases.append(ele)
    new_noun_phrases=unique(new_noun_phrases)
    return new_noun_phrases
'''


def extract_key_phrases(text_name, text, dic, model):
    word_tokens = nltk.word_tokenize(text)
    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    words_set = filter_for_tags(tagged)
    words_set = unique(words_set)
    textlist = [p.stem(x[0], 0, len(x[0]) - 1) for x in tagged]

    graph = build_graph(dic, model)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph,max_iter=150,weight='weight')
    update_PR={}
    for ele in dic:
        stm1=p.stem(ele[0],0,len(ele[0])-1)
        stm2=p.stem(ele[1],0,len(ele[1])-1)
        try:
            if word2vec_distance(ele[0], ele[1], model) < 7:
                update_PR[stm1] = calculated_page_rank[stm1] + calculated_page_rank[stm2]
            else:
                update_PR[stm1]=calculated_page_rank[stm1]
                update_PR[stm2]=calculated_page_rank[stm2]
        except:
            update_PR[stm1]=calculated_page_rank[stm1]
            update_PR[stm2]=calculated_page_rank[stm2]
    # most important words in ascending order of importance
    keyphrases = sorted(update_PR, key=update_PR.get, reverse=True)
    # the number of keyphrases returned will be relative to the size of the
    # text (a third of the number of vertices)
    keyphrases = keyphrases[:40]
    #当处理文档级数据时，使用tf-idf，效果更好
    #for ele in keyphrases:
    #    update_PR[ele]=update_PR[ele] * calculate_tf_idf_score(ele,text_name.replace('.final','.stm'))   
    # take keyphrases with multiple words into consideration as done in the
    # paper - if two words are adjacent in the text and are selected as
    # keywords, join them together
    modified_key_phrases = {}
    # keeps track of individual keywords that have been joined to form a
    # keyphrase
    i = 0

    # 确定连续的关键词
    update_keyphrases=defaultdict(int)
    def get_continue_pharse(index):
        first = textlist[index]
        if first not in keyphrases or update_keyphrases[first] > 3:
            return index
        else:
            update_keyphrases[first]=update_keyphrases[first]+1
            if index + 1 < len(textlist):
                return get_continue_pharse(index + 1)
            else:
                return index + 1

    while i < len(textlist):
        index = get_continue_pharse(i)
        if index > i:
            continue_pharse = textlist[i:index]
            #continue_pharse = continue_pharse[:4]
            phrase = ' '.join(continue_pharse)
            sum_score = sum(
                [update_PR[ele] for ele in continue_pharse])
            modified_key_phrases[phrase] = sum_score
            i = index
        else:
            i = i + 1
    final = sorted(modified_key_phrases,key=modified_key_phrases.get,reverse=True)
    return final[:15]




def write_file():
    f = open('../SemEval2010/test_answer/shengzhang',
             'w')
    # text_list=os.listdir('../SemEval2010/process_text/')
    text_list = [
        'C-1.txt.final', 'C-3.txt.final', 'C-4.txt.final', 'C-6.txt.final',
        'C-8.txt.final', 'C-9.txt.final', 'C-14.txt.final', 'C-17.txt.final',
        'C-18.txt.final', 'C-19.txt.final', 'C-20.txt.final', 'C-22.txt.final',
        'C-23.txt.final', 'C-27.txt.final', 'C-28.txt.final', 'C-29.txt.final',
        'C-30.txt.final', 'C-31.txt.final', 'C-32.txt.final', 'C-33.txt.final',
        'C-34.txt.final', 'C-36.txt.final', 'C-38.txt.final', 'C-40.txt.final',
        'C-86.txt.final', 'H-2.txt.final', 'H-3.txt.final', 'H-4.txt.final',
        'H-5.txt.final', 'H-7.txt.final', 'H-8.txt.final', 'H-9.txt.final',
        'H-10.txt.final', 'H-11.txt.final', 'H-12.txt.final', 'H-13.txt.final',
        'H-14.txt.final', 'H-16.txt.final', 'H-17.txt.final', 'H-18.txt.final',
        'H-19.txt.final', 'H-20.txt.final', 'H-21.txt.final', 'H-24.txt.final',
        'H-25.txt.final', 'H-26.txt.final', 'H-29.txt.final', 'H-30.txt.final',
        'H-31.txt.final', 'H-32.txt.final', 'I-1.txt.final', 'I-4.txt.final',
        'I-5.txt.final', 'I-6.txt.final', 'I-7.txt.final', 'I-9.txt.final',
        'I-10.txt.final', 'I-11.txt.final', 'I-12.txt.final', 'I-14.txt.final',
        'I-15.txt.final', 'I-16.txt.final', 'I-18.txt.final', 'I-19.txt.final',
        'I-20.txt.final', 'I-21.txt.final', 'I-22.txt.final', 'I-26.txt.final',
        'I-29.txt.final', 'I-30.txt.final', 'I-31.txt.final', 'I-32.txt.final',
        'I-33.txt.final', 'I-34.txt.final', 'I-35.txt.final', 'J-1.txt.final',
        'J-2.txt.final', 'J-3.txt.final', 'J-4.txt.final', 'J-7.txt.final',
        'J-8.txt.final', 'J-9.txt.final', 'J-10.txt.final', 'J-11.txt.final',
        'J-13.txt.final', 'J-14.txt.final', 'J-15.txt.final', 'J-17.txt.final',
        'J-18.txt.final', 'J-20.txt.final', 'J-21.txt.final', 'J-22.txt.final',
        'J-23.txt.final', 'J-25.txt.final', 'J-26.txt.final', 'J-27.txt.final',
        'J-28.txt.final', 'J-30.txt.final', 'J-31.txt.final', 'J-32.txt.final'
    ]

    for i, text_name in enumerate(text_list):
        syn_dis_text = text_name + '.depdis'
        all_syn_dis = get_syn_dis(syn_dis_text)
        ff = open('../SemEval2010/process_text/' + text_name)
        text = ff.read().lower()
        ff.close()
        keyphrases = extract_key_phrases(text_name,text, all_syn_dis, model)
        keyphrases_str = ''
        for j, ele in enumerate(keyphrases):
            if j == len(keyphrases) - 1:
                keyphrases_str = keyphrases_str + ele
            else:
                keyphrases_str = keyphrases_str + ele + ','
        f.write(text_name.replace('.txt.final', ' : ') + keyphrases_str)
        f.write('\n')
        print('-----------finished : {}% ------------- '.format(100 * (i+1) / len(text_list)))
    f.close()


if __name__ == "__main__":
    f=open('../SemEval2010/test_answer/P_R_F_score.txt','a')
    f.write("It's processing...")
    f.close()
    write_file()
    os.system('bash ../SemEval2010/test_answer/get_F_score.sh')
    print("It's done !")
