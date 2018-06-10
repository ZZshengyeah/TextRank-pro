# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:35:10 2018

@author: zzshengyeah
"""
from __future__ import division
import sys
import nltk
import networkx as nx
import time
import os
import itertools
from bllipparser import RerankingParser
from collections import  defaultdict
import multiprocessing
rrp = RerankingParser.fetch_and_load('GENIA+PubMed', verbose=True)

def load_stop_words():
    return [ele for ele in nltk.corpus.stopwords.words('english')]

def filter_for_tags(tagged, tags=['NN', 'JJ', 'NNP']):
    """Apply syntactic filters based on POS tags."""
    stop_words=load_stop_words()
    return [item[0] for item in tagged 
    if item[1] in tags and item[0].isalpha() and len(item[0]) > 1 and item[0] not in stop_words]

def Get_Dependency_edge(sentence):
    #获得一个句子中所有依存关系，返回列表
    lis=[]
    Word_set=Get_Word_set(sentence)
    sen=' ' .join(Word_set)
    nbest_list = rrp.parse(sen)
    try:
        tokens = nbest_list[0].ptb_parse.sd_tokens()
    except:
        return lis
    for token in tokens:
        m=list(token)
        if m[6] == 0:
            continue
        try:
            lis.append(tuple([Word_set[m[0]-1],Word_set[m[6]-1]]))
        except:
            continue
    return lis

def get_sen_graph(sentence):
    try:
        lis=Get_Dependency_edge(sentence)
    except:
        distance=6
        return distance
    g=nx.Graph()
    word_set=Get_Word_set(sentence)
    g.add_nodes_from(word_set)
    g.add_edges_from(lis)
    return g

def Get_Word_set(sentence):
    #将句子分词，已经去掉了所有标点符号，返回列表
    word_set=nltk.word_tokenize(sentence)
    while '.' in word_set:
        word_set.remove('.')
    while '!' in word_set:
        word_set.remove('!')
    while '?' in word_set:
        word_set.remove('?')
    while ',' in word_set:
        word_set.remove(',')
    sen=' '.join(word_set)
    Word_set=nltk.word_tokenize(sen)
    return Word_set
    
    
def unique(List):
    l=[]
    for ele in List:
        if ele not in l:
            l.append(ele)
    return l    
    
def Get_sen_NN__JJ_NNP(sen):
    words=nltk.word_tokenize(sen)
    tagged=nltk.pos_tag(words)
    NN_JJ_NNP=filter_for_tags(tagged)
    for ele in NN_JJ_NNP:
        if len(ele) ==1:
            NN_JJ_NNP.remove(ele)
        else:
            continue
    NN_JJ_NNP=unique(NN_JJ_NNP)
    return NN_JJ_NNP

def process_text(filename):
    print("-------------------processing {} -------------------".format(filename))
    start=time.time()
    dic=defaultdict(list)
    f=open('./Input/'+ filename,'r')
    ff=open('./Output/'+ filename+'.depdis','w')
    text=f.read().lower()
    f.close()
    sent_list=nltk.sent_tokenize(text)
    for i,sen in enumerate(sent_list):
        count=0
        NN_JJ_NNP=Get_sen_NN__JJ_NNP(sen)
        if len(NN_JJ_NNP) < 2:
            continue
        else:
            nodePairs=list(itertools.combinations(NN_JJ_NNP,2))
            g=get_sen_graph(sen)
            for pair in nodePairs:
                count=count+1
                try:
                    dis=nx.shortest_path_length(g,source=pair[0],target=pair[1])
                except:
                    dis=6
                if dic[(pair[0],pair[1])] == []:
                    if dic[(pair[1],pair[0])] == []:
                        dic[(pair[0],pair[1])].append(dis)
                    else:
                        dic[(pair[1],pair[0])].append(dis)
        print(i/len(sent_list))
    d = {}
    for ele in dic:
        if dic[ele] != []:
            d[ele] = sum(dic[ele])/len(dic[ele])
        else:
            continue
    ff.write(str(d))  
    end=time.time()
    ff.close()
    print('The single one cost : {}'.format(end-start))


if __name__=="__main__":
    if sys.argv[1] == 'ALL':
        text_list=os.listdir('./Input/')
        start=time.time()
        for filename in text_list:
            process_text(filename)
        end=time.time()
        print('-----------------The ALL cost time : {} ------------------'.format(end-start))
    else:
        filename=sys.argv[1]
        start=time.time()
        process_text(filename)
        end=time.time()
        print('-----------------The singel one cost time : {} ------------------'.format(end-start))















