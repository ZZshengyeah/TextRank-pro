# -*- coding: utf-8 -*-
"""
Created on Thu May 24 08:06:06 2018

@author: zzshengyeah
"""
from __future__ import division
import sys
import networkx as nx
import nltk
from gensim.models import Word2Vec
import os
from PorterStemmer import PorterStemmer
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
    f = open('./Output/' + dic_name)
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
        firstString = ele[0]
        secondString = ele[1]
        word2vec_dis = word2vec_distance(firstString, secondString, model)
        syntactic_dis = dic[ele]
        weight =1/(1+math.exp(-word2vec_dis)) * (11-syntactic_dis)
        gr.add_edge(firstString, secondString, weight=weight)
    return gr



def extract_key_phrases(text_name, text, dic, model):
    word_tokens = nltk.word_tokenize(text)
    # assign POS tags to the words in the text
    tagged = nltk.pos_tag(word_tokens)
    words_set = filter_for_tags(tagged)
    words_set = unique(words_set)
    textlist = [x[0] for x in tagged]

    graph = build_graph(dic, model)

    # pageRank - initial value of 1.0, error tolerance of 0,0001,
    calculated_page_rank = nx.pagerank(graph,max_iter=150,weight='weight')
    update_PR={}
    for ele in dic:
        try:
            if word2vec_distance(ele[0], ele[1], model) < 7:
                update_PR[ele[0]] = calculated_page_rank[ele[0]] + calculated_page_rank[ele[1]]
            else:
                update_PR[ele[0]]=calculated_page_rank[ele[0]]
                update_PR[ele[1]]=calculated_page_rank[ele[1]]
        except:
            update_PR[ele[0]]=calculated_page_rank[ele[0]]
            update_PR[ele[1]]=calculated_page_rank[ele[1]]
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

if __name__ == "__main__":
    textname=sys.argv[1]
    dic_name=textname+'.depdis'
    dic=get_syn_dis(dic_name)
    f=open('./Input/'+textname,'r')
    text=f.read()
    f.close()
    keyphrases=extract_key_phrases(textname,text,dic,model)
    ff=open('./Output/out_'+textname,'w')
    ff.write('\n'.join(keyphrases))
    ff.close()
    print("It's done !")
