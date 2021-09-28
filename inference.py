# inference.py

from models import *
from treedata import *
from utils import *
from collections import Counter
from typing import List

import numpy as np


def decode_bad_tagging_model(model: BadTaggingModel, sentence: List[str]) -> List[str]:
    """
    :param sentence: the sequence of words to tag
    :return: the list of tags, which must match the length of the sentence
    """
    pred_tags = []
    for word in sentence:
        if word in model.words_to_tag_counters:
            pred_tags.append(model.words_to_tag_counters[word].most_common(1)[0][0])
        else:
            pred_tags.append("NN") # unks are often NN
    return labeled_sent_from_words_tags(sentence, pred_tags)


def viterbi_decode(model: HmmTaggingModel, sentence: List[str]) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    #the first index stores the previous value and for the first row it'll store -1
    num_cols=len(model.tag_indexer)-1
    dp=[[[-float('inf'),-float('inf')]for i in range(0,num_cols)] for j in range(0,len(sentence)+1)]
    
    for j in range(0,num_cols):
        #initial values store everything
        val = model.score_init(j)+model.score_emission(sentence,j,0)  
        dp[0][j]=[-1,val]
    
    for i in range(1,len(sentence)):
        for j in range(0,num_cols):
            word_label_score=model.score_emission(sentence,j,i)
            for k in range(0,num_cols):
                #second part of the code represents the transition 
                val=word_label_score+model.score_transition(k,j)+dp[i-1][k][1]
                if dp[i][j][1]<val:
                    dp[i][j]=[k,val]
    #fill in the stop values
    for j in range(0,num_cols):
        val=model.score_transition(j,num_cols)+dp[-2][j][1]
        dp[-1][j]=[j,val] 
     
    #find the max value on the bottom row and go up 
    ret = [0]*len(sentence)
    max_index=0
    for i in range(0,num_cols):
        if dp[-1][max_index][1]<dp[-1][i][1]:
            max_index=i
    index=dp[-1][max_index][0]
    counter=-1
    while index!=-1:
        ret[counter]=model.tag_indexer.get_object(index)
        counter-=1
        index=dp[counter][index][0]
    return labeled_sent_from_words_tags(sentence, ret)


def beam_decode(model: HmmTaggingModel, sentence: List[str], beam_size: int) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :param beam_size: the beam size to use
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    raise Exception("Implement me")
