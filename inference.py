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
    dp=[[[0,0]]*len(sentence)]*len(model.tag_indexer)
    #max lambda compare the second element
    for i in range(0,len(sentence)):
        for j in range(0,len(model.tag_indexer)):
            if i==0:
                #using initial values store everything
                val = model.score_init(j)*model.score_emission(sentence,j,i)  
                dp[i][j]=[-1,val]
            else:
                #want to check every previous tag for the index
                for k in range(0,len(model.tag_indexer)):
                    val=model.score_init(j)*model.score_emission(sentence,j,i)*model.score_transition(k,j)*dp[i-1][k]
                    if dp[i][j][1]<val:
                        dp[i][j]=[k,val]
    #find the max value on the bottom row and go up 
    ret = [0]*len(sentence)
    max_index=0
    for i in range(0,len(model.tag_indexer)):
        if dp[-1][i][2]<dp[-1][max_index][2]:
            max_index=i
    ret[-1]=max_index
    index=dp[-1][max_index][1]
    counter=-2
    while index!=-1:
        ret[counter]=index
        index=dp[counter][index][1]
        counter-=1
    return ret


def beam_decode(model: HmmTaggingModel, sentence: List[str], beam_size: int) -> LabeledSentence:
    """
    :param model: the HmmTaggingModel to use (wraps initial, emission, and transition scores)
    :param sentence: the words to tag
    :param beam_size: the beam size to use
    :return: a LabeledSentence containing the model's predictions. See BadTaggingModel for an example.
    """
    raise Exception("Implement me")
