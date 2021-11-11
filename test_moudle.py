# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:14:41 2021

@author: Administrator
"""
import tensorflow as tf

import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding
import os
from bert4keras.models import build_transformer_model
os.environ["RECOMPUTE"] = '1'
from bert4keras.tokenizers import Tokenizer
base_path='D:/chinese_wobert_plus_L-12_H-768_A-12/uncased_L-12_H-768_A-12/'
dict_path = base_path+'vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True)
config_path = base_path+'bert_config.json'
checkpoint_path = base_path+'bert_model.ckpt'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

batch_size=10
dropout_rate=0.4
keras.losses
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
def load_data(f):
    batch_token_ids, batch_segment_ids = [], []  
    for t in f:
       t[1]=t[1].replace('_',' ')
       t[0]=t[0].replace('_',' ')
       question_word,question_segment=tokenizer.encode(t[0],t[1])
       batch_token_ids.append(question_word)
       batch_segment_ids.append(question_segment)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_token_ids[batch_token_ids==tokenizer._token_unk_id]=tokenizer._token_mask_id
    batch_segment_ids = sequence_padding(batch_segment_ids)
    return [batch_token_ids, batch_segment_ids]
def process(t):
    if t=='is_aged':
        return "'s age is"
    if t=='is_in_country':
        return " is in "
    if t=='is_in_country_inverse':
        return  " have a football club that  is "
    if t=='plays_for_country':
        return  " is playing for the country that  is  "
    if t=='plays_for_country_inverse':
        return " have a citizen that is a football palyer and his name is " 
    if t=='plays_in_club':
        return " plays in the club  "
    if t== 'plays_in_club_inverse':
        return " club employ a football player that's name is  "
    if t== 'plays_position':
        return "'s position of team that is "
    if t== 'plays_position_inverse':
        return " is be played by "
    if t== 'wears_number':
        return "'s number of wears is "
    return ""
def get_answer(e,r):
    return e+process(r)+"饕"
def get_model():
    model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='bert',
    return_keras_model=True,
    with_nsp=True,
    )
    
    return model
#获得三元组
def get_base(f):
    kg=pd.read_csv('WC2014.csv').values[:,1:]
    kb={}
    for entitle in set(kg[:,0]):
        kb[entitle]={}
    for [e1,r,e2] in kg:
            kb[e1][r]=e2
    datas=pd.read_csv(f).values[:,1:]
    new_datas=[]
    for t in datas:
        e=t[2].split('#')[0]
        r_true=t[2].split('#')[1]
        data=[]
        rs=[]
        for r in kb[e].keys():
            answer=get_answer(e,r)
            rs.append(r)
            data.append([t[0],answer])
        x=load_data(data)
        new_datas.append([r_true,rs,x])
    return new_datas,kb
def test_main(model,datas,kb):
    true=0
    #开始测试
    al=len(datas)
    from progressbar import ProgressBar
    progress = ProgressBar()
    print("testing:")
    for m in progress(range(len(datas))):
        [r_true,rs,x]=datas[m]
        y=model.predict(x)
        y=np.argmax(y[:,0])
        if rs[y]==r_true:
            true+=1
    return true/al
def main():
    model=get_model()
    model.load_weights('model.h5')
    datas,kb=get_base('test.csv')
    print(test_main(model,datas,kb))