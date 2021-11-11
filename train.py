# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 19:14:41 2021

@author: Administrator
"""
import tensorflow as tf
from test_moudle import test_main,get_base
import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.snippets import sequence_padding, DataGenerator
import os
from bert4keras.models import build_transformer_model
os.environ["RECOMPUTE"] = '1'
from bert4keras.tokenizers import Tokenizer
base_path='D:/chinese_wobert_plus_L-12_H-768_A-12/uncased_L-12_H-768_A-12/'
#BERT-BASE by geogle
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
def load_data(f):
    f=pd.read_csv(f).values[:,1:]
    data=[]
    for t in f:
       t[1]=t[1].replace('_',' ')
       t[0]=t[0].replace('_',' ')
       question_word,question_segment=tokenizer.encode(t[0],t[1])
       data.append([question_word,question_segment,t[-1]])
    return data
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

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=True):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (question_word,question_segment,label) in self.sample(random):
            batch_token_ids.append(question_word)
            batch_segment_ids.append(question_segment)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_token_ids[batch_token_ids==tokenizer._token_unk_id]=tokenizer._token_mask_id
                batch_segment_ids = sequence_padding(batch_segment_ids)
                #batch_labels = sequence_padding(batch_labels)
                #batch_labels=np.expand_dims(batch_labels,1)
                yield [batch_token_ids, batch_segment_ids], np.array(batch_labels)
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []  

model=get_model()
train=load_data('train.csv')
train= data_generator(train, batch_size)
datas,kb=get_base('val.csv')


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    metrics=['accuracy'],
)
epochs=5
minloss=-1
log=[]

for i in range(epochs):
    h=model.fit(
            train.forfit(),
            steps_per_epoch=len(train),
            epochs=1,
        )
    a=test_main(model,datas,kb)
    print("val acc is"+str(a))
    if a>=minloss:
        model.save_weights('model.h5')
model.load_weights('model.h5')        
datas,kb=get_base('test.csv')
a=test_main(model,datas,kb)
print("the result of test is "+str(a))
