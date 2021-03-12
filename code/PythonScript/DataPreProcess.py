# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:30:59 2021

@author: a
"""

'''
This is the data preprocessing module, which converts the data into the form that
is readable by the paddlehub frame

The module was first deployed on AIStudio platform. To run this module on your local computer, you have to adjust the paths in
the code to the suitable paths on your local computer.
'''

# install paddlehub in local
#!pip install paddlehub==1.8.2
import paddlehub as hub
import numpy as np
import pandas as pd
import json
import time
import re

#load the base model
model=hub.Module(name="bert_chinese_L-12_H-768_A-12")

# Data Preprocessing

# data in result.json is the labeled ming data
with open('/home/aistudio/result.json') as jf:
    ming=json.load(jf)


#unpack .json file, wash the data and transfer them into ndarray
def dataGen(ming):
    person_ids=np.array(list(ming.keys()))
    person_ids.sort()

    #idNum=len(person_ids)

    x_data=[]
    y_data=[]

    indexer=0
    for person_id in person_ids:
        
        char_tag=ming[person_id]['char_tag']
        x_data.append([])
        y_data.append([])
        omit_len=len(person_id)
        for i in range(omit_len+1,len(char_tag)):
            x_data[indexer].append(char_tag[i][0])
            y_data[indexer].append(char_tag[i][1])
         
        indexer=indexer+1

    for i in range(0,len(y_data)):
      for j in range(0,len(y_data[i])):
        old_text=y_data[i][j]
        #convert labels like 'B_date_reign' into 'B-date-reign', which is readable by kashgari
        new_text=old_text.replace("_","-")
        y_data[i][j]=new_text


    return x_data,y_data,person_ids


#construct train,validate and test set
#train_set_rate indicates the proportion of trainning data
#validate_set_rate indicates the proportion of validation data
def splitTrain(x_data,y_data,person_ids,train_set_rate,validate_set_rate):
    x_data=np.array(x_data)
    y_data=np.array(y_data)
    
    temp=np.array([x_data,y_data])
    temp=temp.T
    
    ming_data=pd.DataFrame(temp,index=person_ids,columns=['text_a','label'])

    #test on converting data 2 tsv
    for i in range(0,len(ming_data['text_a'])):
        #print(i)
        ming_data['text_a'][i]='\002'.join(ming_data['text_a'][i])
        ming_data['label'][i]='\002'.join(ming_data['label'][i])
        ##ming_data['text_a'][i]+='\002'
        ##ming_data['label'][i]+='\002'
        #ming_data['text_a'][i]=str(ming_data['text_a'][i])
        #ming_data['label'][i]=str(ming_data['label'][i])
        
        

    np.random.seed(int(time.time()))
    ming_data=ming_data.sample(frac=1.0)
    
    idNum=len(person_ids)
    train_size=int(np.floor(idNum*train_set_rate))
    validate_size=int(np.floor(idNum*validate_set_rate))

    train_set=ming_data[0:train_size]

    validate_set=ming_data[train_size:train_size+validate_size]
    test_set=ming_data[train_size+validate_size:idNum]
    return train_set,validate_set,test_set,ming_data


#Build the first model: 50% of ming as train and 20% of ming as test
x_data,y_data,person_ids=dataGen(ming)
t,v,testing,ming_data=splitTrain(x_data,y_data,person_ids,0.75,0.25)

#generate x,y of train,validate and test
x_train=np.array(t['text_a'])
y_train=np.array(t['label'])
x_validate=np.array(v['text_a'])
y_validate=np.array(v['label'])

x_test=np.array(testing['text_a'])
y_test=np.array(testing['label'])

t.to_csv('train.csv')
v.to_csv('validate.csv')

train_len=[]
for i in range(0,len(x_train)):
  train_len.append(len(x_train[i]))

max_len=max(train_len)


tag_list = tag_list = ["O",
        "B_date_reign", "I_date_reign",
        "B_date_year", "I_date_year",
        "B_office_voa", "I_office_voa",
        "B_office_title", "I_office_title",
        "B_place_placename", "I_place_placename"]

# Part of Task1: save the data into tsv
t.to_csv('/home/aistudio/train.tsv',sep='\t',columns=['text_a','label'],encoding='utf_8_sig',index=None)
v.to_csv('/home/aistudio/validate.tsv',sep='\t',columns=['text_a','label'],encoding='utf_8_sig',index=None)
testing.to_csv('/home/aistudio/testing.tsv',sep='\t',columns=['text_a','label'],encoding='utf_8_sig',index=None)
ming_data.to_csv('/home/aistudio/data/dataset.tsv',sep='\t',columns=['text_a','label'],encoding='utf_8_sig',index=None)

predict_data=pd.read_table('/home/aistudio/test_data.txt')
text=predict_data['content_without_name']
text.to_csv('/home/aistudio/predict.tsv',encoding='utf_8_sig')

# Load the data to predict
bio=predict_data['content_without_name']
bio_ids=predict_data['id']

