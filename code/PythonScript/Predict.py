# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:51:20 2021

@author: a
"""

'''
This is the Predict module, which predict the labels on the task we are interested in and 
output the predicted labels to .txt files

To let this module run properly, you have to run DataPreProcess.py 
and Train.py first to ensure that all 
packages and data has been loaded properly.

The module was first deployed on AIStudio platform. To run this module on your local computer, you have to adjust the paths in
the code to the suitable paths on your local computer.
'''

#执行预测任务
#Execute predict task
#import test_data to predict
predict=[predict_data['content_without_name']]
predict=[]
for i in range(0,len(predict_data['content_without_name'])):
    predict.append([predict_data['content_without_name'][i]])

#print('done')

#pred=seq_label_task_1.predict(data=predict_data['content_without_name'])
    
#predict
pred=seq_label_task1.predict(data=predict)



# The output of paddlehub is written in a strange order
# The following codes convert the output labels into readable tags
results=[p.run_results for p in pred]

inv_label_map={0:'O',1:'B-date-reign',2:'I-date-reign',3:'B-date-year',4:'I-date-year',
5:'B-office-voa',6:'I-office-voa',7:'B-office-title',8:'I-office-title',9:'B-place-placename',
10:'I-place-placename'}

tags=[]

for num_batch, batch_results in enumerate(results):
    infers = batch_results[0].reshape([-1]).astype(np.int32).tolist()
    
    #acquire the length of each text in batch #num_batch
    np_lens = batch_results[1]
    
    
    vernier=0

    for index, np_len in enumerate(np_lens):
        
        #labels = infers[index * 400:(index + 1) * 400]
        labels=infers[vernier:vernier+np_len]
        vernier=vernier+np_len

        label_str = []
        count = 0
        for label_val in labels:
            label_str.append(inv_label_map[label_val])
            count += 1
            if count == np_len-1:
                break
        tags.append(label_str)


for i in range(0,len(tags)):
    #print(i)
    tags[i]=tags[i][1:len(tags[i])]
    

text=[]

#Write predicted tags into .txt files

for i in range(0,len(predict)):
    text.append(predict[i][0])

data={'text':text,'tags_predicted':tags}

frame=pd.DataFrame(data)

for i in range(0,len(text)):
    print(i)
    ner_result={'char':list(text[i]),'tag':list(tags[i])}
    frame=pd.DataFrame(ner_result,columns=['char','tag'])
    frame.to_csv('/home/aistudio/jin_ner_0203/'+str(bio_ids[i])+'.txt',sep=" ")

