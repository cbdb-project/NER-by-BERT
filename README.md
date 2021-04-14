# NER-by-BERT

Named Entity Recognition(NER) on biographical data is of researchers' interest in the CBDB project. It is vital that we could develop a program to automatically recognize the desired entities in the texts(for instance, date time, reign name, etc). In this repository, a BERT model is trained to perform the NER mission on biographics in dynasty Jin. 

## File Description

In this section, I will briefly describe what each files is used for in this repository.

The Data file contain all data we are interested in. Unpacking the **result** data, we will acquire the training data from Ming. **Jin_data** provide samples of Jin biographies that are of our interest. **BERT_Samples** file contain tagged samples performed by BERT.

The Code file contain our training code (saved as JinNER.ipynb Notebook). For code in PythonScripts, we can find Python modules that convert .txt tagged results to xml files that are readable by Markus.

As for Paddlehub_Guide file, You could find an uncompleted Paddlehub tutorial written in Chinese. However, an English version of Paddlehub Tutorial could be found in this readme.md, so researchers could just ignore this file.

## Model Description

To enhance the accuracy and effectiveness of the model, BERT is adopted to perform the task. BERT is a NLP framework published by Google in 2008, aiming at training an embedding model using bidirectional transformers. Although BERT was only capable of processing English texts at first, a Chinese version has later been published by researchers. The model (in tensorflow, pytorch) could be found in: https://github.com/ymcui/Chinese-BERT-wwm

In our practice, we adopt a slight model, with 12 layers, 768 hidden nodes and 12 heads. The model is specified as 'chinese_wwm_ex_L-12_H-768_A-12'.

## Model Detail

The goal of our task is to predict labels in Jin Dynasty using data from Ming Dynasty. Hence, we will first train our model on Jin data, with reigns and office titles coming from Ming replaced by reigns and office titles coming from Jin respectively. After training, we deploy the model onto Jin data, tag the entities and hand them to experts for further supervision.

## Implementation Details

There are a few points I would like emphasize on concerning with the implementation detail:

1. The train-and-predict code was stored in .ipynb files. As for .py files, those are python scripts that transform tagged .txt files into xml file matching the format requirement of Markus. (An online assessment system)
2. The code was written in a rather... non-mainstream manner. To be specific, the model was written based on package Paddlehub, an deep learning framework developed by BAIDU. The Paddlehub code is deployed on AIStudio platform (https://aistudio.baidu.com). The reason for choosing AIStudio and Paddlehub is that powerful GPUs are accessible on AIStudio. Colab GPUs provided by Google are rather slow on BERT training and is are not stable in China mainland.
3. Unfortunately, AIStudio does not support Tensorflow and Pytorch, which is **extremely annoying**. So basically Paddlehub is a trade-off between convenience and high-performance computing resource.
4. For potential researchers, I suggest 3 ways to deal with this problem: 1. Use Paddlehub and AIStudio as the working platform. However, the English version of AIStudio might not be that user-friendly to non-Chinese-speaking researchers. 2. If you have high-performance computing resource (such as a cluster), I suggest you pip install Paddlehub on the cluster. 3. It is also suggested that you can refactor the model using Tensorflow/Pytorch. However, this might be a bit time consuming.

## A Practical Guide to Training and Deployment

### Get Accessed to AIStudio

As we have mentioned before, our model is deployed via Paddlehub supported by AIStudio developed by BAIDU, where much high-performance computing resource is avaliable. **If your model is not deployed via AIStudio, you can skip this subsection**.

#### Language Setting and Log in

First, you shall visit https://aistudio.baidu.com, which is the official website of AIStudio. At the top-right corner of the website, researchers could switch the language to English by clicking the EN icon. Then you could sign up and log in via your Github account.

#### Obtain Free Computing Resource

After you've signed up for AIStudio, you can then finish the questionnaire on https://aistudio.baidu.com/aistudio/questionnaire?activityid=457 to obtain the right to use high-performance GPUs for 12 hours. Technically you may skip the annoying parts collecting your personal information (telephone, address, etc). They won't verify your personal information so you could simply fill in fake ones (to avoid junk mail/call).

After you've signed up, every time when you log in AIStudio, you will obtain a 10-hour right to use high-performance GPUs, which is quite enough for our project.

#### Start a New Project!

Now you can start a new project. Click your avatar icon at the top-right corner, enter 'personal center' - 'projects' - 'my project' - blue 'create' icon in the right- 'Notebook'. Then you've created a new Jupyter Notebook for your new project, and everything follows the rule of a Jupyter Notebook.

### NER Using Paddlehub on AIStudio

Here in this section I will explain how the codes in our model work. The code and instructions below could also be found in the .ipynb file.

#### Environment Setting

The working environment of AIStudio consists of two parts. The left part of the environment is the file directory, where we could upload local files to the cloud. The right part of the environment is simply the classical Jupyter Notebook working environment.

To start training, we shall first install Paddlehub to Jupyter with a right version:

```py
# Install version must be 1.8.2
!pip install paddlehub==1.8.2 
import paddlehub as hub
```

After we've installed the Paddlehub, we load the pre-trained model:

```py
model=hub.Module(name="bert_chinese_L-12_H-768_A-12")
```

Technically speaking, we can load different pre-trained models. If you are interested in other pre-trained models, you can look for them in the Github page of Paddlehub. Researchers with Chinese background may also refer to:  https://aistudio.baidu.com/aistudio/projectdetail/147009 

#### Construct the Dataset

This is the most difficult part for the Paddlehub model. The key point is that how we could load our training (and validation, test) data to Paddlehub. In Paddlehub, the data is required to be encapsulated in a **DemoDataset** object, which we have to write on our own. At the same time, data readable by Paddlehub should be of format .tsv, which is not a very common format. Now I will introduce the way to safe our data as .tsv files.

First, we have to slightly modify the raw data. As for NLP tests, we have to separate Chinese characters using '\002' mark. Hence, for each piece of biography, we have to use .join() function to insert the '\002' mark between all characters to make the text readable by Paddlehub.

Second, we convert our data to DataFrame objects in Pandas. After we've done so, by employing the following code we could save them as .tsv files:

```py
t.to_csv('/home/aistudio/data/train.tsv',sep='\t',columns= ['text_a','label'],encoding='utf_8_sig',index=None) 
v.to_csv('/home/aistudio/data/validate.tsv',sep='\t',columns= ['text_a','label'],encoding='utf_8_sig',index=None) 
testing.to_csv('/home/aistudio/data/testing.tsv',sep='\t',columns= ['text_a','label'],encoding='utf_8_sig',index=None) 
ming_data.to_csv('/home/aistudio/data/dataset.tsv',sep='\t',columns= ['text_a','label'],encoding='utf_8_sig',index=None)
```

Here **t**, **v**, **testing** and **ming_data** are DataFrame objects.

After we've saved the data as .tsv files, we now define the **DemoDataset** class and initialize a **DemoDataset** object:

```py
from paddlehub.dataset.base_nlp_dataset import BaseNLPDataset 
 
 
class DemoDataset(BaseNLPDataset):
"""DemoDataset"""     
def __init__(self):         
self.dataset_dir = "/home/aistudio/data/"
super(DemoDataset, self).__init__(         
base_path=self.dataset_dir,         
train_file="train.tsv",         
dev_file="validate.tsv",         
test_file="testing.tsv",         
# If there is any predict set       
predict_file="predict.tsv", 
train_file_with_header=True,         
dev_file_with_header=True,         
test_file_with_header=True,         
predict_file_with_header=True,         
label_list=["O",     "B-date-reign", "I-date-reign",     "B-date-year", "I-date-year",     "B-office-voa", "I-office-voa",     "B-office-title", "I-office-title",     "B-place-placename", "I-place-placename"]) 
#Encapsulate the dataset into a DemoDataset object
dataset = DemoDataset() 
```

#### Define the training task

After we've loaded the dataset, we can now begin training. First, we shall define various configurations of the task:

```py
#Construct reader to preprocess the data
reader = hub.reader.SequenceLabelReader(
        dataset=task1_dataset,
        vocab_path=model.get_vocab_path(),
        max_seq_len=512)

#Strategy pecifies the fine-tune parameters like learning rate, etc
strategy=hub.AdamWeightDecayStrategy(
    weight_decay=0.01,
    learning_rate=1e-4,
    warmup_proportion=0.1)

# config specifies some hyperparameters
config=hub.RunConfig(
    # If you are using a GPU, set use_cuda to be True
    use_cuda=True,
    # Epoch
    num_epoch=80,
    # The training log will be saved into a file with the name specified by checkpoint_dir
    checkpoint_dir="chinese_wwm_base_seq_label_demo",
    # Batch_size
    batch_size=16,
    # For how many iterations the model will evaluate the statistics (F1, recall, etc)
    eval_interval=50,
    strategy=strategy)

#Construct fine tune task. 
#max_seq_len shall always be larger than the maximum of sequence length in train, validate, test and predict
inputs, outputs, program = model.context(
    trainable=True, max_seq_len=512)

sequence_output=outputs["sequence_output"]

feed_list=[
    inputs["input_ids"].name,
    inputs["position_ids"].name,
    inputs["segment_ids"].name,
    inputs["input_mask"].name
]

#Please be sure that the version of Paddlehub is 1.8.2, or there will be an error（Error: parameter has no attribute...)
seq_label_task1 = hub.SequenceLabelTask(
    data_reader=reader,
    feature=sequence_output,
    feed_list=feed_list,
    max_seq_len=512,
    num_classes=task1_dataset.num_labels,
    config=config,
    # Add conditional random field
    add_crf=True)

```

Now, we execute the fine-tune task. By calling code below, Paddlehub will perform fine-tune based on the BERT model with all hyperparameters defined above.

```py
# Execute finetune task
task1_rate=seq_label_task1.finetune_and_eval()
```

#### Prediction

Hurrey! We've successfully trained the model! The next mission is to tag the data in Jin, which is the prediction task. This section will demonstrate how to do prediction using Paddlehub.

##### Preprocess the Predict Dataset

Paddlehub provides APIs to directly predict the data. However, data passed to the predict API should be in the list format, with each of its element another list containing the biography texts. To be specific, the data to be predict should be in the format like this:

```py
to_predict=[['曾任宣差總管鷹房打捕東勝等處渡河船隻河道所。'],
['河內人，天會十四年參與修北村湯王廟。'],
['孟津人，進義校尉，大定四年任河南路同知兼知陜州。']]
```
So if the data to predict is kept as a DataFrame object in the program, you shall use the following code to transform it into the format above:

```py
predict=[]
for i in range(0,len(predict_data['content_without_name'])):
    predict.append([predict_data['content_without_name'][i]])
```

##### Make Prediction

And then predict:

```py
pred=seq_label_task1.predict(data=predict)
```

##### Convert the Predictions into Readable Tags

The next thing to do is to label the texts with the predict result. However, if you call the **pred** object, you will find that it is a list consists of addresses in the memory. For each address, if you call 'pred[i].run_results', you still get an integer instead of a specific tag. Actually, the integers involved corresponds to the tags (Like B-Place-Name, O, I-Reign, etc). So we have to convert these integers into tags to make them readable. This is not a very easy task.

We first acquire the results in integer form:

```py
# The output of paddlehub is written in a strange order
# The following codes convert the output labels into readable tags

# Acquire results that are shown as integers
results=[p.run_results for p in pred]

# The inverse label map that identifies which tag each integer shall correspond to
# The reason why we know the mapping rule is that it follows the order of the tags in label_map
# dictionary mentioned above
inv_label_map={0:'O',1:'B-date-reign',2:'I-date-reign',3:'B-date-year',4:'I-date-year',
5:'B-office-voa',6:'I-office-voa',7:'B-office-title',8:'I-office-title',9:'B-place-placename',
10:'I-place-placename'}
```

If we call the **results** object, we will find that it is in a strange format. For each element in **results**, it consists of two arrays. The first array is a list of integers varying from 0-10 and the second array is a lists of positive integers. Let me explain why this will happen.

The predict results are returned in batches. Each element in the **results** list is a batch of predictions. The first array in a elements are tagged results for all characters in this batch encoded in integers, while the second array encodes the number of biographies in each batch (the length of the 2nd array) and the length of each biography (the integers in the 2nd array). Hence, to reconstruct the results, we split the first array in each batch according to the lengths encoded in the 2nd array and use the **inv_label_map** to transform them into tags:

```py
# Initialize the tags
tags=[]

# Transform integers to tags
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


# the first position of each prediction is a marker, throw away
for i in range(0,len(tags)):
    #print(i)
    tags[i]=tags[i][1:len(tags[i])]
```

To make it looks COOOOOOOOOL, we save the tags into .txt files along with the biography texts:

```py
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
```

Now you will see a list of .txt files, with each of them storing a piece of biography along with the tagged results.

#### Diagnostics

It is vital for researchers to present statistics of the model such as F1, recall, etc to analyze their models quantitavely and such procedure is called model diagnostics. Recall that when we were defining the training task, we declared a parameter called **checkpoint_dir** using the file name "chinese_wwm_base_seq_label_demo". When the fine-tune is over, you will find a file in your working directory with this name. Enter this file, then enter the subfile visualization, in which you will find a .log file. We can analyze our results using the .log file.

To perform model diagnostic, we have to adopt another assisting package named visualdl. We can simply install it using pip.

```py
!pip install visualdl
```

I suggest we do the diagnostics on our local computer. Hence, installing visualdl via cmd is recommended.

After installing visualdl, create a local file named 'log', with a subfile named 'scalar_test', with a subsubfile named 'train', and download the .log file to the train file.

Now, rerun cmd and type in the following command to run the diagnostic (WARNING: Remember is the Cmd not the Python environment!)

cd into the path where the log file (not the .log file) locates, and

```
visualdl --logdir ./log --port 8080
```

This command creates a local portal, where diagnostics can be applied. We can visit 'http://localhost:8080/' to check the diagnostic results. Basically F1, recall, loss, accuracy, etc are all accessable. 

**Remark**: I suugest visit 'http://localhost:8080/' by Chrome. Using Firefox may lead to various bugs.

### Conclusion

This will be the end of the guide. Should you find any part of this tutorial confusing, please do not hesitate to connect me via 1155151972@link.cuhk.edu.hk.




## Additional Notice

None for now.
