# NER-by-BERT

Named Entity Recognition(NER) on biographical data is of researchers' interest in the CBDB project. It is vital that we could develop a program to automatically recognize the desired entities in the texts(for instance, date time, reign name, etc). In this repository, a BERT model is trained to perform the NER mission on biographics in dynasty Jin. 


## Model Description

To enhance the accuracy and effectiveness of the model, BERT is adopted to perform the task. BERT is a NLP framework published by Google in 2008, aiming at training an embedding model using bidirectional transformers. Although BERT was only capable of processing English texts at first, a Chinese version has later been published by researchers. The model (in tensorflow, pytorch) could be found in: https://github.com/ymcui/Chinese-BERT-wwm

In our practice, we adopt a slight model, with 12 layers, 768 hidden nodes and 12 heads. The model is specified as 'chinese_wwm_ex_L-12_H-768_A-12'.

## Model Detail

The goal of our task is to predict labels in Jin Dynasty using data from Ming Dynasty. Hence, we will first train our model on Jin data, with reigns and office titles coming from Ming replaced by reigns and office titles coming from Jin respectively. After training, we deploy the model onto Jin data, tag the entities and hand them to experts for further supervision.

## Implementation Detail

There are a few points I would like emphasize on concerning with the implementation detail:

1. The train-and-predict code was stored in .ipynb files. As for .py files, those are python scripts that transform tagged .txt files into xml file matching the format requirement of Markus. (An online assessment system)
2. The code was written in a rather... non-mainstream manner. To be specific, the model was written based on package Paddlehub, an deep learning framework developed by BAIDU. The Paddlehub code is deployed on AIStudio platform (https://aistudio.baidu.com). The reason for choosing AIStudio and Paddlehub is that powerful GPUs are accessible on AIStudio. Colab GPUs provided by Google are rather slow on BERT training.
3. Unfortunately, AIStudio does not support Tensorflow and Pytorch, which is extremely annoying. So basically Paddlehub is a trade-off between convenience and high-performance computing resource.
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

To be continued.





## Additional Notice

Well, the readme.md has not finished yet. Since most of the codes are my own working codes, the model may still be hard to deploy because many of the codes are without comment. Hence, I will:

1. Write the comments as soon as possible.
2. Give a comprehensive introduction on how to deploy the model.
