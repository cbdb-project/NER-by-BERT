# NER-by-BERT

Named Entity Recognition(NER) on biographical data is of researchers' interest in the CBDB project. It is vital that we could develop a program to automatically recognize the desired entities in the texts(for instance, date time, reign name, etc). In this repository, a BERT model is trained to perform the NER mission on biographics in dynasty Jin. 

To begin with, we train the model with labeled data acquired from Ming dynasty. Then, the model is run on Jin data to perform the NER task. The result shows that the model does well in recognizing reigns and place names while performs relatively poorly on the task of title recognition. 

However, hyperparameters will be adjusted in the future to enhance the performance of the model. More data and codes will be pulled on this repostitory.

Some Details of the model:

Pretrained Model: 'chinese_wwm_ex_L-12_H-768_A-12', Could be found in: https://github.com/ymcui/Chinese-BERT-wwm

Version of Tensorflow: 1.15.2

Version of Kashgari: 1.1.5


在CBDB项目中，研究人员对命名实体识别（NER)具有强烈的兴趣。我们希望可以通过设计程序从古代人物的传记资料中自动挖掘出诸如年号，地名，官职这样的实词。在本仓库中，我们使用BERT模型在金朝人物传记中执行NER任务。

在最开始，我们使用明朝的已标记数据对模型进行训练。接下来，这一模型被用于金朝的传记数据上执行NER任务。结果表明，我们训练的模型在地名与年号上具有优良的标记能力，但是在对官名的识别能力上相对较弱。

然而，在将来，我们将继续调整模型的超参数以增强模型的表现能力。更多的关于模型的数据与代码将被上传至这一仓库中。

一些模型的细节：

预训练模型：'chinese_wwm_ex_L-12_H-768_A-12'，这一模型可以在以下网址中被找到：https://github.com/ymcui/Chinese-BERT-wwm

Tensorflow版本：1.15.2

Kashgari版本：1.1.5
