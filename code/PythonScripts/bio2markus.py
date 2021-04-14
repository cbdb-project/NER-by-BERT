# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:06:45 2020

@author: dongrong
"""

"""
This file reads the tag results of entity recognition and convert them into 
html files with entity highlighted(observable on MARKUS), including regin, year, office & place

txt files in directory 'bio_input' are read
html files output are stored in dictionary "markus_output"

The Python version of this script is Python 3.7.4

Step1:
    The input and output are folders named "bio_input","markus_output" and should be
    put under the same directory with this file
    At the same time, documents in "bio_input" shall be of n.txt form,
    where n is an integer that n>0

Step 2:
    you can run bio2markus_main.py or simply import this package
    
    At the start of the main() function, the module will perform self-checking 
    by calling function bio2markus.self_check(). At this stage, an error will be
    thrown if folder './bio_input' and './markus_output' does not exist
    At the same time, all files in folder './markus_output' will be deleted so
    that new files could be added into. Please make sure you have back-up the files 
    in it if you do not wish to have them deleted
    
    The script reads the txt files and converts the BIO form into xml forms readable
    on Markus

Step3:
    The script outputs the xml files. The output forms will be organized in a 
    document folder named "markus_output", which will also be put under the same directory

"""

import numpy as np
import pandas as pd
import os
import re
from sys import argv


#tags needs to be highlighted in the output xml
B_tag=['B-office-title','B-date-reign','B-date-year','B-place-placename']
I_tag=['I-office-title','I-date-reign','I-date-year','I-place-placename']
#we don't highlight O tags
O_tag=['O']

#in the current task, B-office_voa and I-office_voa are neglected.
neglect=['B-offic-voa','I-office-voa']

#some punctuations are misclassified into I_tags or B_tags(only common puncs are included)
#you are welcomed to add more punctuations into ch_punc according to your need
ch_punc=['，',' ','。','、','！','~','？','；',':']


#map tags to keys in dictionary "mark"
mapping={'B-office-title':'office',
         'B-date-reign':'reign',
         'B-date-year':'year',
         'B-place-placename':'place',
         I_tag[0]:'span',
         I_tag[1]:'span',
         I_tag[2]:'span',
         I_tag[3]:'span'
        }

#the "mark" dictionary contain abbreviations of complex xml 
mark={'span':'</span>',
        'place':'<span class="markup manual unsolved placeName" type="placeName">',
        'reign':'<span class="markup manual unsolved reign" type="reign">',
        'year':'<span class="markup manual unsolved reign_year" type="reign_year">',
        'office':'<span class="markup manual unsolved officialTitle" type="officialTitle">',
        'id':'<span class="markup manual unsolved tempID" type="tempID">',
        'hidchar':r'''<span class="space hiddenChar" contenteditable="false" unselectable="on" onclick="SelectText(event, this);">·'''
        }



#header and rear of the xml file
head=r'''<div class="doc" markupfullname="false" markuppartialname="false" markupnianhao="false" markupofficaltitle="false" markupplacename="false" filename="%s" tag="{&quot;fullName&quot;:{&quot;buttonName&quot;:&quot;&amp;#(22995);&amp;#(21517);&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#d9534f&quot;,&quot;status&quot;:&quot;&quot;},&quot;partialName&quot;:{&quot;buttonName&quot;:&quot;&amp;#(21029);&amp;#(21517);&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#f0ad4e&quot;,&quot;status&quot;:&quot;&quot;},&quot;placeName&quot;:{&quot;buttonName&quot;:&quot;&amp;#(22320);&amp;#(21517);&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#428bca&quot;,&quot;status&quot;:&quot;&quot;},&quot;officialTitle&quot;:{&quot;buttonName&quot;:&quot;&amp;#(23448);&amp;#(21517);&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#5bc0de&quot;,&quot;status&quot;:&quot;&quot;},&quot;timePeriod&quot;:{&quot;buttonName&quot;:&quot;&amp;#(26178);&amp;#(38291);&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;green&quot;,&quot;status&quot;:&quot;&quot;},&quot;reign&quot;:{&quot;color&quot;:&quot;#99ccff&quot;,&quot;buttonName&quot;:&quot;&amp;#(24180);&amp;#(34399);&quot;,&quot;visible&quot;:true,&quot;status&quot;:&quot;&quot;},&quot;reign_year&quot;:{&quot;color&quot;:&quot;#993366&quot;,&quot;buttonName&quot;:&quot;&amp;#(24180);&amp;#(34399);&amp;#(24180);&quot;,&quot;visible&quot;:true,&quot;status&quot;:&quot;&quot;},&quot;tempID&quot;:{&quot;color&quot;:&quot;#800000&quot;,&quot;buttonName&quot;:&quot;ID&quot;,&quot;visible&quot;:true,&quot;status&quot;:&quot;noColor&quot;},&quot;comparativeus&quot;:{&quot;buttonName&quot;:&quot;comparativus&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;green&quot;,&quot;status&quot;:&quot;&quot;},&quot;dilaPerson&quot;:{&quot;buttonName&quot;:&quot;dilaPerson&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#d6616b&quot;,&quot;status&quot;:&quot;&quot;},&quot;dilaPlace&quot;:{&quot;buttonName&quot;:&quot;dilaPlace&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#6b6ecf&quot;,&quot;status&quot;:&quot;&quot;},&quot;koreanPerson&quot;:{&quot;buttonName&quot;:&quot;KPerson&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#b94a48&quot;,&quot;status&quot;:&quot;&quot;},&quot;koreanBook&quot;:{&quot;buttonName&quot;:&quot;KBook&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#428bca&quot;,&quot;status&quot;:&quot;&quot;},&quot;koreanPlace&quot;:{&quot;buttonName&quot;:&quot;KPlace&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#42ca86&quot;,&quot;status&quot;:&quot;&quot;},&quot;koreanOfficialTitle&quot;:{&quot;buttonName&quot;:&quot;KOfficialTitle&quot;,&quot;visible&quot;:true,&quot;color&quot;:&quot;#17becf&quot;,&quot;status&quot;:&quot;&quot;}}"><pre contenteditable="false">'''
rear='''</pre></div>'''


#self-checking module
#开机自检环节
def self_check():
    input_exist=os.path.exists('./bio_input')
    output_exist=os.path.exists('./markus_output')
    
    if not input_exist:
        raise Exception('File Error: folder bio_input not exist. Please set up first')
    
    if not output_exist:
        raise Exception('File Error: floder markus_output not exist. Please set up first')
    
    del_markus=os.listdir('./markus_output')
    for i in range(0,len(del_markus)):
        os.remove('./markus_output/'+del_markus[i])
    
    return 1

#generate list of file names with path
#files genrated have formats of "./bio_input/xxx.txt"
#output: a list of files names
def fileNameGen():
    
    file_name=os.listdir('./bio_input')
    
    for i in range(0,len(file_name)):
        file_name[i]='./bio_input/'+file_name[i]
    
    
    #regular expression, extract numbers from file names(as the index para to textGen())
    numreg=re.compile(r'(\d)+')
    
    #reindex file_name
    #since the order of file_name acquired above is messy
    indes=[]
    for i in range(0,len(file_name)):
        temp=numreg.search(file_name[i])
        indes.append(int(temp.group()))
    
    df=pd.Series(file_name,index=indes)
    file_name=df.reindex(np.sort(indes))
    file_name=list(file_name)
    
    return file_name

#convert BIO tags to xml mark
#bio receives a pandas frame with 2 series: characters in the bio and their tags
#index is the markus id of the bio
def textGen(bio,index):
    #characters and tags of bio
    char=bio['char']
    tag=bio['tag']
    
    #add xml markers to related characters
    for i in range(0,len(tag)):
        #if tag[i] is in B_tag, then let char[i] be "<span...>char[i]"
        if tag[i] in B_tag:
            char[i]=mark[mapping[tag[i]]]+char[i]
        #when tag[i] is B or I but not voa
        if tag[i] in B_tag or tag[i] in I_tag:
            #detect if tag[i] is the end of a certain kind of tag
            if i<(len(tag)-1) and not(tag[i+1] in I_tag):
                char[i]=char[i]+mark['span']
            elif i==len(tag)-1:
                char[i]=char[i]+mark['span']
    
    #group all xml markers together    
    text=list(char)
    text=''.join(text)
    head=''.join([mark['id'],str(index),mark['span'],mark['hidchar'],mark['span']])
    text=head+text
 
    
    return text

#some punctuations are misclassified as B or I tags, correct them
def textWash(bio):
    char=bio['char']
    tag=bio['tag']
    
    for i in range(0,len(tag)):
        if not(tag[i] in O_tag) and char[i] in ch_punc:
            tag[i]=O_tag[0]
            if i<len(tag)-1 and tag[i+1]!='O':
                if tag[i+1] in I_tag:
                    tag[i+1]=B_tag[I_tag.index(tag[i+1])]
                    '''
                if tag[i+1]=='B-office-voa' or tag[i+1]=='I-office-voa':
                    tag[i+1]='B-office-voa'
                else:
                    tag[i+1]=B_tag[I_tag.index(tag[i+1])]
                    '''
    
    return bio



#generate xml files
#as the output, file_num files will be generated, each containing bio_num biographies inside
#input: file_name determines the list you would like to parse
#bio_num indicates how many biographics you would like to include per xml
#serial=1,then markus id are assigned according to serial numer
#serial=0,markus id are assigned according to file name
def xmlGen(file_name,bio_num=50,serial=1):
    #regular expression, extract numbers from file names(as the index para to textGen())
    numreg=re.compile(r'(\d)+')
    #file_count is the # o total txt in ./tag_results
    file_count=0
    #from 0 to 50, indicates the # of texts written into each xml
    bio_count=0
    #the # of xml generated
    xml_count=0
    
    while file_count in range(0,len(file_name)):
        if bio_count==0:
            f=open(''.join(['./markus_output/',str(xml_count),'.html']),'w',encoding='utf-8')
            file_temp=''.join([str(xml_count),'.html'])
            h_temp=head%file_temp
            f.write(h_temp)
            f.write('\n')
        bio=pd.read_table(file_name[file_count],sep=" ")
        print(file_name[file_count])
        bio=textWash(bio)
        if serial==1:
            index=numreg.search(file_name[file_count])
            index=index.group()
        elif serial==0:
            name=file_name[file_count]
            #remove directory and ".txt" and let file name as its markus id
            index=name[14:len(name)-4]
            
        text=textGen(bio,index)
        f.write(text)
        f.write('\n')
        file_count=file_count+1
        bio_count=bio_count+1
        #if there are bio_num texts or there are no more txt waiting to write, close xml
        if bio_count==bio_num or file_count==len(file_name)-1:
            bio_count=0
            f.write(rear)
            f.close()
            xml_count=xml_count+1
            #print(xml_count)
    return 1





#main() function is left for the assessment of potential future maintenance
#you can use the main() down there or you could just import
#main()
'''          
if __name__=='__main__':
    
    # you can use command "run bio2markus.py 50,1" in spyder console
    bio_num,serial=50,1
    
    #self-checking before 
    self_check()
    
    #Step1: read the files stored in './tag_results'
    file_name=fileNameGen()
    
    #Step 2&3: convert bio to xml and store the output in folder './markus'
    xmlGen(file_name,bio_num,serial)
'''    

    
    
    
    
    
            
    
        
    
    


    