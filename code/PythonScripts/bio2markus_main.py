# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:25:46 2020

@author: a
"""

import sys

sys.path.append('./')

import bio2markus


#a sample of main() function
if __name__=='__main__':
    bio_num,serial=50,1
    
    #self-checking before 
    bio2markus.self_check()
    
    #Step1: read the files stored in './tag_results'
    file_name=bio2markus.fileNameGen()
    
    #Step 2&3: convert bio to xml and store the output in folder './markus'
    bio2markus.xmlGen(file_name,bio_num,serial)
