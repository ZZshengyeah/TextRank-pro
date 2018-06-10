# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:16:43 2018

@author: zzshengyeah
"""

import os
lis=os.listdir("/home/zzshengyeah/Downloads/SemEval2010/test")
for i,ele in enumerate(lis):
    process_filename="cat "+ele+"| tr "+'"\n"'+' " "'+" > /home/zzshengyeah/Downloads/SemEval2010/process_test/process/"+ele
    os.system(process_filename)
    print i
    
