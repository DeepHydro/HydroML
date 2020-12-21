# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:28:04 2019

@author: a7446
"""


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import collections


############## global variable ################
data_type = ['netrad','precip','temperature','wind']
t = data_type[0]
data = collections.defaultdict(list)
con = np.zeros((1,1103))



############## load  data  ###################
for i in range(27):
    filename = r'.\%s\%d.xlsx'%(t,i+1990)
    rawdata=pd.read_excel(filename).values
    d = np.transpose(rawdata)
    con=np.concatenate((con,d))
    
    
    
############   PCA   #########################
x = con[1:,:]
pca=PCA(n_components=10)
pca.fit(x)
res=pca.fit_transform(x)
p = pca.explained_variance_ratio_



##########     write file     ################
write_file=r'.\example_pca.xlsx'  
workbook=xlsxwriter.Workbook(write_file)
worksheet=workbook.add_worksheet('sheet1')
head=[k+1 for k in range(len(res[0]))]
worksheet.write_row('A1',head)
for j in range(len(res)):
    A='A%d' %(j+2)
    worksheet.write_row(A,res[j])
workbook.close()    
out_file_name=r'.\example_pca.txt' 
with open(out_file_name,'w') as f:
    for num in pca.explained_variance_ratio_:
        f.write(str(num)+'\n')


