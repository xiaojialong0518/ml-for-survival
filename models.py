# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:03:21 2021

@author: xiao
"""
#save the models
'''
import os
import joblib
 
dirs = 'C:/code'
if not os.path.exists(dirs):
    os.makedirs(dirs)
    
joblib.dump(cph, dirs+'/cox-en.pkl', compress=1)#max is 9
joblib.dump(cph2, dirs+'/cox.pkl', compress=1)
joblib.dump(estimator, dirs+'/svm.pkl', compress=1)
joblib.dump(rsf, dirs+'/rsf.pkl', compress=9)
'''

#load the models
import os
import joblib
model1 = joblib.load('C:/code/testModel1214/cox-en.pkl')
model2 = joblib.load('C:/code/testModel1214/cox.pkl')
model3 = joblib.load('C:/code/testModel1214/svm.pkl')
model4 = joblib.load('C:/code/testModel1214/rsf.pkl')

#load the data(the data we offer is a template)
#if you don't have the survival outcome, you can ignore the value but leave the viriable name in the sheet
import pandas as pd
from sksurv.datasets import get_x_y
from sklearn.model_selection import train_test_split

coxentest=pd.read_excel(r'C:\code\coxentest.xlsx')
coxtest=pd.read_excel(r'C:\code\coxtest.xlsx')
svmrsftest=pd.read_excel(r'C:\code\svmrsftest.xlsx')

X1, y1 = get_x_y(coxentest,attr_labels=['OS', 'OS_Mon'], pos_label=1)
X2, y2 = get_x_y(coxtest,attr_labels=['OS', 'OS_Mon'], pos_label=1)
X3, y3 = get_x_y(svmrsftest,attr_labels=['OS', 'OS_Mon'], pos_label=1)

#get the risk score
#furture information like how to get Cindex etc. is in https://scikit-survival.readthedocs.io/en/latest/

model1.predict(X1)
model2.predict(X2)
model3.predict(X3)

