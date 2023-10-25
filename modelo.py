# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

#import library
import pandas as pds
from sklearn.neural_network import MLPRegressor
import numpy as npy
import matplotlib.pyplot as mpt
import seaborn as sbn
import openpyxl
#Data Entry 
Model_NT_Logical_Disk__202104281441 = 'Path'
New_Model_NT_Logical_Disk__202104281441 = pds.read_excel(Model_NT_Logical_Disk__202104281441, sheet_name='Sheet0')
New_Model_NT_Logical_Disk__202104281441 = pds.read_excel(Model_NT_Logical_Disk__202104281441, sheet_name=None)
New_Model_NT_Logical_Disk__202104281441.keys()
Concat_New_Model_NT_Logical_Disk__202104281441 = pds.concat(New_Model_NT_Logical_Disk__202104281441,ignore_index=True)
Concat_New_Model_NT_Logical_Disk__202104281441
def diary(Used):
    diary = Used - 0 
    return diary
Concat_New_Model_NT_Logical_Disk__202104281441['RANGE_DATE'] = Concat_New_Model_NT_Logical_Disk__202104281441['%_Used'].apply(diary)
Concat_New_Model_NT_Logical_Disk__202104281441_filter=Concat_New_Model_NT_Logical_Disk__202104281441[Concat_New_Model_NT_Logical_Disk__202104281441['Server_Name']=='bry_wbddsmp:NT']
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part=Concat_New_Model_NT_Logical_Disk__202104281441_filter[Concat_New_Model_NT_Logical_Disk__202104281441_filter['Disk_Name']=='C:']
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part.set_index('STANDARD_TIMESTAMP')
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part.iloc[ : , [0,2,3,6,8]]
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part['STANDARD_TIMESTAMP'] = pds.to_datetime(Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part['STANDARD_TIMESTAMP'])
Concat_New_Model_NT_Logical_Disk__202104281441_Disk_Part.resample('M', on='STANDARD_TIMESTAMP').mean()
#neural training
x = Concat_New_Model_NT_Logical_Disk__202104281441["%_Used"]
y = Concat_New_Model_NT_Logical_Disk__202104281441["RANGE_DATE"]
x=x.to_numpy()
X = x[:,npy.newaxis]
from sklearn.model_selection import train_test_split
X_train,  X_test,  y_train, y_test =  train_test_split(X, y)
r = MLPRegressor(hidden_layer_sizes=(4,6), activation='relu',solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant',
learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=1, tol=0.0001, verbose=False,
warm_start = False, momentum = 0.9, nesterovs_momentum = True, early_stopping = False, validation_fraction = 0.1, beta_1 = 0.9, beta_2 = 0.999,
epsilon = 1e-05, n_iter_no_change = 10, max_fun = 20000)
r.fit(X_train,y_train)
r.score(X_train,y_train)
x.shape
prediction=38.519596
r.predict([[prediction]])
Concat_New_Model_NT_Logical_Disk__202104281441.to_csv('Path', sep='\t')
