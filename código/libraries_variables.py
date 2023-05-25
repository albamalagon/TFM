'''
ALBA MALAGÓN MÁRQUEZ
Trabajo Final de Máster: Análisis predictivo de la supervivencia del cáncer de mama mediante datos clínicos y genéticos: un enfoque basado en aprendizaje automático
'''

# LIBRERÍAS  -------------------------------------------------------------------------------------------

import os
import glob
import timeit
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from lazyme.string import color_print
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from lifelines import *

# RUTAS  -----------------------------------------------------------------------------------------------

global_path = "/Users/alba/Desktop/TFM"
imatges_path = glob.glob(f"{global_path}/data/imatges/*")
taules_path = f"{global_path}/data/taules"

# DEFINICIÓN DE VARIABLES  ------------------------------------------------------------------------------

test_size=0.2
cumulative_importance=0.99

color_print('\n¿Qué conjunto de datos deseas usar? ¿met o nki?', color='magenta')
dataset=input()
while dataset!= 'met' and dataset!= 'nki':
        color_print(f'Has elegido el conjunto {dataset}, pero solo "met" o "nki" son posibles. ¿Cuál de ellos quieres usar?',color='red')
        dataset=input()

color_print('\n¿Cuántos años de supervivencia deseas predecir? ¿5 o 10?', color='magenta')
years=input()
while int(years)!= 5 and int(years)!= 10:
    color_print(f'Has elegido {years} años, pero solo "5" o "10" son posibles. ¿Cuál de ellos quieres predecir?',color='red')
    years=input()
survival_name=f'survival_{years}years'

color_print('\n¿Quieres aplicar técnicas de "data aumentation"? ¿y o n?', color='magenta')
data_aug =input()
while data_aug != 'y' and data_aug != 'n':
    color_print(f'Has elegido {data_aug}, pero solo "y" o "n" son posibles. ¿Quieres aplicar estas técnicas?',color='red')
    data_aug=input()
if data_aug == 'y':
    data_aug=True
elif data_aug == 'n':
    data_aug=False

color_print('\n¿Qué algoritmo quieres aplicar"? ¿dt, rf, nn, lr o gb?', color='magenta')
algorithm = input()
while algorithm != 'dt' and algorithm != 'rf' and algorithm != 'nn' and algorithm != 'lr' and algorithm != 'gb':
    color_print(f'Has elegido el algoritmo {algorithm}, pero solo "dt", "rf", "nn", "lr" o "gb son posibles. ¿Qué algorimo quieres implementar?',color='red')
    algorithm = input()


DTD = DecisionTreeClassifier(random_state=0)
RFD = RandomForestClassifier(random_state=0)
LRD = LogisticRegression(random_state=0)
GBD = GradientBoostingClassifier(random_state=0)
    

if dataset=='met':
    dataset_name='METABRIC'
    sep = '\t'
    table = 'all_metabric_dataset'
    drop_features = ['Patient ID','Study ID','SAMPLE_ID','Sex','Sample Type','Number of Samples Per Patient','STUDY_ID','Relapse Free Status (Months)','Relapse Free Status']
    tiempo_sup='overall_survival_(months)'
    tiempo_pred=int(years)*12
    survival_features=["patient's_vital_status","overall_survival_(months)","overall_survival_status"]
    drop_features_postexploration = ['cancer_type_Breast Cancer','cancer_type_detailed_Breast Angiosarcoma','cancer_type_detailed_Invasive Breast Carcinoma','cancer_type_detailed_Metaplastic Breast Cancer','cancer_type_detailed_Breast Invasive Ductal Carcinoma', 'cancer_type_detailed_Breast Invasive Mixed Mucinous Carcinoma','cancer_type_detailed_Breast','cancer_type_detailed_Breast Mixed Ductal and Lobular Carcinoma','cancer_type_detailed_Breast Invasive Lobular Carcinoma', 'tumor_other_histologic_subtype_Ductal/NST','tumor_other_histologic_subtype_Lobular', 'tumor_other_histologic_subtype_Medullary','tumor_other_histologic_subtype_Metaplastic','tumor_other_histologic_subtype_Mixed','tumor_other_histologic_subtype_Mucinous','tumor_other_histologic_subtype_Other','tumor_other_histologic_subtype_Tubular/ cribriform', 'tumor_other_histologic_subtype_missing','her2_status_measured_by_snp6','er_status_measured_by_ihc','inferred_menopausal_state','type_of_breast_surgery_BREAST CONSERVING','tmb_(nonsynonymous)','pam50_+_claudin-low_subtype_NC','pam50_+_claudin-low_subtype_claudin-low','neoplasm_histologic_grade']
    datasetcoded_name='datasetcoded_met'


    if not data_aug: # no data augmentation
        if int(years) == 5:
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=4, min_samples_split= 15, max_features='sqrt', max_depth=70, criterion='entropy', class_weight= None)                                      
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 3,min_samples_split= 15, max_depth= 60, criterion= 'entropy', max_features= 'sqrt', class_weight= None) 
                
            RFR = RandomForestClassifier(random_state=0,n_estimators = 600, min_samples_split= 15, min_samples_leaf= 4, max_features= 'sqrt', max_depth= None, criterion= 'gini', bootstrap= True,class_weight= 'balanced') 
            RFG = RandomForestClassifier(random_state=0,bootstrap= True, criterion='gini', max_depth= None, max_features='sqrt',min_samples_leaf= 3, min_samples_split= 15, n_estimators= 400, class_weight='balanced')     

            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight=None)   

            GBR = GBG = GradientBoostingClassifier(random_state=0, n_estimators=1400, min_samples_leaf=2, max_features=None, max_depth=40, learning_rate=0.05)

            if algorithm == 'dt':
                param_grid = {'criterion' : ['entropy'],'max_depth': [60, 70, 80],'max_features': ['sqrt'],'min_samples_split': [10,15,20],'min_samples_leaf': [3,4,5], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [True],'criterion' : ['gini'],'max_depth': [None],'max_features': ['sqrt'],'min_samples_leaf': [3,4,5],'min_samples_split': [10,15,20],'n_estimators': [400, 600, 800], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'SGD','random_uniform',50,55
                param_grid = {'optimizer': ['SGD'], 'init':['random_uniform'], 'epochs':[40,50,60,70],'batch_size':[50,55,60]}
                og,ig,eg,bg = 'SGD','random_uniform',60,55
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [1200,1400,1600], 'min_samples_leaf': [2,3], 'max_features': [None],'max_depth': [30,40,50], 'learning_rate': [0.025,0.05]}

        else: # 10 years
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=4, min_samples_split= 10, max_features=None, max_depth=10, criterion='entropy', class_weight= None)                             
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 4,min_samples_split= 15, max_depth= 5, criterion= 'entropy', max_features=None, class_weight= None)
            
            RFR = RandomForestClassifier(random_state=0,n_estimators = 2000, min_samples_split= 15, min_samples_leaf= 4, max_features= None, max_depth= 30, criterion= 'gini', bootstrap= True,class_weight= None) 
            RFG = RandomForestClassifier(random_state=0,bootstrap= True, criterion='gini', max_depth= 20, max_features=None,min_samples_leaf= 4, min_samples_split= 15, n_estimators= 2000, class_weight=None)

            LRR = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight=None)  
            LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight=None)  

            GBR = GBG = GradientBoostingClassifier(random_state=0, n_estimators=400, min_samples_leaf=1, max_features='log2', max_depth=30, learning_rate=0.05) 

            if algorithm == 'dt': 
                param_grid =  {'criterion' : ['entropy'],'max_depth': [5,10,15],'max_features': [None],'min_samples_split': [5,10,15],'min_samples_leaf': [3,4,5], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [True],'criterion' : ['gini'],'max_depth': [20,30],'max_features': [None],'min_samples_leaf': [3,4],'min_samples_split': [10,15],'n_estimators': [1800,2000,2200], 'class_weight': [None]}
            elif algorithm == 'nn':
                o,i,e,b = 'SGD','random_uniform',20,40
                param_grid = {'optimizer': ['SGD'], 'init':['random_uniform'], 'epochs':[15,20,25],'batch_size':[35,40,45]}
                og,ig,eg,bg = 'SGD','random_uniform',25,40
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [200,400,600], 'min_samples_leaf': [1,2], 'max_features': ['log2'],'max_depth': [20,30,40], 'learning_rate': [0.025,0.05]} 

    else: # yes data augmentation
        if int(years) == 5:
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=1,min_samples_split= 2, max_features=None, max_depth=20, criterion='gini', class_weight=None)                                   
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 1,min_samples_split= 3, max_depth= 20, criterion= 'gini', max_features= None, class_weight=None) 
                
            RFR = RandomForestClassifier(random_state=0,n_estimators = 600, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 20, criterion= 'gini', bootstrap= False, class_weight='balanced') 
            RFG = RandomForestClassifier(random_state=0,bootstrap= False, criterion='gini', max_depth= None, max_features='sqrt',min_samples_leaf= 1, min_samples_split= 2, n_estimators= 400, class_weight='balanced')                            

            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight='balanced')  

            GBR = GradientBoostingClassifier(random_state=0, n_estimators=800, min_samples_leaf=1, max_features='sqrt', max_depth=30, learning_rate=0.25)
            GBG = GradientBoostingClassifier(random_state=0, n_estimators=600, min_samples_leaf=1, max_features='sqrt', max_depth=20, learning_rate=0.25)

            if algorithm == 'dt':
                param_grid = {'criterion' : ['gini'],'max_depth': [15,20,25],'max_features':[None],'min_samples_split': [1,2,3],'min_samples_leaf': [1, 2, 3], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [False],'criterion' : ['gini'],'max_depth': [None,20],'max_features': ['sqrt'],'min_samples_leaf': [1,2],'min_samples_split': [2, 3],'n_estimators': [400,600,800],'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'rmsprop','random_uniform',10,55
                param_grid = {'optimizer': ['rmsprop'], 'init':['random_uniform'], 'epochs':[10,15,20],'batch_size':[50,55,60]}
                og,ig,eg,bg = 'SGD','random_uniform',15,50
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [600,800,1000], 'min_samples_leaf': [1,2], 'max_features': ['sqrt'],'max_depth': [20,30,40], 'learning_rate': [0.25,0.35]}
        else: #10 años

            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=2, min_samples_split= 10, max_features='sqrt', max_depth=30, criterion='entropy', class_weight='balanced')
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 2,min_samples_split= 10, max_depth= 20, criterion= 'entropy', max_features= 'sqrt', class_weight= 'balanced')
            
            RFR = RandomForestClassifier(random_state=0,n_estimators = 400, min_samples_split= 5, min_samples_leaf= 2, max_features= None, max_depth= None, criterion= 'gini', bootstrap= True,class_weight= 'balanced')
            RFG = RandomForestClassifier(random_state=0,bootstrap= True, criterion='gini', max_depth= None, max_features=None,min_samples_leaf= 2, min_samples_split= 5, n_estimators= 400, class_weight='balanced')

            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight='balanced')  

            GBR = GradientBoostingClassifier(random_state=0, n_estimators=200, min_samples_leaf=1, max_features='sqrt', max_depth=90, learning_rate=0.05)
            GBG = GradientBoostingClassifier(random_state=0, n_estimators=200, min_samples_leaf=1, max_features='sqrt', max_depth=80, learning_rate=0.05)

            if algorithm == 'dt':
                param_grid = {'criterion' : ['entropy'],'max_depth': [20,30,40],'max_features': ['sqrt'],'min_samples_split': [5,10,15],'min_samples_leaf': [1,2,3], 'class_weight': ['balanced']}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [True],'criterion' : ['gini'],'max_depth': [None],'max_features': [None],'min_samples_leaf': [2,3],'min_samples_split': [4,5],'n_estimators': [200, 400, 600], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'adam','he_uniform',20,55
                param_grid = {'optimizer': ['adam'], 'init':['he_uniform'], 'epochs':[15,20,25],'batch_size':[50,55,60]}
                og,ig,eg,bg = 'adam','he_uniform',15,50
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [100,200,300], 'min_samples_leaf': [1,2], 'max_features': ['sqrt'],'max_depth': [80,90,100], 'learning_rate': [0.025,0.05]} 

if dataset=='nki':
    dataset_name='NKI'
    sep = ','
    table = 'NKI'
    drop_features = ['Patient','ID','barcode','timerecurrence','eventdeath'] 
    tiempo_sup='survival'
    tiempo_pred=int(years)
    survival_features='survival'
    datasetcoded_name='datasetcoded_nki'
    drop_features_postexploration = []

    if not data_aug: # no data aug
        if int(years) == 5:
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=4,min_samples_split= 15, max_features=None, max_depth=100, criterion='gini', class_weight=None) 
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 3,min_samples_split= 15, max_depth= 80, criterion= 'gini', max_features= None, class_weight=None)

            RFR = RandomForestClassifier(random_state=0,n_estimators = 1200, min_samples_split= 2, min_samples_leaf= 1, max_features= None, max_depth= 50, criterion= 'entropy', bootstrap= True) 
            RFG = RandomForestClassifier(random_state=0,bootstrap= True, criterion='entropy', max_depth= None, max_features=None,min_samples_leaf= 1, min_samples_split= 2, n_estimators= 1000, class_weight='balanced') 
            
            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight=None)  

            GBR = GradientBoostingClassifier(random_state=0, n_estimators=1400, min_samples_leaf=2, max_features=None, max_depth=60, learning_rate=0.25)
            GBG = GradientBoostingClassifier(random_state=0, n_estimators=1200, min_samples_leaf=2, max_features=None, max_depth=50, learning_rate=0.25)

            if algorithm == 'dt':
                param_grid = {'criterion' : ['gini'],'max_depth': [80,100,120],'max_features': [None],'min_samples_split': [10,15,20],'min_samples_leaf': [3,4,5], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [True],'criterion' : ['entropy'],'max_depth': [None,50],'max_features': [None],'min_samples_leaf': [1,2],'min_samples_split': [2, 3],'n_estimators': [1000, 1200, 1400], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'SGD','he_uniform',50,55
                param_grid = {'optimizer': ['SGD'], 'init':['he_uniform'], 'epochs':[40,50,60],'batch_size':[50,55,60]}
                og,ig,eg,bg = 'SGD','he_uniform',50,50
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [1200,1400,1600], 'min_samples_leaf': [2,3], 'max_features': [None],'max_depth': [50,60,70], 'learning_rate': [0.25,0.40]}

        else: # 10 years

            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=4, min_samples_split= 2, max_features='sqrt', max_depth=20, criterion='gini', class_weight=None)
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 4,min_samples_split= 2, max_depth= 10, criterion= 'gini', max_features= 'sqrt', class_weight= None)
            
            RFR = RandomForestClassifier(random_state=0,n_estimators = 200, min_samples_split= 10, min_samples_leaf= 2, max_features= 'log2', max_depth= 80, criterion= 'gini', bootstrap= False, class_weight='balanced') 
            RFG = RandomForestClassifier(random_state=0,bootstrap= False, criterion='gini', max_depth= 70, max_features='log2',min_samples_leaf= 2, min_samples_split= 10, n_estimators= 100, class_weight='balanced')

            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight=None)  

            GBR = GBG = GradientBoostingClassifier(random_state=0, n_estimators=200, min_samples_leaf=2, max_features='log2', max_depth=80, learning_rate=0.25) 

            if algorithm == 'dt':
                param_grid = {'criterion' : ['gini'],'max_depth': [10,20,30],'max_features': ['sqrt'],'min_samples_split': [2,3],'min_samples_leaf': [3,4,5], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [False],'criterion' : ['gini'],'max_depth': [70,80,90],'max_features': ['log2'],'min_samples_leaf': [2,3],'min_samples_split': [5,10,15],'n_estimators': [100,200,300], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'adam','random_uniform',10,55
                param_grid = {'optimizer': ['adam'], 'init':['random_uniform'], 'epochs':[10,15,20],'batch_size':[50,55,60]}
                og,ig,eg,bg = 'adam','random_uniform',10,60
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [100,200,300], 'min_samples_leaf': [1,2], 'max_features': ['log2'],'max_depth': [80,90], 'learning_rate': [0.25,0.35]} 


    else: # yes: data augmentation
        if int(years) == 5:
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=2,min_samples_split= 2, max_features=None, max_depth=30, criterion='entropy', class_weight=None)
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 1,min_samples_split= 2, max_depth= 20, criterion= 'entropy', max_features= None, class_weight=None)

            RFR = RandomForestClassifier(random_state=0,n_estimators = 1600, min_samples_split= 5, min_samples_leaf= 1, max_features= 'log2', max_depth= 70, criterion= 'gini', bootstrap= False, class_weight='balanced') 
            RFG = RandomForestClassifier(random_state=0,bootstrap= False, criterion='gini', max_depth= 60, max_features='log2',min_samples_leaf= 1, min_samples_split= 4, n_estimators= 1400, class_weight='balanced') 

            LRR = LRG = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight='balanced')  
 
            GBR = GradientBoostingClassifier(random_state=0, n_estimators=1800, min_samples_leaf=4, max_features=None, max_depth=70, learning_rate=0.5) 
            GBG = GradientBoostingClassifier(random_state=0, n_estimators=1600, min_samples_leaf=4, max_features=None, max_depth=70, learning_rate=0.5) 

            if algorithm == 'dt':
                param_grid = {'criterion' : ['entropy'],'max_depth': [20,30,40],'max_features': [None],'min_samples_split': [1,2,3],'min_samples_leaf': [1, 2, 3], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [False],'criterion' : ['gini'],'max_depth': [60,70,80],'max_features': ['log2'],'min_samples_leaf': [1,2],'min_samples_split': [4,5,6],'n_estimators': [1400, 1600, 1800], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'rmsprop','he_uniform',20,20
                param_grid = {'optimizer': ['rmsprop'], 'init':['he_uniform'], 'epochs':[20,25,30],'batch_size':[15,20,25]}
                og,ig,eg,bg = 'rmsprop','he_uniform',20,20
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [1600,1800], 'min_samples_leaf': [3,4], 'max_features': [None],'max_depth': [70,80], 'learning_rate': [0.5,0.75,1]} 

        else: # 10 años
            DTR = DecisionTreeClassifier(random_state=0,min_samples_leaf=1, min_samples_split= 10, max_features=None, max_depth=40, criterion='entropy', class_weight=None)
            DTG = DecisionTreeClassifier(random_state=0,min_samples_leaf= 1,min_samples_split= 10, max_depth= 30, criterion= 'entropy', max_features= None, class_weight= None)
            
            RFR = RandomForestClassifier(random_state=0,n_estimators = 1600, min_samples_split= 5, min_samples_leaf= 1, max_features= 'log2', max_depth= 70, criterion= 'gini', bootstrap= False) 
            RFG = RandomForestClassifier(random_state=0,bootstrap= False, criterion='gini', max_depth= 60, max_features='log2',min_samples_leaf= 1, min_samples_split= 5, n_estimators= 1600, class_weight='balanced') 
            
            LRR = LRG= LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight='balanced')  

            GBR = GBG = GradientBoostingClassifier(random_state=0, n_estimators=200, min_samples_leaf=4, max_features='log2', max_depth=50, learning_rate=0.05) 
            
            if algorithm == 'dt':
                param_grid = {'criterion' : ['entropy'],'max_depth': [30,40,50],'max_features': [None],'min_samples_split': [5,10,15],'min_samples_leaf': [1,2,3], 'class_weight': [None]}
            elif algorithm == 'rf':
                param_grid = {'bootstrap': [False],'criterion' : ['gini'],'max_depth': [60,70,80],'max_features': ['log2'],'min_samples_leaf': [1,2],'min_samples_split': [5,10],'n_estimators': [1400,1600,1800], 'class_weight': ['balanced']}
            elif algorithm == 'nn':
                o,i,e,b = 'adam','random_uniform',10,20
                param_grid = {'optimizer': ['adam'], 'init':['random_uniform'], 'epochs':[10,15,20],'batch_size':[15,20,25]}
                og,ig,eg,bg = 'adam','random_uniform',10,15
            elif algorithm == 'lr':
                param_grid = {'class_weight': ['balanced',None], 'dual':[False], 'penalty':['none','l2']}
            else: #gb
                param_grid = {'n_estimators': [100,200,300], 'min_samples_leaf': [3,4], 'max_features': ['log2'],'max_depth': [50,60], 'learning_rate': [0.025,0.05]} 

ops={}
np.random.seed(7)

           



# Para usar RandomizedSearchCV, primero necesitamos crear una cuadrícula de parámetros para muestrear 

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)] # Número de árboles 
max_features = ['sqrt', 'log2', None] # Número de características a considerar en cada división
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)] # Número máximo de niveles en cada árbol de decisiones
max_depth.append(None)
min_samples_split = [2, 5, 10, 15] # Número mínimo de muestras necesarias para dividir un nodo
min_samples_leaf = [1, 2, 4] # Número mínimo de muestras requeridas en cada nodo hoja
bootstrap = [True, False] # Método de selección de muestras para el entrenamiento de cada árbol
criterion = ['entropy', 'gini'] # Función para medir la calidad de una división
class_weight=['balanced',None] # Pesos asociados a las clases
optimizers = ['rmsprop','adam','SGD'] # Método para ajustar los parámetros de la red neuronal durante el proceso de entrenamiento
init = ['random_uniform','he_uniform'] # Método para la inicialización de los pesos de la red neuronal antes de comenzar el proceso 
epochs = [10, 20, 50] # Pasada completa de todos los datos de entrenamiento a través de la red neuronal
batches = [20,40,55] # Número de ejemplos de entrenamiento que se utilizan en una iteración para realizar una actualización de los pesos
penalty = ['none','l2'] # Término de regularización a aplicar en el algoritmo de regresión logística
dual = [False] # Forma en que se resuelve el problema de optimización asociado a la regresión logística
learning_rate = [0.05,0.1,0.25,0.5] # Cantidad en la que se actualizan los pesos en cada iteración del algoritmo


if algorithm == 'dt':
    default_model = DTD
    random_grid = {"max_features": max_features,"max_depth": max_depth,'min_samples_split': min_samples_split,"min_samples_leaf": min_samples_leaf,"criterion": criterion,'class_weight': class_weight}
    classifiers = [DTD,DTR,DTG]
    names = ['DTD','DTR','DTG']
    pretty_names = ['\n####        D E C I S I O N    T R E E S        ####','\n####        D E C I S I O N    T R E E S    R A N D O M        ####','\n####        D E C I S I O N    T R E E S    G R I D        ####']

elif algorithm == 'rf':
    default_model = RFD
    random_grid = {'n_estimators': n_estimators,'max_features': max_features,'max_depth': max_depth,'min_samples_split': min_samples_split,'min_samples_leaf': min_samples_leaf,'criterion' : criterion,'bootstrap': bootstrap,'class_weight': class_weight}
    classifiers = [RFD,RFR,RFG]
    names = ['RTD','RTR','RTG']
    pretty_names = ['\n####        R A N D O M     F O R E S T        ####','\n####        R A N D O M     F O R E S T    R A N D O M        ####','\n####        R A N D O M     F O R E S T    G R I D        ####']

elif algorithm == 'nn':
    names = ['NND','NNR','NNG']
    random_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=init)
    pretty_names = ['\n####        A R T I F I C I A L    N E U R A L     N E T W O R K S    ####','\n####        A R T I F I C I A L    N E U R A L     N E T W O R K S     R A N D O M ####','\n####        A R T I F I C I A L    N E U R A L     N E T W O R K S      G R I D   ####']

elif algorithm == 'lr':
    default_model = LRD
    random_grid = {'class_weight': class_weight, 'penalty': penalty, 'dual': dual}
    names = ['LRD','LRR','LRG']
    classifiers = [LRD,LRR,LRG]
    pretty_names = ['\n####        L O G I S T I C     R E G R E S S I O N        ####','\n####        R A N D O M     L O G I S T I C     R E G R E S S I O N        ####','\n####        L O G I S T I C     R E G R E S S I O N    G R I D        ####']


elif algorithm == 'gb':
    default_model = GBD
    random_grid = {'n_estimators': n_estimators, 'learning_rate':learning_rate, 'max_features': max_features,'max_depth': max_depth,"min_samples_leaf": min_samples_leaf}
    names = ['GBD','GBR','GBG']
    classifiers = [GBD,GBR,GBG]
    pretty_names = ['\n####        G R A D I E N T     B O O S T I N G        ####','\n####        R A N D O M     G R A D I E N T     B O O S T I N G        ####','\n####        G R A D I E N T     B O O S T I N G    G R I D        ####']

