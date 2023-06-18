'''
ALBA MALAGON MARQUEZ
Trabajo Final de Máster: Análisis predictivo de la supervivencia del cáncer de mama mediante datos clínicos y genéticos: un enfoque basado en aprendizaje automático
'''

from libraries_variables import *
from functions import *

pd.set_option('display.max_columns', None)

raw_df = load_data(taules_path)
raw_df = data_preprocessing(raw_df, drop_features)
datasetcoded, y = codification(raw_df)

feature_importances, ops = identify_zero_importance(datasetcoded, y, ops, n_iterations=1)
feature_importances, ops = identify_low_importance(cumulative_importance, feature_importances, ops)

datasetcoded=datasetcoded.drop(ops['low_importance'], axis=1)

feature_importances.to_csv(f'{global_path}/data/resultados/best/feature_importances_{dataset}_{years}.csv', header=True, sep="\t")

if algorithm == 'nn' or algorithm == 'lr' or algorithm == 'gb':
    datasetcoded = to_standard(datasetcoded)


if int(years)==5:
    model = RandomForestClassifier(random_state=0)
    datasett='output_realvspred_nki_5_RFD_True'
    name='RFD'
else:
    model = LogisticRegression(random_state=0, penalty='l2', dual=False, class_weight='balanced')  
    datasett='output_realvspred_nki_10_LRG_True'
    name='LRG'



# K FOLD CROSS VALIDATION ------------------------------

accu_test_fold,precision_test_fold,recall_test_fold,f1score_test_fold  = folds_bestmodel(datasetcoded, y, model)

print('NKI',years)
for each in [accu_test_fold,precision_test_fold,recall_test_fold,f1score_test_fold]:
    print(f'Mitjana {each}: {np.mean(each)}')
    print(f'Desviació estàndard {each}: {np.std(each)}')
    print()


# FEATURE IMPORTANCES  ------------------------------

# separamos datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test = data_split(datasetcoded,y,data_aug)


classifier_fitted = model.fit(X_train,y_train)
coefficients = model.coef_[0]

feature_importance = pd.DataFrame({'Feature': datasetcoded.columns, 'Importance': np.abs(coefficients)})
feature_importance = feature_importance.sort_values('Importance', ascending=True)

plot_feature_importances(datasetcoded, y, 15, feature_importances)







#    C O M P U T I N G    P R O B A B I L I T I E S   ------------------------------------------------------------



datasetALL = pd.read_csv(f'{global_path}/data/resultados/predicciones/{datasett}.csv', sep="\t")
datasetALL.drop(['Unnamed: 0'], axis=1, inplace=True)
datasetALL['REAL_LABELS']=datasetALL['REAL_LABELS'].replace(0,'survive')
datasetALL['REAL_LABELS']=datasetALL['REAL_LABELS'].replace(1,'die')
datasetALL['PREDICTED_LABELS']=datasetALL['PREDICTED_LABELS'].replace(0,'survive')
datasetALL['PREDICTED_LABELS']=datasetALL['PREDICTED_LABELS'].replace(1,'die')
print()


print()

for each in ['survive','die']:
    color_print(' {}'.format(each.upper()), color='magenta')
    print()
    real = datasetALL[datasetALL['REAL_LABELS']==each]
    predicted = datasetALL[datasetALL['PREDICTED_LABELS']==each]
    predicted_ok = datasetALL[(datasetALL['REAL_LABELS']==each) & (datasetALL['PREDICTED_LABELS']==each)]
    predicted_bad= datasetALL[(datasetALL['REAL_LABELS']!=each) & (datasetALL['PREDICTED_LABELS']==each)]
    
    print('     Real',each.upper(),len(real))
    print('     Predicted as',each.upper(),':',len(predicted))
    print('     Well predicted',each.upper(),':',len(predicted_ok))
    print('     Wrong predicted',each.upper(),':',len(predicted_bad))
    print('     Accuracy:',(round(((len(predicted_ok)/len(real))*100),2)),'%')
    

    print()
    
    real_survive = predicted_bad[(predicted_bad['REAL_LABELS']=='survive')]
    real_die = predicted_bad[(predicted_bad['REAL_LABELS']=='die')]

    if each != 'survive':
        print('     The predicted is',each,'and the real is survive',len(real_survive),'    -   ', (round(((len(real_survive)/len(predicted))*100),2)),'%')
    else:
        print('     The predicted is',each,'and the real is die',len(real_die),'    -   ', (round(((len(real_die)/len(predicted))*100),2)),'%')

    print('     The predicted is ',each,'and the real is',each,len(predicted_ok),'    -   ', (round(((len(predicted_ok)/len(predicted))*100),2)),'%')
    print()


print()






