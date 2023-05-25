'''
ALBA MALAGÓN MÁRQUEZ
Trabajo Final de Máster: Análisis predictivo de la supervivencia del cáncer de mama mediante datos clínicos y genéticos: un enfoque basado en aprendizaje automático
'''

from libraries_variables import *

def load_data(taules_path):
    '''
    lectura del conjunto de datos
    '''
    color_print(f"\n### GENERANDO CONJUNTO DE DATOS {dataset_name} ###\n", color="cyan")
    raw_df = pd.read_csv(f"{taules_path}/{table}.csv", delimiter =sep,low_memory=False)

    return raw_df


def data_preprocessing(raw_df, drop_features):
    '''
    preprocesamiento de los datos
    '''
    # eliminar variables insignificativas
    raw_df.drop(drop_features,axis=1,inplace=True)
    # substituir valores ausentes por missing
    raw_df.fillna('missing', inplace=True)

    if dataset_name == 'METABRIC':
        # renombración nombres columnas
        raw_df.columns = raw_df.columns.str.replace(' ', '_').str.lower()
        # eliminar pacientes que han muerto por causas externas al cáncer 
        raw_df = raw_df[raw_df["patient's_vital_status"].isin(['Living', 'Died of Disease'])]
        # renombración valores variables
        for str_ in ['er_status','her2_status','pr_status']:
            raw_df[str_] = raw_df[str_].replace(['Positive','Negative'],[1,0])
        for bin_ in ['chemotherapy','hormone_therapy','radio_therapy']:
            raw_df[bin_] = raw_df[bin_].replace(['NO','YES'],[0,1])
        raw_df.er_status_measured_by_ihc = raw_df.er_status_measured_by_ihc.replace(['Positve','Negative','missing'],[1,0,0.5])
        raw_df.primary_tumor_laterality = raw_df.primary_tumor_laterality.replace(['Right','Left','missing'],[1,0,0.5])
        raw_df.cellularity = raw_df.cellularity.replace(['Low','Moderate','High'],[1,2,3])
        raw_df.inferred_menopausal_state = raw_df.inferred_menopausal_state.replace(['Post','Pre'],[1,0])
        raw_df.her2_status_measured_by_snp6 = raw_df.her2_status_measured_by_snp6.replace(['GAIN','NEUTRAL','LOSS','UNDEF'],[1,0,-1,0])  
        raw_df.integrative_cluster = raw_df.integrative_cluster.replace(['4ER+','4ER-'],['4','4'])
        raw_df.overall_survival_status = raw_df.overall_survival_status.replace(['0:LIVING','1:DECEASED'],[0,1])
        for miss_ in ['cellularity','lymph_nodes_examined_positive','mutation_count','tumor_size','neoplasm_histologic_grade']:
            raw_df[miss_] = raw_df[miss_].replace(['missing'],[0])
        for int_ in ['neoplasm_histologic_grade','cellularity','lymph_nodes_examined_positive','integrative_cluster','cohort','primary_tumor_laterality','er_status_measured_by_ihc']:
            raw_df[int_] = raw_df[int_].astype(int)
    '''
    if dataset_name == 'NKI':
        # selección variables clínicas y genéticas
        clinical_df = raw_df.drop(raw_df.columns[11::], axis=1)
        data_gen = raw_df.drop(raw_df.columns[:11:], axis=1)
        # reducción de la dimensionalidad solo de las variables genéticas
        X = data_gen.values 
        pca = PCA(n_components=100)
        # normalizar y transformar los datos
        X_pca = pca.fit_transform(preprocessing.scale(X))
        # definimos el nombre de las nuevas variables: C (de componente) + el número 
        name_col=['C'+str(num) for num in range(0,100)]
        # creamos un conjunto de datos que incluya la información genética reducida junto con los datos clínicos preprocesados
        raw_df = pd.concat([clinical_df,pd.DataFrame(X_pca,columns=name_col)],axis=1)
    '''
    # crear variable objetivo
    raw_df[survival_name] = raw_df[tiempo_sup].apply(lambda x: 0 if x >= tiempo_pred else 1)
    # eliminar variables directamente relacionadas con el objetivo
    raw_df.drop(survival_features,axis=1,inplace=True)

    return raw_df




def codification(dataset):

    # y contiene la variable objetivo
    # X el resto de variables
    y = dataset[survival_name]
    X = dataset.drop(survival_name,axis=1)

    cat_cols = list(X.select_dtypes(include=['object']).columns)
    num_cols = list(X.select_dtypes(include=['int64', 'float64']).columns)
    dfX_cat=X[cat_cols] # conjunto de datos categóricos
    dfX_num=X[num_cols] # conjunto de datos numéricos

    # codificamos el conjunt de datos, usando get_dummies para las variables categóricas
    if dfX_cat.shape[1] == 0:
        datasetcoded = dfX_num
    else:
        datasetcoded = pd.concat((dfX_num,
          pd.get_dummies(dfX_cat, dummy_na=False)),
          axis=1)
        
    # eliminar características insignificantes o correlacionadas identificadas en la exploración
    datasetcoded.drop(drop_features_postexploration, axis=1,inplace=True)
    datasetcoded.to_csv(f'{global_path}/data/taules/{datasetcoded_name}.csv', header=True, sep="\t")

    return datasetcoded,y


def data_split(datasetcoded, y, data_aug):
    '''
    división entrenamiento y test
    '''
    X_train,X_test,y_train,y_test = train_test_split(datasetcoded, y, test_size=test_size, stratify = y, random_state=1)
    # data augmentation
    if data_aug:
        ros = RandomOverSampler(sampling_strategy='minority')
        X_train,y_train = ros.fit_resample(X_train,y_train)

    return X_train,X_test,y_train,y_test



def summarizing(X_train,X_test,y_train,y_test,datasetcoded,col):
    '''
    resúmenes generales de los datos
    '''
    color_print('\nDimensions coded dataset:', color=col)
    print(f"{datasetcoded.shape[0]} Rows and {datasetcoded.shape[1]} Columns")

    color_print('\nDimensions X and Y data:', color=col)
    color_print('X train: {}'.format(X_train.shape), color='white')
    color_print('y train: {}'.format(y_train.shape), color='white')
    color_print('X test: {}'.format(X_test.shape), color='white')
    color_print('y test: {}'.format(y_test.shape), color='white')

    color_print('\nNumber of elements per class in y_train:', color=col)
    print(y_train.value_counts())
    color_print('\nNumber of elements per class in y_test:', color=col)
    print(y_test.value_counts())

def identify_zero_importance(dataset, labels, ops, n_iterations):
        '''
        Función extraída de 'https://github.com/WillKoehrsen/feature-selector' y adaptada a nuestro caso.
        Identificación de las características que tienen importancia 0.
        '''

        feature_names = list(dataset.columns)
        features = np.array(dataset)
        labels = np.array(labels).reshape((-1, ))
        feature_importance_values = np.zeros(len(feature_names))
    
        # se calcula el promedio de diferentes ejecuciones para reducir la varianza
        for _ in range(n_iterations):
            model = RandomForestClassifier(bootstrap= False, criterion='gini', max_depth= None, max_features='auto',min_samples_leaf= 3, min_samples_split= 4, n_estimators= 500, class_weight='balanced')
            model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # orden según la importancia
        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)

        # normalización
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # extracción de las características con importancia 0
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]
        to_drop = list(record_zero_importance['feature'])
        feature_importances = feature_importances
        record_zero_importance = record_zero_importance
        ops['zero_importance'] = to_drop
        
        print('\n%d features with zero importance.\n' % len(ops['zero_importance']))

        # mostramos la lista con las características con importancia 0
        print('The list of zero importance features:')
        print(ops['zero_importance'])
        print()

        return feature_importances, ops


def identify_low_importance(cumulative_importance, feature_importances, ops):
    '''
    Function taken from 'https://github.com/WillKoehrsen/feature-selector' y adaptada a este caso.
    Identificación de las características de menor importancia.
    '''

    cumulative_importance = cumulative_importance
    
    # ordenación de las importancias
    feature_importances = feature_importances.sort_values('cumulative_importance')

    # identificación de las características no necesarias para llegar a la cumulative_importance deseada
    record_low_importance = feature_importances[feature_importances['cumulative_importance'] > cumulative_importance]

    to_drop = list(record_low_importance['feature'])

    record_low_importance = record_low_importance
    ops['low_importance'] = to_drop

    print('%d características son requeridas para la importancia acumulada de %0.2f.' % (len(feature_importances) -
                                                                        len(record_low_importance), cumulative_importance))
    print('%d características no contribuyen en la importancia acumulada de %0.2f.\n' % (len(ops['low_importance']),
                                                                                            cumulative_importance))

    print(ops['low_importance'])

    return feature_importances, ops


def to_standard(df):
    # función para normalizar un conjunto de datos numéricos
    ss = StandardScaler()
    num_df = df[df.select_dtypes(include = np.number).columns.tolist()]
    std = ss.fit_transform(num_df)
    
    return pd.DataFrame(std, index = num_df.index, columns = num_df.columns)

def evaluating_algorithm_training(pretty_name, classifier, X_train, y_train, X_test, y_test, name, col):

    color_print('{}'.format(pretty_name), color=col)
    if (name != 'NND') and (name != 'NNR') and (name != 'NNG'): # considerando todos los algoritmos, excepto NN
        start_algo_train=timeit.default_timer()
        classifier_fitted = classifier.fit(X_train,y_train)
        stop_algo_train=timeit.default_timer()
        start_algo_test=timeit.default_timer()
        # predicción de y test
        y_pred = classifier_fitted.predict(X_test)
        # predicción de y train
        y_pred_train = classifier_fitted.predict(X_train)
        stop_algo_test=timeit.default_timer()

        color_print('\nTiempo invertido entrenando el algoritmo:  {}'.format(stop_algo_train-start_algo_train), color='white')
        color_print('Tiempo invertido prediciendo:  {}\n'.format(stop_algo_test-start_algo_test), color='white')
    
    else: # considerando Artificial Neural Networks
        start_algo_test=timeit.default_timer()
        # predicción de y test
        y_pred = (classifier.predict(X_test) > 0.5).astype("int32")
        # predicción de y train
        y_pred_train = (classifier.predict(X_train) > 0.5).astype("int32")
        stop_algo_test=timeit.default_timer()
        color_print('\nTiempo invertido prediciendo:  {}\n'.format(stop_algo_test-start_algo_test), color='white')
        classifier_fitted = classifier

    target_names_REAL = []
    if 0 in y_test.values:
        target_names_REAL.append('survive')
    if 1 in y_test.values:
        target_names_REAL.append('die')



    # métricas de evaluación
    accu_train = accuracy_score(y_train, y_pred_train)
    accu_test = accuracy_score(y_test, y_pred)
    precision_train = precision_score(y_train, y_pred_train, average='binary')
    precision_test = precision_score(y_test, y_pred, average='binary')
    recall_train = recall_score(y_train, y_pred_train, average='binary')
    recall_test = recall_score(y_test, y_pred, average='binary')
    f1score_train = f1_score(y_train, y_pred_train, average='binary')
    f1score_test = f1_score(y_test, y_pred, average='binary')

    print('\nAccuracy on train: {} %'.format(round(accu_train*100,2)))
    print('Accuracy on test: {} %'.format(round(accu_test*100,2)))

    print('\nPrecision on train: {} %'.format(round(precision_train*100,2)))
    print('Precision on test: {} %'.format(round(precision_test*100,2)))

    print('\nRecall on train: {} %'.format(round(recall_train*100,2)))
    print('Recall on test: {} %'.format(round(recall_test*100,2)))

    print('\nF1score on train: {} %'.format(round(f1score_train*100,2)))
    print('F1score on test: {} %'.format(round(f1score_test*100,2)))



    # classification report
    color_print('\nClassification Report\n', color=col)
    print(classification_report(y_test.astype(str), y_pred.astype(str), target_names=target_names_REAL))

    classes=target_names_REAL

    # matriz de confusión
    cm = confusion_matrix(y_test.astype(str), y_pred.astype(str)) #, labels=classifier.classes_

    color_print('\nConfusion Matrix\n', color=col)
    print(cm)
    print()



    if (name != 'NND') and (name != 'NNR') and (name != 'NNG'): #considering all except NN
        plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Purples, display_labels=target_names_REAL)
        plt.tight_layout()
        plt.savefig(f'{global_path}/data/grafics/plot_confusionmatrix_{dataset}_{years}_{name}_{data_aug}.png')
        plt.close()

        plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Purples, display_labels=target_names_REAL,normalize='true')
        plt.tight_layout()
        plt.savefig(f'{global_path}/data/grafics/plot_confusionmatrix_normalized_{dataset}_{years}_{name}_{data_aug}.png')
        plt.close()

        
    else: # considerando Artificial Neural Networks
        
        disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=target_names_REAL)
        disp.plot(cmap=plt.cm.Purples)
        plt.tight_layout()
        plt.savefig(f'{global_path}/data/grafics/plot_confusionmatrix_{dataset}_{years}_{name}_{data_aug}.png')
        plt.close()
        

    
    # prediciones vs valor real
    output = X_test.copy()
    color_print('\nPREDICTIONS: ', color=col, end='')
    print(len(output))
    output['REAL_LABELS'] = y_test.astype(str)
    output['PREDICTED_LABELS'] = y_pred.astype(str)
    output['REAL_LABELS'] = output.REAL_LABELS.replace(0,'survive')
    output['REAL_LABELS'] = output.REAL_LABELS.replace(1,'die')

    output['PREDICTED_LABELS'] = output.PREDICTED_LABELS.replace(0,'survive')
    output['PREDICTED_LABELS'] = output.PREDICTED_LABELS.replace(1,'die')


    outputWRONG = output.copy()
    outputWRONG = outputWRONG[~(outputWRONG['REAL_LABELS']==outputWRONG['PREDICTED_LABELS'])]
    outputWRONG.reset_index(inplace=True)

    color_print('\nWRONG PREDICTIONS: ', color=col, end='')
    print(len(outputWRONG))


    return output, outputWRONG, classifier_fitted





