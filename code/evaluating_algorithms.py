'''
ALBA MALAGÓN MÁRQUEZ
Trabajo Final de Máster: Análisis predictivo de la supervivencia del cáncer de mama mediante datos clínicos y genétcos: un enfoque basado en aprendizaje automático

Script que crea e implementa varios algoritmos de aprendizaje automático para seleccionar el mejor para predecir la supervivencia del cáncer de mama.

USO: python3 evaluating_algorithms.py
'''


from libraries_variables import *
from functions import *

starttime = timeit.default_timer()
pd.set_option('display.max_columns', None)



#----------------------------------       STEP 1       ----------------------------------      
#                                    LOADING THE DATA
#----------------------------------------------------------------------------------------- 


# cargando el conjunto de datos
raw_df = load_data(taules_path)


#----------------------------------         STEP 2        ----------------------------------      
#                                     DATA PREPROCESSING
#------------------------------------------------------------------------------------------- 


# preprocesando los datos
raw_df = data_preprocessing(raw_df, drop_features)




#----------------------------------         STEP 3        ----------------------------------      
#                                     ENCODING FEATURES
#------------------------------------------------------------------------------------------- 

# coodificando los datos
datasetcoded, y = codification(raw_df)



#----------------------------------         STEP 4        ----------------------------------      
#                    FEATURE SELECTION: REMOVING LOW IMPORTANCES FEATURES
#------------------------------------------------------------------------------------------- 

# selección de características de cero y de baja importancia
feature_importances, ops = identify_zero_importance(datasetcoded, y, ops, n_iterations=1)
feature_importances, ops = identify_low_importance(cumulative_importance, feature_importances, ops)

datasetcoded=datasetcoded.drop(ops['low_importance'], axis=1)


f = open(f"{global_path}/data/tmp/dict_{dataset}_{years}.txt","w")
f.write( str(ops) )
f.close()

feature_importances.to_csv(f'{global_path}/data/tmp/feature_importances_{dataset}_{years}.csv', header=True, sep="\t")

# guardamos todas las columnas del conjunto de entrenamiento
text_file = open(f'{global_path}/data/tmp/allCOLUMNStraining_{dataset}_{years}.txt', 'w') 
text_file.write(str(datasetcoded.columns.values))
text_file.close()


#----------------------------------         STEP 5        ----------------------------------      
#                                    DATA NORMALIZATION
#------------------------------------------------------------------------------------------- 

# normalizamos los datos en aquellos algoritmos que lo requieren
if algorithm == 'nn' or algorithm == 'lr' or algorithm == 'gb':
    datasetcoded = to_standard(datasetcoded)

#----------------------------------         STEP 6        ----------------------------------      
#                             SPLITTING THE DATA (training only)
#------------------------------------------------------------------------------------------- 


# separamos datos de entrenamiento y de prueba
X_train,X_test,y_train,y_test = data_split(datasetcoded,y,data_aug)




#----------------------------------         STEP 7        ----------------------------------      
#                                    SUMMARIZING THE DATA
#------------------------------------------------------------------------------------------- 

# resúmenes de los datos
color_print(f"\n### {dataset_name} DATASET ###", color='cyan')
summarizing(X_train,X_test,y_train,y_test,datasetcoded,col='cyan')




#----------------------------------         STEP 8        ----------------------------------      
#                                  TUNNING HYPERPARAMETERS 
#------------------------------------------------------------------------------------------- 


def create_model(optimizer, init):
	# función para crear un modelo para NN
    model = Sequential()
    model.add(Dense(100, input_dim=X_test.shape[1], activation='relu',kernel_initializer=init)) 
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))
    # compilación
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   
    return model

if algorithm == 'nn':
    default_model = KerasClassifier(build_fn=create_model)


# BÚSQUEDA ALEATORIA para buscar los mejores hiperparámetros
# usando validación cruzada 3 veces, buscando entre 100 combinaciones diferentes, y usando todos los núcleos disponibles
# comentamos esta parte porque ya conocemos los resultados


'''
random = RandomizedSearchCV(estimator = default_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
random.fit(X_train,y_train)
# We can view the best parameters from fitting the random search:
print(f'RANDOM FIT BEST PARAMETERS using {algorithm}:')
print(random.best_params_)
'''

# GRID SERACH con validación cruzada
# Creación de la cuadrícula de parámetros basada en los resultados de la búsqueda aleatoria
# comentamos esta parte porque ya conocemos los resultados

'''
grid_search = GridSearchCV(estimator = default_model, param_grid = param_grid, cv = 2, n_jobs = -1, verbose = 2)
grid_search.fit(X_train,y_train)
print(f'GRID SEARCH BEST PARAMETERS using {algorithm}:')
print(grid_search.best_params_)

'''







#----------------------------------         STEP 9        ----------------------------------      
#                                   EVALUATING ALGORITHMS
#-------------------------------------------------------------------------------------------- 



# el ajuste de parámetros ya se ha realizado, por lo que podemos ir directamente a evaluar los algoritmos

if algorithm == 'nn': # considerando ANN
    NND = create_model(optimizer='adam', init='he_uniform')
    start_algo_train_NND=timeit.default_timer()
    NND.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=55)
    stop_algo_train_NND=timeit.default_timer()
    color_print('Tiempo invertido entrenando el algoritmo NND:  {}'.format(stop_algo_train_NND-start_algo_train_NND), color='white')

    start_algo_train_NNR=timeit.default_timer()
    NNR = create_model(optimizer=o, init=i)
    NNR.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=e, batch_size=b)
    stop_algo_train_NNR=timeit.default_timer()
    color_print('Tiempo invertido entrenando el algoritmo NNR:  {}'.format(stop_algo_train_NNR-start_algo_train_NNR), color='white')
    
    start_algo_train_NNG=timeit.default_timer()
    NNG = create_model(optimizer=og, init=ig)
    NNG.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=eg, batch_size=bg)
    stop_algo_train_NNG=timeit.default_timer()
    color_print('Tiempo invertido entrenando el algoritmo NNG:  {}'.format(stop_algo_train_NNG-start_algo_train_NNG), color='white')

    default_model=NND
    classifiers=[NND, NNR, NNG] 
    

for i in range(len(classifiers)):
    output, outputWRONG, classifer_fitted = evaluating_algorithm_training(pretty_names[i], classifiers[i], X_train, y_train, X_test, y_test, names[i], col='red')
    output.to_csv(f'{global_path}/data/resultats/output_realvspred_{dataset}_{years}_{names[i]}_{data_aug}.csv', header=True, sep="\t")
    outputWRONG.to_csv(f'{global_path}/data/resultats/output_realvspred_WRONG_{dataset}_{years}_{names[i]}_{data_aug}.csv', header=True, sep="\t")








stoptime = timeit.default_timer()
color_print('\nTime:', color='red')
print(stoptime - starttime)



