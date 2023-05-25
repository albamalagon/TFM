# Análisis predictivo de la supervivencia del cáncer de mama mediante datos clínicos y genéticos: un enfoque basado en aprendizaje automático

## Trabajo Final de Máster - Alba Malagón Márquez


_De forma general, el objetivo de este trabajo es identificar qué factores clínicos y genéticos influyen en la supervivencia de pacientes con cáncer de mama y en qué medida, así como predecir dicha supervivencia, para proporcionar información útil para el desarrollo de tratamientos personalizados y la mejora del pronóstico de los pacientes con esta enfermedad._

_Objetivos principales:_

- _Identificar los factores clínicos y genéticos que contribuyen a la supervivencia de los pacientes con cáncer de mama, y en qué medida._

- _Predecir la supervivencia en pacientes con cáncer de mama utilizando información clínica y genética, mediante un modelo de aprendizaje automático._

### Ficheros

- [Conjunto de datos](https://github.com/albamalagon/TFM/tree/main/conjunto%20de%20datos)
  - [elección conjuntos de datos.html](https://github.com/albamalagon/TFM/blob/main/conjunto%20de%20datos/conjuntos%20de%20datos.html): Explicación de los conjuntos de datos seleccionados. Ambos están formados por un seguido de características clínicas y un conjunto de características genéticas, juntamente con una variable que informa sobre el tiempo de supervivencia del paciente y otra que informa sobre su estado vital.
  - [all_metabric_dataset.csv](https://github.com/albamalagon/TFM/blob/main/conjunto%20de%20datos/all_metabric_dataset.csv): Conjunto de datos que se encuentra disponible en el repositorio [_CBioPortal for Cancer Genomics_](https://www.cbioportal.org/).
  - [NKI](https://github.com/albamalagon/TFM/blob/main/conjunto%20de%20datos/NKI.csv): Conjunto de datos descargados del sitio web [Data World](https://data.world/deviramanan2016/nki-breast-cancer-data).
  - [Otros](https://github.com/albamalagon/TFM/tree/main/conjunto%20de%20datos/otros): Contiene otros conjuntos de datos, como los listados extraídos de las bases de datos [_Uniprot_](https://www.uniprot.org/) y [_Ensembl_](https://useast.ensembl.org/index.html), y el conjunto de datos NKI con dimensionalidad reducida.

- [Análisis](https://github.com/albamalagon/TFM/tree/main/ana%CC%81lisis)

  - [analisis_exploratorio_MET.html](https://github.com/albamalagon/TFM/blob/main/ana%CC%81lisis/analisis_exploratorio_MET.html): Exploración de la distribución de las características y su relación con la supervivencia, así como la identificación de patrones y relaciones entre las variables, mediante pruebas estadísticas o técnicas de selección de variables. Aplicado al conjunto de datos METABRIC.
  - [analisis_exploratorio_NKI.html](https://github.com/albamalagon/TFM/blob/main/ana%CC%81lisis/analisis_exploratorio_NKI.html): Exploración de la distribución de las características y su relación con la supervivencia, así como la identificación de patrones y relaciones entre las variables, mediante pruebas estadísticas o técnicas de selección de variables. Aplicado al conjunto de datos NKI.
  - [analisis_supervivencia_MET.html](https://github.com/albamalagon/TFM/blob/main/ana%CC%81lisis/analisis_supervivencia_MET.html): Implementación de técnicas estadísticas enfocadas a analizar la probabilidad de supervivencia de los pacientes después del diagnóstico. Aplicado al conjunto de datos METABRIC.
  - [analisis_supervivencia_NKI.html](https://github.com/albamalagon/TFM/blob/main/ana%CC%81lisis/analisis_supervivencia_NKI.html): Implementación de técnicas estadísticas enfocadas a analizar la probabilidad de supervivencia de los pacientes después del diagnóstico. Aplicado al conjunto de datos NKI.

- [Código](https://github.com/albamalagon/TFM/tree/main/c%C3%B3digo):

- [Resultados]():





### Uso

Todos los _scripts_ se ejecutan de la misma manera:
```
python3 filename.py
```
