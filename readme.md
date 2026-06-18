# Implementación de estrategias de aprendizaje activo en modelos de regresión

En este repositorio está el código desarrollado para el TFG con el título: **"Implementación de una estrategia de aprendizaje activo en modelos de regresión"**.

Las estrategias implementadas son:

- Random Sampling
- Greedy Sampling
- Greedy Sampling optimizado con vector de distancias mínimas
- Incremental K-medoids Sampling

Además del modleo base: baselineRandomForest.py

## Estructura del proyecto

La estructura principal del proyecto es la siguiente:

- En la carpeta `data` se encuentran los datasets utilizados en los experimentos. Los datos están separados en conjunto de entrenamiento y conjunto de test. Durante la ejecución del modelo base también se generan los datasets reducidos.

- En la carpeta `src` se encuentra todo el código del proyecto. En esta carpeta están los ficheros principales para ejecutar el modelo base, las diferentes estrategias de muestreo y funciones auxiliares. 

- En la carpeta `results` se guardan todos los resultados generados por los scripts.

## Dependencias

Para poder ejecutar los métodos y volver a generar los resultados, es necesario disponer de las siguientes librerías de Python:

- numpy
- pandas
- matplotlib
- scikit-learn
- scipy


## Ejecución de los métodos

Para ejecutar los métodos hay que poner el siguiente comando:

`python nombreMetodo.py`

Por ejemplo:
- python baselineRandomForest.py
- python randomSampling.py
- python greedySampling.py
- python greedySampling_minDist.py
- python k_medoidsSampling.py

También se pueden generar las gráficas y tablas ejecutando los scripts:

- python plots/generateComparisonPlots.py
- python plots/generateDatasetPlots.py
- python plots/generateTables.py

Los parámetros principales de ejecución, como los datasets utilizados, el número inicial de muestras, el tamaño del lote, la métrica de distancia o las rutas de salida, se pueden modificar en el fichero `config.py`.

Por defecto los parametros estan configurados como los de la experimentación del TFG:

### Parámetros de muestreo
- INITIAL_SIZE = 50
- BATCH_SIZE = 5
- DISTANCE_METRIC = "euclidean"