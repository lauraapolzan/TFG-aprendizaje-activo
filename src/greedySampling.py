import time
import numpy as np
import pandas as pd
import os
from sklearn.metrics import pairwise_distances
from samplingUtils import load_data, save_plot, train_and_evaluate, initialize_sampling
from config import DATASETS, DISTANCE_METRIC, METRICS_TO_PLOT, GS_PATH, INITIAL_SIZE, BATCH_SIZE


def select_k_greedySample(X_train: np.ndarray, x_indices: np.ndarray, z_indices: np.ndarray, k: int, M: np.ndarray, 
                          metric: str, matrizVacia: bool, new_idx: int | None) -> tuple[np.ndarray, np.ndarray, int | None]:
    """
    Selecciona una muestra S de tamaño k mediante el algoritmo Greedy.

    En cada paso se selecciona el punto de Z que maximiza su distancia mínima
    respecto a los puntos ya seleccionados en X U S.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        x_indices: índices de las muestras ya seleccionadas.
        z_indices: índices de las muestras no seleccionadas.
        k: número de muestras a seleccionar.
        M: matriz de distancias.
        metric: métrica a utilizar para calcular las distancias.
        matrizVacia: indicador de si la matriz de distancias está vacía.
        new_idx: último índice seleccionado en la iteración anterior.

    Devuelve:    
        S: índices de las nuevas muestras seleccionadas.
        z_indices: índices restantes no seleccionados.
        new_idx: último índice seleccionado.
    """
    S = []
    k_actual = min(k, len(z_indices))
    
    # Cada punto seleccionado se añade aquí para que el siguiente paso lo tenga en cuenta.
    current_x_indices = np.copy(x_indices)

    for _ in range(k_actual):

        if matrizVacia:
            # Calcular la distancia entre los puntos de X y los puntos de Z
            for x in current_x_indices:
                for z in z_indices:
                    M[x, z] = pairwise_distances(X_train[[x]],X_train[[z]],metric=metric)[0][0]
                    M[z, x] = M[x, z]
            matrizVacia = False

        else:
            # Calcular las distancias entre el nuevo punto seleccionado y los puntos restantes de Z
            for z in z_indices:
                if np.isnan(M[new_idx, z]):
                    M[new_idx, z] = pairwise_distances( X_train[[new_idx]], X_train[[z]], metric=metric)[0][0]
                    M[z, new_idx] = M[new_idx, z]

        # Para cada z ∈ Z, obtener su distancia mínima a los puntos seleccionados
        max_pos = None
        max_min_dist = -1.0

        for z_pos in range(len(z_indices)):
            z_actual = z_indices[z_pos]
            min_dist = float("inf")

            for x in current_x_indices:
                if M[x, z_actual] < min_dist:
                    min_dist = M[x, z_actual]

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                max_pos = z_pos

        # Añadir z a S y eliminarlo de Z
        new_idx = z_indices[max_pos]
        S.append(new_idx)

        # Actualizar la copia auxiliar para que el siguiente punto tenga en cuenta el punto elegido
        current_x_indices = np.append(current_x_indices, new_idx)

        z_indices = np.delete(z_indices, max_pos)

    return np.array(S, dtype=int), z_indices, new_idx


def greedy_sampling(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n: int, s: int, 
                    k: int, num_iter: int, metric: str, return_history: bool = False) ->  pd.DataFrame | tuple[pd.DataFrame, list[np.ndarray]]:
    """
    Implementa el algoritmo de greddy sampling, seleccionando de forma iterativa las muestras.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        y_train: etiquetas del conjunto de entrenamiento.
        X_test: conjunto de datos de prueba.
        y_test: etiquetas del conjunto de prueba.
        n: tamaño total de los datos.
        s: número de muestras iniciales a seleccionar.
        k: Número de muestras a seleccionar en cada iteración.
        num_iter: Número de iteraciones.
        metric: Métrica a utilizar para calcular las distancias.
        return_history: indica si se devuelve también el historial de índices seleccionados.

    Devuelve:
        DataFrame con los resultados obtenidos en cada iteración.
        Si return_history=True, también devuelve el historial de muestras seleccionadas para poder hacer comparaciones.
    """

    # Inicializar X con s muestras aleatorias de Z y eliminarlas de Z
    x_indices, z_indices = initialize_sampling(n_samples=len(X_train), initial_size=s)
    history = [np.copy(x_indices)]
    
    # Inicializar la matriz de distancias M con tamaño n × n
    M = np.full((n, n), np.nan)
    
    results = []
    matriz_vacia = True
    new_idx = None
    previous_sampling_time = np.nan

    for _ in range (num_iter):

        # Entrenar el modelo con X y evaluarlo
        metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=x_indices)
        metrics["sampling_time_seconds"] = previous_sampling_time
        results.append(metrics)

        if len(z_indices) == 0:
            break
        
        start_time = time.perf_counter()

        S, z_indices, new_idx = select_k_greedySample( X_train=X_train, x_indices=x_indices, z_indices=z_indices, k=k,  M=M,
                                                       metric=metric, matrizVacia=matriz_vacia, new_idx=new_idx )
        matriz_vacia = False
        
        # Añadir los puntos seleccionados a X
        x_indices = np.append(x_indices, S)

        previous_sampling_time = time.perf_counter() - start_time

        history.append(np.copy(x_indices))

    # Añadir una evaluación final si la última selección no ha sido evaluada.
    if results[-1]["n_samples"] != len(x_indices):
            metrics = train_and_evaluate( X_train=X_train, y_train=y_train, X_test=X_test,  y_test=y_test, selected_indices=x_indices)
            metrics["sampling_time_seconds"] = previous_sampling_time
            results.append(metrics)

    if return_history:
        return pd.DataFrame(results), history
    else:
        return pd.DataFrame(results)



def run_greedy_for_dataset(dataset_name: str, train_path: str, test_path: str, output_dir: str, s: int, k: int) -> None:
    """
    Ejecuta el algoritmo greedy para un dataset concreto.
    Lee los datos, separa variables y etiquetas, escala las características,
    ejecuta el algoritmo y guarda los resultados y gráficos.
    """
    # Cargar y escalar los datos
    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    # Calcular el número de iteraciones necesarias para recorrer todos los datos (T)
    num_iter = int(np.ceil((len(X_train_scaled) - s) / k))

    # Ejecutar Greedy Sampling
    results_df = greedy_sampling(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test,
                                    n=len(X_train_scaled), s=s,k=k, num_iter= num_iter, metric=DISTANCE_METRIC, return_history=False)

    # Guardar los resultados en un archivo CSV
    results_df.to_csv(os.path.join(output_dir, "greedy_sampling_top20_features.csv"), index=False)

    # Generar y guardar gráficos para cada métrica
    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"Greedy Sampling - {name} ({dataset_name})", output_path=f"{output_dir}/plot_{col}.png", method_label="Greedy sampling")


def main() -> None:
    s = INITIAL_SIZE
    k = BATCH_SIZE
    
    for dataset_name in DATASETS:
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"{GS_PATH}/greedy_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Empieza GS para el dataset {dataset_name}.")

        run_greedy_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path,
            output_dir=output_dir, s=s, k=k)

        print(f"Ha acabado GS para el dataset {dataset_name}.")

if __name__ == "__main__":
    main()