import time
import numpy as np
import pandas as pd
import os
from samplingUtils import load_data, save_plot, train_and_evaluate, initialize_sampling, get_distance

RANDOM_STATE = 0
N_ESTIMATORS = 100
DATASETS = ["HLM", "MDR1_ER", "RLM", "Sol", "hPPB", "rPPB"]
METRICS_TO_PLOT = [("mse", "MSE"), ("pearson_r", "Pearson r"), ("mae", "MAE"), ("r2", "R2")]

def initialize_min_distances( X_train: np.ndarray, x_indices: np.ndarray, z_indices: np.ndarray, M: np.ndarray, metrica: str ) -> np.ndarray:
    """
    Calcula el vector inicial de distancias mínimas.

    Para cada punto z de Z, calcula su distancia mínima respecto a los
    puntos inicialmente seleccionados en X. Las distancias calculadas
    se guardan también en la matriz M.
    """

    min_distances = np.full(len(z_indices), np.inf)

    for z_pos in range(len(z_indices)):
        z_actual = z_indices[z_pos]

        for x in x_indices:
            distance = get_distance( X_train=X_train, M=M, i=x, j=z_actual, metrica=metrica )

            if distance < min_distances[z_pos]:
                min_distances[z_pos] = distance

    return min_distances


def select_k_greedySample(X_train: np.ndarray, z_indices: np.ndarray, k: int, M: np.ndarray, 
                           min_distances: np.ndarray, metrica: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Selecciona una muestra S de tamaño k mediante el algoritmo Greedy.

    En cada paso se selecciona el punto de Z que maximiza su distancia mínima
    respecto a los puntos ya seleccionados en X U S.

    El vector min_distances almacena, para cada punto de Z, su distancia
    mínima actual al conjunto seleccionado. La matriz M se mantiene
    actualizada con las distancias calculadas.
    """
    S = []
    k_actual = min(k, len(z_indices))

    for _ in range(k_actual):

        # Seleccionar el punto z con mayor distancia mínima
        max_pos = None
        max_min_dist = -1.0

        for z_pos in range(len(z_indices)):
            if min_distances[z_pos] > max_min_dist:
                max_min_dist = min_distances[z_pos]
                max_pos = z_pos

        # Añadir z a S
        new_idx = z_indices[max_pos]
        S.append(new_idx)

        # Eliminar el punto seleccionado de Z y del vector de distancias mínimas
        z_indices = np.delete(z_indices, max_pos)
        min_distances = np.delete(min_distances, max_pos)

        if len(z_indices) == 0:
            break

        # Actualizar el vector de distancias mínimas con el nuevo punto seleccionado.
        # Al mismo tiempo, las distancias nuevas se guardan en M.
        for z_pos in range(len(z_indices)):
            z_actual = z_indices[z_pos]

            distance = get_distance(X_train=X_train, M=M, i=new_idx, j=z_actual, metrica=metrica )

            if distance < min_distances[z_pos]:
                min_distances[z_pos] = distance

    return np.array(S, dtype=int), z_indices, min_distances



def greedy_sampling(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n: int, s: int, 
                    k: int, num_iter: int, metrica: str, return_history: bool = False) ->  pd.DataFrame | tuple[pd.DataFrame, list[np.ndarray]]:
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
        metrica: Métrica a utilizar para calcular las distancias.
    """

    # Inicializar X con s muestras aleatorias de Z y eliminarlas de Z
    x_indices, z_indices = initialize_sampling(n_samples=len(X_train), initial_size=s)
    history = [np.copy(x_indices)]
    # Inicializar la matriz de distancias M con tamaño n × n
    M = np.full((n, n), np.nan)
    
    # Calcular el vector inicial de distancias mínimas entre Z y X, guardando las distancias calculadas en M
    min_distances = initialize_min_distances( X_train=X_train, x_indices=x_indices, z_indices=z_indices,
                                                M=M, metrica=metrica )

    results = []
    for _ in range (num_iter):

        # Entrenar el modelo con X y evaluarlo
        metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=x_indices)
        results.append(metrics)

        if len(z_indices) == 0:
            break
        
        S, z_indices, min_distances = select_k_greedySample( X_train=X_train, z_indices=z_indices, k=k, M=M, min_distances=min_distances, metrica=metrica )
                                                     
        # Añadir los puntos seleccionados a X
        x_indices = np.append(x_indices, S)
        history.append(np.copy(x_indices))

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
    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    num_iter = int(np.ceil((len(X_train_scaled) - s) / k))

    results_df = greedy_sampling(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test,
                                n=len(X_train_scaled), s=s,k=k, num_iter= num_iter, metrica="euclidean", return_history=False)

    results_df.to_csv(os.path.join(output_dir, "greedy_sampling_top20_features.csv"), index=False)

    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"Greedy Sampling - {name} ({dataset_name})", output_path=f"{output_dir}/plot_{col}.png", method_label="Greedy sampling")


def main() -> None:
    s = 50
    k = 5
    
    for dataset_name in DATASETS:
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"../experiments/greedySampling/greedy_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        run_greedy_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path,
            output_dir=output_dir, s=s, k=k)


if __name__ == "__main__":
    main()