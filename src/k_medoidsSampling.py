import time
import numpy as np
import pandas as pd
import os
from samplingUtils import load_data, save_plot, train_and_evaluate, initialize_sampling, get_distance
from greedySampling_minDist import select_k_greedySample, initialize_min_distances
from config import DATASETS, DISTANCE_METRIC, METRICS_TO_PLOT, KMS_PATH, INITIAL_SIZE, BATCH_SIZE


def update_minDistances( X_train: np.ndarray, previous_z_indices: np.ndarray, previous_min_distances: np.ndarray,
                                           z_indices: np.ndarray, S: np.ndarray, M: np.ndarray, metrica: str) -> np.ndarray:
    """
    Actualiza el vector de distancias mínimas después del refinamiento
    realizado por K-medoids.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        previous_z_indices: índices de Z antes de aplicar Greedy.
        previous_min_distances: vector de distancias mínimas antes de aplicar Greedy.
        z_indices: índices de las muestras no seleccionadas después del refinamiento.
        S: conjunto final de muestras seleccionadas en la iteración.
        M: matriz de distancias.
        metrica: métrica a utilizar para calcular las distancias.

    Devuelve:
        updated_minDist: vector de distancias mínimas actualizado.
    """

    previous_min_by_index = np.full(len(X_train), np.inf)
    previous_min_by_index[previous_z_indices] = previous_min_distances
    updated_minDist = np.full(len(z_indices), np.inf)

    for z_pos, z_actual in enumerate(z_indices):
        min_distance = previous_min_by_index[z_actual]

        for s_i in S:
            distance = get_distance(X_train=X_train, M=M, i=int(s_i), j=int(z_actual), metrica=metrica)
            if distance < min_distance:
                min_distance = distance

        updated_minDist[z_pos] = min_distance

    return updated_minDist

def k_medoids_sampling( X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                        n: int, s: int, k: int, num_iter: int, metrica: str) -> pd.DataFrame:
    """
    Implementa Incremental K-medoids Sampling.

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
    
    Devuelve:
        DataFrame con los resultados de cada iteración.
    """

    # Inicializar X con s muestras aleatorias y eliminarlas de Z.
    x_indices, z_indices = initialize_sampling(n_samples=len(X_train), initial_size=s)

    # Inicializar matriz de distancias utilizada como caché.
    M = np.full((n, n), np.nan)

    # Calcular las distancias mínimas iniciales entre Z y X.
    min_distances = initialize_min_distances(X_train=X_train, x_indices=x_indices, z_indices=z_indices, M=M, metrica=metrica)
    
    results = []
    previous_sampling_time = np.nan

    for _ in range(num_iter):

        # Entrenar y evaluar el modelo con el conjunto actual.
        metrics = train_and_evaluate(  X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=x_indices)
        metrics["sampling_time_seconds"] = previous_sampling_time
        results.append(metrics)

        if len(z_indices) == 0:
            break

        # Guardar el estado anterior a Greedy para reconstruir correctamente min_distances
        previous_z_indices = np.copy(z_indices)
        previous_min_distances = np.copy(min_distances)

        start_time = time.perf_counter()

        # Selección inicial mediante Greedy con vector de distancias mínimas.
        S, z_indices, _ = select_k_greedySample(X_train=X_train, z_indices=z_indices, k=k,  M=M, min_distances=np.copy(min_distances), metrica=metrica)

        swap = True
        while swap:
            swap = False
            
            # Asignar cada punto z de Z a su punto más cercano de X U S.
            assigned_to = {}
            for z in z_indices:
                closest_point = None
                min_distance = float("inf")

                for p in np.concatenate([x_indices, S]):
                    distance = get_distance(X_train=X_train,  M=M, i=int(z), j=int(p), metrica=metrica)

                    if distance < min_distance:
                        min_distance = distance
                        closest_point = p

                assigned_to[z] = closest_point

            # Construir Z' con los puntos asignados a algún punto de S.
            Z_prime_list = []
            for z in z_indices:
                assigned_point = assigned_to[z]

                if assigned_point in S:
                    Z_prime_list.append(z)

            Z_prime = np.array(Z_prime_list, dtype=int)

            if len(Z_prime) == 0:
                break

            # Calcular d* como la suma de las distancias entre cada z'
            # y el punto de S al que está asignado.
            d_estrella = 0.0

            for z_prime in Z_prime:
                assigned_point = assigned_to[z_prime]
                d_estrella += get_distance(X_train=X_train, M=M, i=int(z_prime), j=int(assigned_point), metrica=metrica )

            # Evaluar posibles intercambios entre S y Z'.
            for i in range(len(S)):
                for j in range(len(Z_prime)):
                    old_s = S[i]
                    old_z = Z_prime[j]

                    S_temp = S.copy()
                    Z_prime_temp = Z_prime.copy()

                    S_temp[i] = old_z
                    Z_prime_temp[j] = old_s

                    # Calcular d como la suma de las distancias entre cada z' y el punto de S más cercano.
                    d = 0.0
                    for z_aux in Z_prime_temp:
                        min_distance = float("inf")

                        for s_j in S_temp:
                            distance = get_distance( X_train=X_train, M=M, i=int(z_aux), j=int(s_j), metrica=metrica)

                            if distance < min_distance:
                                min_distance = distance

                        d += min_distance

                    if d < d_estrella:
                        # Aceptar el intercambio.
                        S[i] = old_z
                        Z_prime[j] = old_s

                        # old_z entra en S y deja de pertenecer a Z.
                        z_indices = z_indices[z_indices != old_z]

                        # old_s sale de S y vuelve a Z.
                        z_indices = np.append(z_indices, old_s)

                        d_estrella = d
                        swap = True

        # Añadir los puntos finales de S a X.
        x_indices = np.append(x_indices, S)

        # Asegurar que los puntos seleccionados no permanecen en Z.
        for s_i in S:
            z_indices = z_indices[z_indices != s_i]

        # Actualizar el vector de distancias mínimas 
        min_distances = update_minDistances(X_train=X_train, previous_z_indices=previous_z_indices, 
                                                              previous_min_distances=previous_min_distances, z_indices=z_indices, S=S,  M=M, metrica=metrica )

        previous_sampling_time = time.perf_counter() - start_time

    # Evaluar una última vez después de incorporar las últimas muestras.
    if results[-1]["n_samples"] != len(x_indices):
        metrics = train_and_evaluate( X_train=X_train, y_train=y_train, X_test=X_test,  y_test=y_test, selected_indices=x_indices)
        metrics["sampling_time_seconds"] = previous_sampling_time
        results.append(metrics)

    return pd.DataFrame(results)
            

def run_k_medoids_for_dataset(dataset_name: str, train_path: str, test_path: str, output_dir: str, s: int, k: int) -> None:
    """
    Ejecuta el algoritmo k-medoids para un dataset concreto.
    Lee los datos, separa variables y etiquetas, escala las características,
    ejecuta el algoritmo y guarda los resultados y gráficos.
    """
    # Cargar y escalar los datos
    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    # Calcular el número de iteraciones necesarias para recorrer todos los datos (T)
    num_iter = int(np.ceil((len(X_train_scaled) - s) / k))

    # Ejecutar K-medoids Sampling
    results_df = k_medoids_sampling(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test,
        n=len(X_train_scaled), s=s,k=k, num_iter= num_iter, metrica=DISTANCE_METRIC)

    # Guardar los resultados en un archivo CSV
    results_df.to_csv(os.path.join(output_dir, "k_medoids_sampling_top20_features.csv"), index=False)

    # Generar y guardar gráficos para cada métrica
    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"K-Medoids Sampling - {name} ({dataset_name})", output_path=f"{output_dir}/plot_{col}.png", method_label="K-medoids sampling",
                  show_baseline=True, dataset_name=dataset_name)


def main() -> None:
    s = INITIAL_SIZE
    k = BATCH_SIZE
    
    for dataset_name in DATASETS:
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"{KMS_PATH}/k_medoids_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Empieza K-Medoids para el dataset {dataset_name}.")

        run_k_medoids_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path,
            output_dir=output_dir, s=s, k=k)

        print(f"Ha acabado K-Medoids para el dataset {dataset_name}.")

if __name__ == "__main__":
    main()