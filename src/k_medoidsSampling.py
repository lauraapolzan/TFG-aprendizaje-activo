import time
import numpy as np
import pandas as pd
import os
from samplingUtils import load_data, save_plot, train_and_evaluate, initialize_sampling, get_distance
from greedySampling import select_k_greedySample

RANDOM_STATE = 0
N_ESTIMATORS = 100
DATASETS = ["HLM", "MDR1_ER", "RLM", "Sol", "hPPB", "rPPB"]
METRICS_TO_PLOT = [("mse", "MSE"), ("pearson_r", "Pearson r"), ("mae", "MAE"), ("r2", "R2")]


def k_medoids_sampling(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                                y_test: np.ndarray, n: int, s: int, k: int, num_iter: int, metrica: str ) -> pd.DataFrame:
    """
    Implementa el algoritmo de k-medoids sampling, seleccionando de forma iterativa las muestras.

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
    # Inicializar matriz de distancias
    M = np.full((n, n), np.nan)
    matriz_vacia = True
    new_idx = None
    results = []
    for iteracion in range(num_iter):
        print(f"Iteración {iteracion + 1}/{num_iter} - X={len(x_indices)}, Z={len(z_indices)}")

        # Entrenar el modelo con X y evaluarlo
        metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=x_indices)
        results.append(metrics)

        if len(z_indices) == 0:
            break
        
        
        S, z_indices, new_idx = select_k_greedySample( X_train=X_train, x_indices=x_indices, z_indices=z_indices, k=k, M=M, metrica=metrica,  matrizVacia=matriz_vacia, new_idx=new_idx )
        matriz_vacia = False
        swap = True
        while swap:
            swap = False
            # Para cada punto z en Z, asignar z a su punto más cercano de X U S
            assigned_to = {}
            for z in z_indices:
                closest_point = None
                min_distance = float("inf")

                for p in np.concatenate([x_indices, S]):
                    distance = get_distance(X_train=X_train, M=M, i=z, j=p, metrica=metrica)

                    if distance < min_distance:
                        min_distance = distance
                        closest_point = p

                assigned_to[z] = closest_point


            # Construir Z' con los puntos de Z asignados a algún punto de S
            Z_prime_list = []

            for z in z_indices:
                assigned_point = assigned_to[z]

                # Comprobar si el punto asignado pertenece a S
                assigned_to_S = False

                for s_i in S:
                    if assigned_point == s_i:
                        assigned_to_S = True
                        break

                # Si el punto más cercano de z está en S, entonces z pertenece a Z'
                if assigned_to_S:
                    Z_prime_list.append(z)

            Z_prime = np.array(Z_prime_list, dtype=int)

            if len(Z_prime) == 0:
                break
            
            # Calcular d* como la suma de las distancias entre cada z' y su punto más cercano en S
            d_estrella = 0.0
            for z_prime in Z_prime:
                assigned_point = assigned_to[z_prime]
                d_estrella += get_distance(X_train=X_train, M=M, i=z_prime, j=assigned_point, metrica=metrica )


            for i in range(len(S)):
                for j in range(len(Z_prime)):
                    old_s = S[i]
                    old_z = Z_prime[j]

                    # Swap temporal
                    S[i] = old_z
                    Z_prime[j] = old_s

                    # Calcular d con S y Z_prime modificados temporalmente
                    d = 0.0
                    for z_aux in Z_prime:
                        min_distance = float("inf")

                        for s_j in S:
                            distance = get_distance(X_train=X_train, M=M, i=z_aux, j=s_j, metrica=metrica )

                            if distance < min_distance:
                                min_distance = distance

                        d += min_distance

                    if d < d_estrella:
                        # Se mantiene el swap y se actualiza Z con el cambio de s y z
                        z_indices = z_indices[z_indices != old_z]
                        z_indices = np.append(z_indices, old_s)

                        d_estrella = d
                        swap = True
                    #    break

                    else:
                        # Deshacer el swap temporal
                        S[i] = old_s
                        Z_prime[j] = old_z

                #if swap:
                #    break
        
        # Añadir los puntos finales de S a X
        x_indices = np.append(x_indices, S)

        # Asegurar que los puntos de S no permanecen en Z
        for s_i in S:
            z_indices = z_indices[z_indices != s_i]

    return pd.DataFrame(results)
            


def run_k_medoids_for_dataset(dataset_name: str, train_path: str, test_path: str, output_dir: str, s: int, k: int) -> None:
    """
    Ejecuta el algoritmo k-medoids para un dataset concreto.
    Lee los datos, separa variables y etiquetas, escala las características,
    ejecuta el algoritmo y guarda los resultados y gráficos.
    """
    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    num_iter = int(np.ceil((len(X_train_scaled) - s) / k))

    results_df = k_medoids_sampling(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test,
        n=len(X_train_scaled), s=s,k=k, num_iter= num_iter, metrica="euclidean")


    results_df.to_csv(os.path.join(output_dir, "k_medoids_sampling_top20_features.csv"), index=False)

    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"K-Medoids Sampling - {name} ({dataset_name})", output_path=f"{output_dir}/plot_{col}.png", method_label="K-medoids sampling")


def main() -> None:
    s = 50
    k = 5
    
    for dataset_name in DATASETS:
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"../experiments/k_medoidsSampling/k_medoids_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        run_k_medoids_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path,
            output_dir=output_dir, s=s, k=k)


if __name__ == "__main__":
    main()