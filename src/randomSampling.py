import time
import numpy as np
import pandas as pd
import os
from samplingUtils import save_plot, train_and_evaluate, initialize_sampling, load_data
from config import RANDOM_STATE, DATASETS, METRICS_TO_PLOT, RS_PATH, INITIAL_SIZE, BATCH_SIZE


def random_sampling(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                    s: int, k: int) -> pd.DataFrame:
    """
    Ejecuta la estrategia Random Sampling sobre un conjunto de datos. 
    Se selecciona un conjunto inicial de tamaño s y luego se añaden k muestras aleatorias en cada iteración.
    Se entrena un modelo de Random Forest con las muestras seleccionadas y se evalúa su rendimiento.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        y_train: etiquetas del conjunto de entrenamiento.
        X_test: conjunto de datos de prueba.
        y_test: etiquetas del conjunto de prueba.
        s: número de muestras iniciales a seleccionar.
        k: Número de muestras a seleccionar en cada iteración.

    Devuelve:
        DataFrame con los resultados obtenidos en cada iteración.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    # Iniciallizar X (selected_indices) con s muestras aleatorias de Z (pool_indices) y eliminarlas de Z
    selected_indices, pool_indices = initialize_sampling(n_samples=len(X_train), initial_size=s)

    results = []
    previous_sampling_time = np.nan

    # Ejecutar el muestreo aleatorio hasta que no queden más muestras en Z (pool_indices)
    while len(pool_indices) > 0:
        
        # Entrenar el modelo con X y evaluarlo
        metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=selected_indices)
        metrics["sampling_time_seconds"] = previous_sampling_time
        results.append(metrics)

        # Seleccionar como máximo batch_size muestras, evitando superar el tamaño del pool
        current_batch_size = min(k, len(pool_indices))

        start_time = time.perf_counter()
        
        new_indices = rng.choice(pool_indices, size=current_batch_size, replace=False)

        # Añadir los nuevos índices seleccionados a selected_indices y eliminarlos de pool_indices
        selected_indices = np.concatenate((selected_indices, new_indices))
        pool_indices = np.setdiff1d(pool_indices, new_indices, assume_unique=False)
        
        previous_sampling_time = time.perf_counter() - start_time

    # Evaluación final usando todas las muestras seleccionadas
    metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=selected_indices)
    metrics["sampling_time_seconds"] = previous_sampling_time
    results.append(metrics)

    return pd.DataFrame(results)


def run_random_for_dataset(dataset_name: str, train_path: str, test_path: str, output_dir: str, s: int, k: int) -> None:
    """
    Ejecuta Random Sampling para un conjunto de datos concreto.
    Lee los datos, separa variables y etiquetas, escala las características,
    ejecuta el algoritmo y guarda los resultados y gráficos.
    """
    # Cargar y escalar los datos
    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    # Ejecutar Random Sampling
    results_df = random_sampling(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, s=s, k=k)

    # Guardar los resultados en un archivo CSV
    results_df.to_csv(os.path.join(output_dir, "random_sampling_top20_features.csv"), index=False)

    # Generar y guardar gráficos para cada métrica
    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"Random Sampling - {name} ({dataset_name})",
            output_path=f"{output_dir}/plot_{col}.png", method_label="Random Sampling", show_baseline=True, dataset_name=dataset_name)


def main() -> None:
    
    for dataset_name in DATASETS:
        
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"{RS_PATH}/random_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Empieza RS para el dataset {dataset_name}.")
        
        run_random_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path, output_dir=output_dir,
            s=INITIAL_SIZE, k=BATCH_SIZE)
        
        print(f"Ha acabado RS para el dataset {dataset_name}.")


if __name__ == "__main__":
    main()