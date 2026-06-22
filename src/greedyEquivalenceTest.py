from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from config import DATASETS, DISTANCE_METRIC
from greedySampling import greedy_sampling as greedy_original
from greedySampling_minDist import greedy_sampling as greedy_min_dist


DATA_DIR = Path("../data/reduced_datasets")
OUTPUT_DIR = Path("../results/tests/greedy_equivalence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_SIZE = 50
BATCH_SIZE = 5
NUM_ITER = 5

METRIC_COLUMNS = ["mse", "rmse", "mae", "r2", "pearson_r"]

def compare_histories(hist_original: list[np.ndarray], hist_min_dist: list[np.ndarray]) -> tuple[bool, int, int]:
    """
    Compara si las muestras seleccionadas en cada iteración son las mismas.
    Devuelve un booleano indicando si son iguales, el número de iteraciones coincidentes y el número total de iteraciones.
    """
    if len(hist_original) != len(hist_min_dist):
        return False, 0, max(len(hist_original), len(hist_min_dist))

    matching_iterations = 0

    for i in range(len(hist_original)):
        if np.array_equal(hist_original[i], hist_min_dist[i]):
            matching_iterations += 1

    return matching_iterations == len(hist_original), matching_iterations, len(hist_original)


def max_difference(results_original: pd.DataFrame, results_min_dist: pd.DataFrame) -> float:
    """
    Calcula la diferencia máxima entre las métricas de las dos implementaciones.
    """
    max_diff = 0.0

    for col in METRIC_COLUMNS:
        if col in results_original.columns and col in results_min_dist.columns:
            diff = np.abs(results_original[col] - results_min_dist[col]).max()
            max_diff = max(max_diff, diff)

    return max_diff


def plot_metric_differences(summary_df: pd.DataFrame) -> None:
    """
    Genera una gráfica con la diferencia máxima entre métricas por dataset.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["Dataset"], summary_df["Diferencia máxima métricas"])
    plt.xlabel("Dataset")
    plt.ylabel("Diferencia máxima absoluta")
    plt.title("Diferencia máxima entre métricas de ambas implementaciones Greedy")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "greedy_metric_differences.png", dpi=300)
    plt.close()


def main() -> None:
    summary_rows = []

    for dataset_name in DATASETS:
        print(f"Dataset: {dataset_name}")

        train_path = DATA_DIR / f"ADME_{dataset_name}_train_feat_top20.csv"
        test_path = DATA_DIR / f"ADME_{dataset_name}_test_feat_top20.csv"

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        X_train = df_train.drop(columns=["activity", "ID"])
        y_train = df_train["activity"].to_numpy()
        X_test = df_test.drop(columns=["activity", "ID"])
        y_test = df_test["activity"].to_numpy()

        # Se escala igual que en los experimentos principales.
        scaler = RobustScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Se ejecutan las dos implementaciones de Greedy.
        results_original, history_original = greedy_original(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, n=len(X_train_scaled), s=INITIAL_SIZE, k=BATCH_SIZE, num_iter=NUM_ITER, metric=DISTANCE_METRIC, return_history=True)
        results_min_dist, history_min_dist = greedy_min_dist(X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, n=len(X_train_scaled), s=INITIAL_SIZE, k=BATCH_SIZE, num_iter=NUM_ITER, metrica=DISTANCE_METRIC, return_history=True)

        same_history, matching_iterations, total_iterations = compare_histories(history_original, history_min_dist)

        try:
            pd.testing.assert_frame_equal(results_original[METRIC_COLUMNS].reset_index(drop=True), results_min_dist[METRIC_COLUMNS].reset_index(drop=True), check_exact=False, check_dtype=False, atol=1e-5)
            same_metrics = True
        except AssertionError as e:
            same_metrics = False
            print(e)

        max_diff = max_difference(results_original, results_min_dist)

        summary_rows.append({
            "Dataset": dataset_name,
            "Iteraciones comparadas": total_iterations,
            "Iteraciones iguales": matching_iterations,
            "Mismas muestras": "Sí" if same_history else "No",
            "Mismas métricas": "Sí" if same_metrics else "No",
            "Diferencia máxima métricas": max_diff,
        })

        print(f"Mismas muestras seleccionadas: {same_history}")
        print(f"Iteraciones iguales: {matching_iterations}/{total_iterations}")
        print(f"Mismas métricas: {same_metrics}")
        print(f"Diferencia máxima entre métricas: {max_diff}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "greedy_equivalence_summary.csv", index=False, sep=";", encoding="utf-8-sig")

    #plot_metric_differences(summary_df)



if __name__ == "__main__":
    main()