import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from greedySampling import greedy_sampling as greedy_original
from greedySampling_minDist import greedy_sampling as greedy_min_dist


DATASETS = ["HLM", "MDR1_ER", "RLM", "Sol", "hPPB", "rPPB"]


def compare_histories(
    history_original: list[np.ndarray],
    history_min_dist: list[np.ndarray]
) -> tuple[bool, int, int]:
    """
    Compara los historiales de muestras seleccionadas.

    Devuelve:
        - si coinciden todas las iteraciones
        - número de iteraciones coincidentes
        - número total de iteraciones comparadas
    """

    if len(history_original) != len(history_min_dist):
        total_iter = max(len(history_original), len(history_min_dist))
        return False, 0, total_iter

    matching_iterations = 0

    for i in range(len(history_original)):
        if np.array_equal(history_original[i], history_min_dist[i]):
            matching_iterations += 1

    same_history = matching_iterations == len(history_original)

    return same_history, matching_iterations, len(history_original)


def max_metric_difference(
    results_original: pd.DataFrame,
    results_min_dist: pd.DataFrame,
    metric_columns: list[str]
) -> float:
    """
    Calcula la diferencia máxima absoluta entre métricas numéricas.
    """

    max_diff = 0.0

    for col in metric_columns:
        if col in results_original.columns and col in results_min_dist.columns:
            diff = np.abs(results_original[col] - results_min_dist[col]).max()
            max_diff = max(max_diff, diff)

    return max_diff


def plot_metric_differences(summary_df: pd.DataFrame, output_path: str) -> None:
    """
    Genera una gráfica con la diferencia máxima absoluta entre métricas
    para cada dataset.
    """

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["Dataset"], summary_df["Diferencia máxima métricas"])
    plt.xlabel("Dataset")
    plt.ylabel("Diferencia máxima absoluta")
    plt.title("Diferencia máxima entre métricas de ambas implementaciones Greedy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    s = 50
    k = 5
    num_iter = 2
    metrica = "euclidean"

    output_dir = "../experiments/greedy_equivalence"
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []

    metric_columns = ["mse", "rmse", "mae", "r2", "pearson_r"]

    for dataset_name in DATASETS:
        print("\n======================================")
        print(f"Dataset: {dataset_name}")
        print("======================================")

        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        X_train = df_train.drop(columns=["activity", "ID"])
        y_train = df_train["activity"].to_numpy()

        X_test = df_test.drop(columns=["activity", "ID"])
        y_test = df_test["activity"].to_numpy()

        scaler = RobustScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results_original, history_original = greedy_original(
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            n=len(X_train_scaled),
            s=s,
            k=k,
            num_iter=num_iter,
            metrica=metrica,
            return_history=True
        )

        results_min_dist, history_min_dist = greedy_min_dist(
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            n=len(X_train_scaled),
            s=s,
            k=k,
            num_iter=num_iter,
            metrica=metrica,
            return_history=True
        )

        same_history, matching_iterations, total_iterations = compare_histories(
            history_original=history_original,
            history_min_dist=history_min_dist
        )

        try:
            pd.testing.assert_frame_equal(
                results_original,
                results_min_dist,
                check_exact=False,
                atol=1e-10
            )
            same_metrics = True
        except AssertionError:
            same_metrics = False

        max_diff = max_metric_difference(
            results_original=results_original,
            results_min_dist=results_min_dist,
            metric_columns=metric_columns
        )

        total_selected = s + (k * num_iter)

        summary_rows.append({
            "Dataset": dataset_name,
            "Iteraciones comparadas": total_iterations,
            "Iteraciones coincidentes": matching_iterations,
            "Muestras seleccionadas": total_selected,
            "Mismas muestras": "Sí" if same_history else "No",
            "Mismas métricas": "Sí" if same_metrics else "No",
            "Diferencia máxima métricas": max_diff
        })

        print(f"¿Mismas muestras seleccionadas?: {same_history}")
        print(f"Iteraciones coincidentes: {matching_iterations}/{total_iterations}")
        print(f"¿Mismas métricas?: {same_metrics}")
        print(f"Diferencia máxima entre métricas: {max_diff}")

    summary_df = pd.DataFrame(summary_rows)

    csv_path = os.path.join(output_dir, "greedy_equivalence_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    latex_path = os.path.join(output_dir, "greedy_equivalence_summary.tex")
    latex_table = summary_df.to_latex(
        index=False,
        escape=False,
        float_format="%.2e",
        caption="Comparación experimental entre la implementación Greedy basada en matriz y la implementación basada en vector de distancias mínimas.",
        label="tab:greedy_equivalence"
    )

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    plot_path = os.path.join(output_dir, "greedy_metric_differences.png")
    plot_metric_differences(summary_df, plot_path)

    print("\nResumen guardado en:", csv_path)
    print("Tabla LaTeX guardada en:", latex_path)
    print("Gráfica guardada en:", plot_path)
    print("\nResumen:")
    print(summary_df)


if __name__ == "__main__":
    main()