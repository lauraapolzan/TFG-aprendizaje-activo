import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14}) 
from config import COMPARISON_PATH, DATASETS, METRICS_TO_PLOT, BASE_PATH, RS_PATH, GS_MIN_DIST_PATH, KMS_PATH


def save_comparison_plot(greedy_df: pd.DataFrame, random_df: pd.DataFrame, k_medoids_df: pd.DataFrame, x_col: str, y_col: str, y_label: str, title: str, 
                         output_path: str, baseline_value: float | None = None) -> None:
    """
    Crea una gráfica que compara Greedy, Random y K-medoids para una métrica concreta.
    """

    plt.figure(figsize=(10, 6))

    greedy_y = greedy_df[y_col].rolling(window=5, min_periods=1).mean()
    random_y = random_df[y_col].rolling(window=5, min_periods=1).mean()
    k_medoids_y = k_medoids_df[y_col].rolling(window=5, min_periods=1).mean()

    plt.plot(greedy_df[x_col], greedy_y, label="Greedy sampling", linewidth=2.5)

    plt.plot(random_df[x_col], random_y, label="Random sampling", linewidth=2.5)

    plt.plot(k_medoids_df[x_col], k_medoids_y, label="K-medoids sampling", linewidth=2.5)

    if baseline_value is not None:
        plt.axhline(y=baseline_value, color="red", linestyle=":", linewidth=2, label="Baseline RF top-20")

    all_values = pd.concat([greedy_y, random_y, k_medoids_y])

    if baseline_value is not None:
        all_values = pd.concat([all_values, pd.Series([baseline_value])])

    y_min = all_values.min()
    y_max = all_values.max()
    margin = (y_max - y_min) * 0.08

    if margin == 0:
        margin = 0.01

    plt.ylim(y_min - margin, y_max + margin)
    plt.xlabel("Número de muestras de entrenamiento")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
 
    for dataset in DATASETS:

        greedy_csv = f"{GS_MIN_DIST_PATH}/greedy_sampling_{dataset.lower()}/greedy_sampling_top20_features.csv"
        random_csv = f"{RS_PATH}/random_sampling_{dataset.lower()}/random_sampling_top20_features.csv"
        k_medoids_csv = f"{KMS_PATH}/k_medoids_sampling_{dataset.lower()}/k_medoids_sampling_top20_features.csv"
        baseline_csv = f"{BASE_PATH}/baseline_{dataset.lower()}/metrics_top20_features.csv"

        output_dir = f"{COMPARISON_PATH}/comparacion_{dataset.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        greedy_df = pd.read_csv(greedy_csv)
        random_df = pd.read_csv(random_csv)
        k_medoids_df = pd.read_csv(k_medoids_csv)
        baseline_metrics = pd.read_csv(baseline_csv).iloc[0]

        
        for metric_col, metric_label in METRICS_TO_PLOT:
            output_path = os.path.join(
                output_dir,
                f"comparacion_{dataset.lower()}_{metric_col}.png"
            )

            save_comparison_plot(greedy_df=greedy_df, random_df=random_df, k_medoids_df=k_medoids_df, x_col="n_samples",
                y_col=metric_col, y_label=metric_label, title=f"{dataset} - {metric_label} vs número de muestras",
                output_path=output_path, baseline_value=baseline_metrics.get(metric_col))

    print("\nGráficas guardadas.")



if __name__ == "__main__":
    main()