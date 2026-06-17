from pathlib import Path
import pandas as pd
from config import DATASETS, BASE_PATH, RS_PATH, GS_MIN_DIST_PATH, KMS_PATH

TOTAL_SAMPLES = {
    "HLM": 2469,
    "RLM": 2443,
    "MDR1_ER": 2113,
    "Sol": 1738,
    "hPPB": 1446,
    "rPPB": 708,
}

OUTPUT_DIR = Path("../results/tables")

# margen de 0,5%
MARGIN = 0.05

SAMPLING_CONFIGS = {
    "random": {
        "base_path": Path(RS_PATH),
        "folder_prefix": "random_sampling",
        "file_name": "random_sampling_top20_features.csv",
        "output_file": "random_samples_to_baseline.csv",
    },
    "greedy": {
        "base_path": Path(GS_MIN_DIST_PATH),
        "folder_prefix": "greedy_sampling",
        "file_name": "greedy_sampling_top20_features.csv",
        "output_file": "greedy_samples_to_baseline.csv",
    },
    "k_medoids": {
        "base_path": Path(KMS_PATH),
        "folder_prefix": "k_medoids_sampling",
        "file_name": "k_medoids_sampling_top20_features.csv",
        "output_file": "k_medoids_samples_to_baseline.csv",
    },
}


def first_sample_margin(df: pd.DataFrame, metric: str, baseline_value: float) -> int | None:
    """
    Devuelve el primer número de muestras en el que la métrica indicada
    se aproxima al valor del modelo base dentro del margen definido.
    """
    relativeDiff = (df[metric] - baseline_value).abs() / abs(baseline_value)
    valid_rows = df[relativeDiff <= MARGIN]

    if valid_rows.empty:
        return None

    return int(valid_rows.iloc[0]["n_samples"])


def percentageResult(n_samples: int | None, t_samples: int) -> str:
    """
    Devuelve el número de muestras junto con su porcentaje respecto al total.
    """

    if n_samples is None:
        return "-"

    percentage = 100 * n_samples / t_samples
    percentage_txt = f"{percentage:.1f}".replace(".", ",")

    return f"{n_samples} ({percentage_txt}%)"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for strategy_config in SAMPLING_CONFIGS.values():
        rows = []

        for dataset_name in DATASETS:
            folder_name = dataset_name.lower()

            results_path = strategy_config["base_path"] / f"{strategy_config['folder_prefix']}_{folder_name}" / strategy_config["file_name"]
            baseline_path = Path(BASE_PATH) / f"baseline_{folder_name}" / "metrics_top20_features.csv"

            df = pd.read_csv(results_path)
            baseline = pd.read_csv(baseline_path).iloc[0]
            total_samples = TOTAL_SAMPLES[dataset_name]

            first_mse = first_sample_margin(df, "mse", baseline["mse"])
            first_r2 = first_sample_margin(df, "r2", baseline["r2"])

            rows.append({
                "Dataset": dataset_name,
                "Muestras totales": total_samples,
                "MSE": percentageResult(first_mse, total_samples),
                "R2": percentageResult(first_r2, total_samples),
            })

        table = pd.DataFrame(rows)
        output_path = OUTPUT_DIR / strategy_config["output_file"]
        table.to_csv(output_path, index=False, sep=";", encoding="utf-8-sig")


if __name__ == "__main__":
    main()