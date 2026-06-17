import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from config import DATASETS, BASE_PATH


def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Guarda un diccionario de métricas en un fichero de texto.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def plot_feature_importance(importances: np.ndarray, feature_names: list[str], output_path: str, top_n: int = 20) -> None:
    """
    Genera una gráfica de barras horizontales con las características más importantes.
    """
    importance_df = pd.DataFrame({ "feature": feature_names, "importance": importances
                                  }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    plt.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
    plt.xlabel("Importancia")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} features más importantes")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_top_features_csv(importances: np.ndarray, feature_names: list[str], output_path: str, top_n: int) -> list[str]:
    """
    Guarda las características más importantes en un archivo CSV y devuelve sus nombres.
    """
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    top_features_df = importance_df.head(top_n)
    top_features_df.to_csv(output_path, index=False)

    return top_features_df["feature"].tolist()


def save_reduced_datasets(df_train: pd.DataFrame, df_test: pd.DataFrame, selected_features: list[str], dataset_name: str) -> tuple[str, str]:
    """
    Guarda los conjuntos de entrenamiento y test reducidos a las características seleccionadas.
    Devuelve las rutas de los archivos guardados.
    """
    
    reduced_data_dir = "../data/reduced_datasets"
    os.makedirs(reduced_data_dir, exist_ok=True)

    reduced_train = df_train[["ID", "activity"] + selected_features]
    reduced_test = df_test[["ID", "activity"] + selected_features]

    reduced_train_path = os.path.join(reduced_data_dir, f"ADME_{dataset_name}_train_feat_top20.csv")
    reduced_test_path = os.path.join(reduced_data_dir, f"ADME_{dataset_name}_test_feat_top20.csv")

    reduced_train.to_csv(reduced_train_path, index=False)
    reduced_test.to_csv(reduced_test_path, index=False)

    return reduced_train_path, reduced_test_path



def plot_feature_count_metric(results: list[dict], dataset_name: str, metric: str, output_path: str) -> None:
    """
    Guarda una gráfica de una métrica en función del número de características.
    """
    df_results = pd.DataFrame(results)

    metric_labels = {
        "rmse": "RMSE",
        "r2": "R2",
        "mse": "MSE",
        "mae": "MAE",
        "pearson_r": "Pearson r"
    }

    metric_label = metric_labels.get(metric, metric)

    plt.figure(figsize=(8, 5))
    plt.plot(df_results["n_features"], df_results[metric], marker="o")
    plt.xlabel("Número de características")
    plt.ylabel(metric_label)
    plt.title(f"{dataset_name} - {metric_label} según número de características")
    plt.xticks(df_results["n_features"])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_global_feature_summary():
    """
    Crea un resumen global de los resultados de la evaluación de distintos números de características
    y para cada dataset se carga el fichero feature_count_results.csv y se genera una tabla con el RMSE por número de características.
    """
    all_results = []

    for dataset_name in DATASETS:
        path = os.path.join(
            BASE_PATH,
            f"baseline_{dataset_name.lower()}",
            "feature_count_results.csv"
        )

        df = pd.read_csv(path)
        df["dataset"] = dataset_name
        all_results.append(df)

    df_all = pd.concat(all_results)

    # Tabla pivot (datasets en columnas)
    pivot = df_all.pivot_table(index="n_features", columns="dataset", values="rmse")

    # Media global
    pivot["mean_rmse"] = pivot.mean(axis=1)

    output_path = os.path.join(BASE_PATH, "global_feature_summary.csv")
    pivot.to_csv(output_path)

    print("Resumen global guardado en:", output_path)


def read_metric_from_txt(metrics_path: str, metric: str) -> float:
    """
    Lee una métrica concreta desde un fichero de métricas en formato txt.
    """
    with open(metrics_path, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith(f"{metric}:"):
                return float(line.split(":")[1].strip())

    raise ValueError(f"No se ha encontrado la métrica '{metric}' en {metrics_path}")


def load_feature_count_results(base_path: str, datasets: list[str], metric: str) -> tuple[pd.DataFrame, list[float]]:
    """
    Carga los resultados obtenidos con distintos números de características
    y la métrica correspondiente al modelo con todas las características.
    """
    all_results = []
    all_features_values = []

    for dataset_name in datasets:
        dataset_dir = os.path.join(base_path, f"baseline_{dataset_name.lower()}")

        feature_count_path = os.path.join(dataset_dir, "feature_count_results.csv")
        metrics_path = os.path.join(dataset_dir, "metrics_all_features.txt")

        df = pd.read_csv(feature_count_path)
        df["dataset"] = dataset_name
        all_results.append(df)

        metric_value = read_metric_from_txt(metrics_path, metric)
        all_features_values.append(metric_value)

    return pd.concat(all_results, ignore_index=True), all_features_values


def plot_mean_feature_count_metric_with_baseline(metric: str, base_path: str = BASE_PATH) -> None:
    """
    Guarda una gráfica con el valor medio de una métrica según el número de
    características seleccionadas, comparándolo con el modelo entrenado con
    todas las características.
    """
    metric_labels = {
        "mse": "MSE",
        "rmse": "RMSE",
        "r2": "R2",
        "mae": "MAE",
        "pearson_r": "Pearson r"
    }

    metric_label = metric_labels.get(metric, metric)

    df_all, all_features_values = load_feature_count_results(base_path=base_path, datasets=DATASETS, metric=metric)

    pivot = df_all.pivot_table(index="n_features", columns="dataset", values=metric)

    mean_metric_column = f"mean_{metric}"
    pivot[mean_metric_column] = pivot.mean(axis=1)

    mean_all_features = sum(all_features_values) / len(all_features_values)

    plt.figure(figsize=(8, 5))
    plt.plot(pivot.index, pivot[mean_metric_column], marker="o", linewidth=2.5, label=f"{metric_label} medio")
    plt.axvline(x=20, color="red", linestyle="--", linewidth=2, label="n = 20")
    plt.axhline(y=mean_all_features, color="yellow", linestyle=":", linewidth=2, label=f"{metric_label} medio con todas las características")

    plt.xlabel("Número de características")
    plt.ylabel(f"{metric_label} medio")
    plt.title(f"{metric_label} medio según número de características")
    plt.xticks(pivot.index)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(base_path, f"mean_feature_count_{metric}_with_baseline.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print("Gráfica guardada en:", output_path)
