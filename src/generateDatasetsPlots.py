from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATASETS

DATA_DIR = Path("../data/reduced_datasets")
OUTPUT_DIR = Path("../results/analisisDatasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga los ficheros de entrenamiento y test de un dataset.
    """
    train_path = DATA_DIR / f"ADME_{dataset_name}_train_feat_top20.csv"
    test_path = DATA_DIR / f"ADME_{dataset_name}_test_feat_top20.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    return df_train, df_test


def plot_target_boxplot() -> None:
    """
    Genera un boxplot de la variable objetivo para cada dataset.
    Se utiliza el conjunto de entrenamiento.
    """
    datos = []
    labels = []

    for dataset_name in DATASETS:
        df_train, _ = load_dataset(dataset_name)
        datos.append(df_train["activity"])
        labels.append(dataset_name)

    plt.figure(figsize=(10, 6))
    plt.boxplot(datos, tick_labels=labels, showmeans=True)

    plt.xlabel("Dataset")
    plt.ylabel("Variable objetivo")
    plt.title("Distribución de la variable objetivo en el conjunto de entrenamiento")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/target_boxplot_train.png", dpi=300)
    plt.close()


def plot_target_histograms() -> None:
    """
    Genera un histograma de la variable objetivo para cada dataset.
    """

    for dataset_name in DATASETS:
        df_train, df_test = load_dataset(dataset_name)

        plt.figure(figsize=(8, 5))
        plt.hist(df_train["activity"], bins=30, alpha=0.7, label="Train")
        plt.hist( df_test["activity"], bins=30, alpha=0.5, label="Test" )
        plt.xlabel("Variable objetivo")
        plt.ylabel("Frecuencia")
        plt.title(f"Distribución de la variable objetivo - {dataset_name}")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/hist_activity_{dataset_name.lower()}.png", dpi=300)
        plt.close()

def generate_target_descriptive_stats() -> pd.DataFrame:
    """
    Genera una tabla con estadísticos descriptivos de la variable objetivo.
    """
    rows = []

    for dataset_name in DATASETS:
        file_path = DATA_DIR / f"ADME_{dataset_name}_train_feat_top20.csv"
        df = pd.read_csv(file_path)

        target = df["activity"]

        rows.append(
            {
                "Dataset": dataset_name,
                "n": target.shape[0],
                "Media": target.mean(),
                "Desv. típ.": target.std(),
                "Mín.": target.min(),
                "Q1": target.quantile(0.25),
                "Mediana": target.median(),
                "Q3": target.quantile(0.75),
                "Máx.": target.max(),
            }
        )

    return pd.DataFrame(rows).round(3)

def main() -> None:

    plot_target_boxplot()
    plot_target_histograms()
    print(f"\nTablas y gráficas guardadas en: {OUTPUT_DIR}")
    descriptive_stats = generate_target_descriptive_stats()
    descriptive_stats.to_csv(OUTPUT_DIR / "target_descriptive_stats.csv", index=False, sep=";", encoding="utf-8-sig")


if __name__ == "__main__":
    main()