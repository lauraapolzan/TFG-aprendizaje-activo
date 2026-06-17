import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import RANDOM_STATE, N_ESTIMATORS, BASE_PATH


def get_distance(X_train: np.ndarray, M: np.ndarray, i: int, j: int, metrica: str) -> float:
    """
    Devuelve la distancia entre dos puntos.
    Si la distancia no está calculada en la matriz M, se calcula y se guarda.
    
    Paramétros:
        X_train: conjunto de datos de entrenamiento.
        M: matriz de distancias.
        i: índice del primer punto.
        j: índice del segundo punto.
        metrica: métrica a utilizar para calcular la distancia.
    
    Devuelve:
        La distancia entre los puntos i y j.
    """
    if np.isnan(M[i, j]):
        dist = pairwise_distances(X_train[[i]], X_train[[j]],  metric=metrica)[0][0]
        M[i, j] = dist
        M[j, i] = dist
    
    return M[i, j]

def initialize_sampling(n_samples: int, initial_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Inicializa el algoritmo de muestreo seleccionando aleatoriamente un conjunto inicial de tamaño 
    initial_size y deja el resto de las muestras en z_indices.

    Parámetros:
        n_samples: Número total de muestras.
        initial_size: Tamaño inicial del conjunto inicial a seleccionar.
    Devuelve:
        selected_indices: índices de las muestras seleccionadas inicialmente.
        z_indices: índices de las muestras no seleccionadas.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # Se crean los indices de todas las muestras
    indices = np.arange(n_samples)

    # Se seleccionan aleatoriamente initial_size indices para el conjunto inicial
    selected_indices = rng.choice(indices, size=initial_size, replace=False)
    z_indices = np.delete(indices, selected_indices)

    return selected_indices, z_indices


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    selected_indices: np.ndarray) -> dict:
    """
    Entrena un modelo de Random Forest con las muestras seleccionadas y evalúa su rendimiento en el conjunto de prueba.
    """
    
    X_selected = X_train[selected_indices]
    y_selected = y_train[selected_indices]

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf.fit(X_selected, y_selected)
    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_r, _ = pearsonr(y_test, y_pred)

    return {
        "n_samples": len(selected_indices),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_r": pearson_r
    }

def get_baseline_value(dataset_name: str, metric_name: str) -> float | None:
    """
    Devuelve el valor de una métrica del modelo base con 20 features
    para un dataset concreto.
    """
    baseline_path = f"{BASE_PATH}/baseline_{dataset_name.lower()}/metrics_top20_features.txt"

    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, value = line.split(":", 1)

                    if key.strip() == metric_name:
                        try:
                            return float(value.strip())
                        except ValueError:
                            return None
    except FileNotFoundError:
        print(f"No se ha encontrado el baseline para {dataset_name}: {baseline_path}")
        return None

    return None

def save_plot(results_df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: str, method_label: str, 
                show_baseline: bool = False, dataset_name: str | None = None ) -> None:
    """
    Guarda un gráfico de evolución de una métrica concreta.
    En el eje X se representa el número de muestras seleccionadas.
    En el eje Y se representa la métrica indicada.
    """
    
    plt.figure(figsize=(8, 5))
    plt.plot(results_df[x_col], results_df[y_col], color="green", marker="o", markersize=3, markevery=20, linewidth=1.2, label=method_label)

    if show_baseline:
        if dataset_name is None:
            raise ValueError("Para mostrar el baseline se debe indicar dataset_name.")

        baseline_value = get_baseline_value(dataset_name, y_col)

        if baseline_value is not None:
            plt.axhline(y=baseline_value, color="red", linestyle="--", linewidth=1.2, label="Baseline" )

    plt.xlabel("Número de muestras de entrenamiento")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True, linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def load_data(train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga los datos de entrenamiento y test, separa características y variable objetivo,
    y aplica RobustScaler.
    """

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    X_train = df_train.drop(columns=["activity", "ID"])
    y_train = df_train["activity"].to_numpy()

    X_test = df_test.drop(columns=["activity", "ID"])
    y_test = df_test["activity"].to_numpy()

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test