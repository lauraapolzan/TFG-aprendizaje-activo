import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
from config import (RANDOM_STATE, N_ESTIMATORS, DATASETS, BASE_PATH)
from baselineUtils import (save_metrics, plot_feature_importance, save_top_features_csv,
                            save_reduced_datasets, plot_mean_feature_count_metric_with_baseline,
                            create_global_feature_summary, plot_feature_count_metric)

# Parámetros de selección de características
TOP_N_FEATURES = 20
NUM_FEATURES = [5, 10, 15, 20, 25, 30]


def train_and_evaluate_random_forest(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[RandomForestRegressor, dict]:
    """
    Entrena y evalúa un modelo de Random Forest con los datos proporcionados.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        y_train: etiquetas del conjunto de entrenamiento.
        X_test: conjunto de datos de prueba.
        y_test: etiquetas del conjunto de prueba.

    Devuelve:
        rf: modelo de Random Forest entrenado.
        metrics: diccionario con las métricas de evaluación.
    """
    
    #Escalado de los datos
    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Se entrena el RF
    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    # Se calculan y guardan las métricas
    pearson_r, _ = pearsonr(y_test, y_pred)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "pearson_r": pearson_r,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }

    return rf, metrics


def evaluate_numFeatures(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, ranked_features: list[str], feature_counts: list[int]) -> list[dict]:
    """
    Evalúa el modelo con diferentes números de características.

    Parámetros:
        X_train: conjunto de datos de entrenamiento.
        y_train: etiquetas del conjunto de entrenamiento.
        X_test: conjunto de datos de prueba.
        y_test: etiquetas del conjunto de prueba.
        ranked_features: lista de características ordenadas por importancia.
        feature_counts: lista de números de características a evaluar.

    Devuelve:
        results: lista de diccionarios con las métricas para cada número de características.
    """
    results = []

    for n_features in feature_counts:
        selected_features = ranked_features[:n_features]
        X_train_reduced = X_train[selected_features]
        X_test_reduced = X_test[selected_features]

        _, metrics_reduced = train_and_evaluate_random_forest(X_train_reduced, y_train, X_test_reduced, y_test)

        metrics_reduced["n_features"] = n_features
        results.append(metrics_reduced)

    return results


def run_baseline_for_dataset(dataset_name: str) -> None:
    """
    Ejecuta todo el proceso del modelo base para un dataset concreto.
    Lee los datos, entrena el modelo con todas las características, guarda métricas y gráficos,
    selecciona las 20 características más importantes, guarda datasets reducidos y vuelve a entrenar el modelo.
    """
    
    # Se cargan los datos
    train_path = f"../data/ADME_{dataset_name}_train_feat.csv"
    test_path = f"../data/ADME_{dataset_name}_test_feat.csv"

    df_train = pd.read_csv(train_path, low_memory=False)
    df_test = pd.read_csv(test_path, low_memory=False)

    X_train = df_train.drop(columns=["activity", "ID"])
    y_train = df_train["activity"]

    X_test = df_test.drop(columns=["activity", "ID"])
    y_test = df_test["activity"]

    # Se creaan las carpetas para los experimentos
    output_dir = f"{BASE_PATH}/baseline_{dataset_name.lower()}"
    os.makedirs(output_dir, exist_ok=True)

    # Entrenar modelo base con todas las características
    rf_full, metrics_full = train_and_evaluate_random_forest(X_train, y_train, X_test, y_test)

    metrics_full["dataset"] = dataset_name
    save_metrics(metrics_full, os.path.join(output_dir, "metrics_all_features.txt"))

    plot_feature_importance(importances=rf_full.feature_importances_, feature_names=X_train.columns.tolist(), output_path=os.path.join(output_dir, "feature_importance_top20.png"), top_n=TOP_N_FEATURES)

    # Ranking de features según importancia
    importance_df = pd.DataFrame({"feature": X_train.columns.tolist(), "importance": rf_full.feature_importances_}).sort_values("importance", ascending=False)

    ranked_features = importance_df["feature"].tolist()

    # Probar distintos números de features
    feature_count_results = evaluate_numFeatures(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, ranked_features=ranked_features, feature_counts=NUM_FEATURES)

    for result in feature_count_results:
        result["dataset"] = dataset_name

    # Guardar resultados de la evaluación de distintos números de features
    df_results = pd.DataFrame(feature_count_results)
    df_results.to_csv(os.path.join(output_dir, "feature_count_results.csv"), index=False)
    
    plot_feature_count_metric(feature_count_results, dataset_name, metric="rmse", output_path=os.path.join(output_dir, "feature_count_rmse.png"))

    plot_feature_count_metric(feature_count_results, dataset_name, metric="r2", output_path=os.path.join(output_dir, "feature_count_r2.png"))

    # Guardar top 20 features en CSV
    top_features_path = os.path.join(output_dir, "top_20_features.csv")
    top_features = save_top_features_csv(importances=rf_full.feature_importances_, feature_names=X_train.columns.tolist(), output_path=top_features_path, top_n=TOP_N_FEATURES)

    # Guardar datasets reducidos en carpeta data/reduced_datasets
    reduced_train_path, reduced_test_path = save_reduced_datasets(df_train=df_train, df_test=df_test, selected_features=top_features, dataset_name=dataset_name)

    # Volver a entrenar solo con las 20 features seleccionadas
    X_train_reduced = df_train[top_features]
    X_test_reduced = df_test[top_features]

    _, metrics_reduced = train_and_evaluate_random_forest(X_train_reduced, y_train, X_test_reduced, y_test)

    metrics_reduced["dataset"] = dataset_name
    metrics_reduced["reduced_train_path"] = reduced_train_path
    metrics_reduced["reduced_test_path"] = reduced_test_path

    save_metrics(metrics_reduced, os.path.join(output_dir, "metrics_top20_features.txt"))
    pd.DataFrame([metrics_reduced]).to_csv(os.path.join(output_dir, "metrics_top20_features.csv"), index=False)


def main() -> None:
    for dataset_name in DATASETS:
        run_baseline_for_dataset(dataset_name)

    create_global_feature_summary()
    plot_mean_feature_count_metric_with_baseline(metric="mse")
    plot_mean_feature_count_metric_with_baseline(metric="r2")
    
if __name__ == "__main__":
    main()