import time
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler
from samplingUtils import save_plot, train_and_evaluate, initialize_sampling, load_data

RANDOM_STATE = 0
N_ESTIMATORS = 100
DATASETS = ["HLM", "MDR1_ER", "RLM", "Sol", "hPPB", "rPPB"]
METRICS_TO_PLOT = [("mse", "MSE"), ("pearson_r", "Pearson r"), ("mae", "MAE"), ("r2", "R2")]


def random_sampling(X_train: np.ndarray,y_train: np.ndarray,X_test: np.ndarray,y_test: np.ndarray,
                    initial_size: int, batch_size: int) -> pd.DataFrame:

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    rng = np.random.default_rng(RANDOM_STATE)

    selected_indices, pool_indices = initialize_sampling(n_samples=len(X_train), initial_size=initial_size)

    results = []

    while len(pool_indices) > 0:
        metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=selected_indices)
        results.append(metrics)

        current_batch_size = min(batch_size, len(pool_indices))
        new_indices = rng.choice(pool_indices, size=current_batch_size, replace=False)

        selected_indices = np.concatenate((selected_indices, new_indices))
        pool_indices = np.setdiff1d(pool_indices, new_indices, assume_unique=False)

    metrics = train_and_evaluate(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, selected_indices=selected_indices)
    results.append(metrics)

    return pd.DataFrame(results)


def run_random_for_dataset(dataset_name: str, train_path: str, test_path: str, output_dir: str, initial_size: int, batch_size: int) -> None:

    X_train_scaled, y_train, X_test_scaled, y_test = load_data(train_path=train_path, test_path=test_path)

    results_df = random_sampling( X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, initial_size=initial_size,
                                    batch_size=batch_size )


    results_df.to_csv(os.path.join(output_dir, "random_sampling_top20_features.csv"), index=False)

    for col, name in METRICS_TO_PLOT:
        save_plot(results_df, x_col="n_samples", y_col=col, title=f"Random Sampling - {name} ({dataset_name})", output_path=f"{output_dir}/plot_{col}.png", method_label="Random sampling")

def main() -> None:
    for dataset_name in DATASETS:
        train_path = f"../data/reduced_datasets/ADME_{dataset_name}_train_feat_top20.csv"
        test_path = f"../data/reduced_datasets/ADME_{dataset_name}_test_feat_top20.csv"

        output_dir = f"../experiments/randomSampling/random_sampling_{dataset_name.lower()}"
        os.makedirs(output_dir, exist_ok=True)

        run_random_for_dataset(dataset_name=dataset_name, train_path=train_path, test_path=test_path, output_dir=output_dir,
            initial_size=50, batch_size=5)


if __name__ == "__main__":
    main()