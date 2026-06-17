# Parámetros Random Forest
RANDOM_STATE = 0
N_ESTIMATORS = 100

# Parámetros de muestreo
INITIAL_SIZE = 50
BATCH_SIZE = 5
DISTANCE_METRIC = "euclidean"

# Datasets utilizados
DATASETS = ["HLM", "MDR1_ER", "RLM", "Sol", "hPPB", "rPPB"]
METRICS_TO_PLOT = [("mse", "MSE"), ("pearson_r", "Pearson r"), ("mae", "MAE"), ("r2", "R2")]

# Rutas donde se guardan los resultados de los diferentes modelos
BASE_PATH: str = "../results/baseline"
RS_PATH: str = "../results/randomSampling"
GS_PATH: str = "../results/greedySampling"
GS_MIN_DIST_PATH: str = "../results/greedySampling_minDist"
KMS_PATH: str = "../results/k_medoidsSampling"
COMPARISON_PATH: str = "../results/comparison"