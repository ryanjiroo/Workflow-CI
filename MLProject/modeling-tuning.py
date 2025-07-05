import os
import time
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Konfigurasi paths dan variabel
EXPERIMENT_NAME = "Air Quality Prediction - ISPU Hyperparameter Tuning"
PROCESSED_DATA_PATH = "./ispu_preprocessing/polutan_processed.csv"
LABEL_ENCODER_PATH = "./ispu_preprocessing/label_encoders.pkl"
SCALER_PATH = "./ispu_preprocessing/scaler.pkl"

FEATURES = ['pm10', 'so2', 'co', 'o3', 'no2', 'max', 'stasiun', 'critical', 'categori', 'dayofweek', 'is_weekend']
TARGETS = ['pm10', 'so2', 'co', 'o3', 'no2']

if os.getenv("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment(EXPERIMENT_NAME)

CI_MODE = os.getenv("CI", "false").lower() == "true"

# Parameter tuning grid
if CI_MODE:
    param_grid = {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [2]
    }
    n_jobs_setting = 1
else:
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    n_jobs_setting = -1

with mlflow.start_run(run_name="RandomForest_Tuned_ISPU"):
    print("Membaca dan menyiapkan data...")

    df = pd.read_csv(PROCESSED_DATA_PATH)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df['dayofweek'] = df['tanggal'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df = df.drop(columns=['tanggal'])

    X = df[FEATURES]
    y = df[TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Melakukan pencarian hyperparameter...")
    model = RandomForestRegressor(random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=n_jobs_setting, verbose=2)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    mlflow.log_params(best_params)

    preds = best_model.predict(X_test)

    print("Logging metrik per target...")
    for i, col in enumerate(TARGETS):
        y_true = y_test[col]
        y_pred = preds[:, i]
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        mlflow.log_metric(f"{col}_mse", mse)
        mlflow.log_metric(f"{col}_mae", mae)
        mlflow.log_metric(f"{col}_r2", r2)
        mlflow.log_metric(f"{col}_rmse", rmse)

    # Simpan model secara lokal dan log ke artifacts/model
    print("Menyimpan dan melog model...")
    model_path = "random_forest_model"
    mlflow.sklearn.save_model(best_model, model_path)
    mlflow.log_artifacts(model_path, artifact_path="model")

    # Log preprocessor
    if os.path.exists(SCALER_PATH):
        mlflow.log_artifact(SCALER_PATH, artifact_path="preprocessor")
    if os.path.exists(LABEL_ENCODER_PATH):
        mlflow.log_artifact(LABEL_ENCODER_PATH, artifact_path="preprocessor")

    print("Pelatihan selesai. Semua hasil dilog ke MLflow.")
