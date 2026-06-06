# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Skrip Pemodelan & Tracking MLflow - Proyek Akhir
Nama: Whenny Zenica
Dataset: Melbourne Housing Snapshot
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 1. Menangkap parameter dinamis dari CLI / file MLProject
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=15)
    args = parser.parse_args()

    print("="*50)
    print("MEMULAI EVALUASI MODEL & TRACKING MLFLOW")
    print("="*50)

    # 2. Jalur data disesuaikan dengan folder lokal di VS Code
    data_path = 'melb_preprocessed.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File tidak ditemukan di {data_path}")
        print("Pastikan file melb_preprocessed.csv sudah dipindahkan ke folder melb_data_preprocessing/")
        sys.exit(1)
        
    df_model = pd.read_csv(data_path)
    print(f"Dataset preprocessed berhasil dimuat. Shape: {df_model.shape}")

    # Memisahkan Fitur dan Target
    X = df_model.drop(columns=['Price'])
    y = df_model['Price']

    # Splitting Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Ukuran Data Training: {X_train.shape}")
    print(f"Ukuran Data Testing: {X_test.shape}")

    # Mengatur Nama Eksperimen di MLflow
    mlflow.set_experiment("Eksperimen_Prediksi_Harga_Melbourne_Whenny")

    # ==========================================
    # RUN 1: Regresi Linear (Baseline Model)
    # ==========================================
    print("\n" + "-"*40)
    print("Menjalankan Run 1: Linear Regression")
    print("-"*40)

    with mlflow.start_run(run_name="Linear_Regression_Baseline"):
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        y_pred_lr = lr_model.predict(X_test)

        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        r2_lr = r2_score(y_test, y_pred_lr)

        # Logging ke MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_metric("MAE", mae_lr)
        mlflow.log_metric("RMSE", rmse_lr)
        mlflow.log_metric("R2_Score", r2_lr)
        mlflow.sklearn.log_model(lr_model, "linear_regression_model")

        print(f"Linear Regression - MAE     : {mae_lr:.2f}")
        print(f"Linear Regression - RMSE    : {rmse_lr:.2f}")
        print(f"Linear Regression - R2 Score: {r2_lr:.4f}")

    # ==========================================
    # RUN 2: Random Forest Regressor (Hyperparameter Dinamis)
    # ==========================================
    print("\n" + "-"*40)
    print("Menjalankan Run 2: Random Forest Regressor")
    print("-"*40)

    with mlflow.start_run(run_name="Random_Forest_Regressor"):
        # Menggunakan nilai parameter hasil inputan CLI / MLProject
        rf_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)

        mae_rf = mean_absolute_error(y_test, y_pred_rf)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mse_rf)
        r2_rf = r2_score(y_test, y_pred_rf)

        # Logging ke MLflow
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("MAE", mae_rf)
        mlflow.log_metric("RMSE", rmse_rf)
        mlflow.log_metric("R2_Score", r2_rf)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")

        print(f"Random Forest - MAE     : {mae_rf:.2f}")
        print(f"Random Forest - RMSE    : {rmse_rf:.2f}")
        print(f"Random Forest - R2 Score: {r2_rf:.4f}")

    print("\n" + "="*50)
    print("EVALUASI MODEL & TRACKING MLFLOW SELESAI!")
    print("="*50)

if __name__ == "__main__":
    main()