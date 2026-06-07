# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
modelling.py
============
Kriteria 2 & 3 - Melatih model machine learning dengan MLflow autolog
Nama    : Whenny Zenica
Dataset : melb_preprocessed.csv (Melbourne Housing Dataset)
Model   : Linear Regression (Baseline)
"""

import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi MLflow — tracking URI dari environment variable (CI) atau lokal
EXPERIMENT_NAME = "Eksperimen_Prediksi_Harga_Melbourne_Whenny"

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)

# Membaca dataset
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "melb_preprocessed.csv")

if not os.path.exists(DATA_PATH):
    print(f"Error: Berkas '{DATA_PATH}' tidak ditemukan!")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"Dataset Loaded : {df.shape[0]} baris x {df.shape[1]} kolom")

# Memisahkan Fitur (X) dan Target (y)
X = df.drop(columns=["Price"])
y = df["Price"]

print(f"Fitur Proyek    : {list(X.columns)}")

# Pembagian Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"Ukuran Data Train : {X_train.shape[0]}")
print(f"Ukuran Data Test  : {X_test.shape[0]}")

# MLflow Autolog
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    silent=False,
)

# Memulai Sesi Perekaman Eksperimen
with mlflow.start_run(run_name="LinearRegression_Baseline") as run:
    print(f"\n[MLflow] Run ID     : {run.info.run_id}")
    print(f"[MLflow] Experiment : {EXPERIMENT_NAME}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2",  r2)

    print(f"\nTest MAE      : {mae:.2f}")
    print(f"Test R2 Score : {r2:.4f}")

print("\n" + "="*60)
print("MODEL BASELINE BERHASIL DIEKSEKUSI & DIKUNCI DI MLRUNS!")
print("="*60)
