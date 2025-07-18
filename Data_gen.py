import pandas as pd
import numpy as np
from scipy.stats import skew, entropy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from save_load import save
import os

def datagen():
    # Ensure the save directory exists
    save_dir = './Saved data'
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    df = pd.read_csv("Dataset/smart_grid_dataset.csv")

    # === Step 2: Handle missing values with mean imputation ===
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # === Step 3: Normalize using Z-score ===
    features = df.drop(columns=['Timestamp', 'Predicted Load (kW)'])  # exclude label and time
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # === Step 4: Extract label ===
    y = df['Predicted Load (kW)'].values
    # === Step 5: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # === Output shapes ===
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    save('X_train_Load_Forecasting',X_train)
    save('X_test_Load_Forecasting', X_test)
    save('y_train_Load_Forecasting', y_train)
    save('y_test_Load_Forecasting',y_test)

    df = pd.read_csv("Dataset/smart_grid_dataset.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    # === Step 3: Z-score normalization (exclude Timestamp and Transformer Fault) ===
    features = df.drop(columns=['Timestamp', 'Transformer Fault'])  # all except time and label
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    # === Step 4: Extract label ===
    y = df['Transformer Fault'].values
    # === Step 5: Train-test split ===
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
    save('X_train_Fault_Detection', X_train)
    save('X_test_Fault_Detection', X_test)
    save('y_train_Fault_Detection', y_train)
    save('y_test_Fault_Detection', y_test)


