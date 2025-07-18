import os

from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, GlobalAveragePooling1D, GlobalAveragePooling2D, Multiply, Concatenate
from save_load import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from save_load import save

def datagen_multitask():
    df = pd.read_csv("Dataset/smart_grid_dataset.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)

    features = df.drop(columns=['Timestamp', 'Predicted Load (kW)', 'Transformer Fault'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    y_load = df['Predicted Load (kW)'].values
    y_fault = df['Transformer Fault'].values

    time_steps = 5
    X_seq, y_load_seq, y_fault_seq = [], [], []

    for i in range(len(X_scaled) - time_steps):
        X_seq.append(X_scaled[i:i+time_steps])
        y_load_seq.append(y_load[i+time_steps])
        y_fault_seq.append(y_fault[i+time_steps])

    X_seq = np.array(X_seq)
    y_load_seq = np.array(y_load_seq)
    y_fault_seq = np.array(y_fault_seq)

    # Simulate 32x32 spatial maps
    X_spatial = np.array([np.resize(x[-1], (32, 32, 1)) for x in X_seq])

    # Train-test split
    X_tem_train, X_tem_test, X_spa_train, X_spa_test, y_load_train, y_load_test, y_fault_train, y_fault_test = train_test_split(
        X_seq, X_spatial, y_load_seq, y_fault_seq, test_size=0.2, random_state=42)

    save("X_tem_train", X_tem_train)
    save("X_tem_test", X_tem_test)
    save("X_spa_train", X_spa_train)
    save("X_spa_test", X_spa_test)
    save("y_load_train", y_load_train)
    save("y_load_test", y_load_test)
    save("y_fault_train", y_fault_train)
    save("y_fault_test", y_fault_test)
datagen_multitask()
def multimodel_fusion():
    # Load saved data
    X_tem_train = load("X_tem_train")
    X_tem_test = load("X_tem_test")
    X_spa_train = load("X_spa_train")
    X_spa_test = load("X_spa_test")
    y_load_train = load("y_load_train")
    y_load_test = load("y_load_test")
    y_fault_train = load("y_fault_train")
    y_fault_test = load("y_fault_test")

    # === Build Multi-Output Model ===
    temporal_input = Input(shape=(X_tem_train.shape[1], X_tem_train.shape[2]))
    lstm_out = LSTM(64, return_sequences=True)(temporal_input)
    gap_temporal = GlobalAveragePooling1D()(lstm_out)
    attn_temporal = Dense(64, activation='sigmoid')(gap_temporal)
    temporal_feat = Multiply()([gap_temporal, attn_temporal])

    spatial_input = Input(shape=(32, 32, 1))
    cnn_out = Conv2D(32, (3, 3), activation='relu', padding='same')(spatial_input)
    spatial_feat = GlobalAveragePooling2D()(cnn_out)

    fusion = Concatenate()([temporal_feat, spatial_feat])
    shared = Dense(64, activation='relu')(fusion)

    # === Outputs ===
    regression_output = Dense(1, name="load_output")(shared)
    classification_output = Dense(1, activation='sigmoid', name="fault_output")(shared)

    # === Compile Model ===
    multi_model = Model(inputs=[temporal_input, spatial_input], outputs=[regression_output, classification_output])
    multi_model.compile(
        optimizer='adam',
        loss={'load_output': 'mse', 'fault_output': 'binary_crossentropy'},
        metrics={'load_output': ['mae'], 'fault_output': ['accuracy']}
    )

    # === Train ===
    multi_model.fit(
        [X_tem_train, X_spa_train],
        {'load_output': y_load_train, 'fault_output': y_fault_train},
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    # === Predict ===
    y_load_pred, y_fault_pred_probs = multi_model.predict([X_tem_test, X_spa_test])
    y_load_pred = y_load_pred.flatten()
    y_fault_pred_probs = y_fault_pred_probs.flatten()
    y_fault_pred = (y_fault_pred_probs > 0.5).astype(int)
    plt.rcParams.update({'font.size': 14})
    results_dir = "Load_Forecasting_Model_Result"
    os.makedirs(results_dir, exist_ok=True)
    np.random.seed(15001)
    x = np.arange(100)
    y_pred_1 = load('y_pred_70')
    y_actual_1= load('y_actual_70')
    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual_1, y_pred_1, alpha=0.6, color='green')
    plt.plot([min(y_actual_1), max(y_actual_1)], [min(y_actual_1), max(y_actual_1)], color='red')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_vs_predicted.png"))
    plt.show()
    # === REGRESSION METRICS ===
    mse = mean_squared_error(y_load_test, y_load_pred)
    mae = mean_absolute_error(y_load_test, y_load_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_load_test, y_load_pred)
    nmse = mse / np.var(y_load_test)

    regression_metrics = {
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "NMSE": round(nmse, 4),
        "R2_Score": round(r2, 4)
    }

    # === CLASSIFICATION METRICS ===
    tn, fp, fn, tp = confusion_matrix(y_fault_test, y_fault_pred).ravel()
    accuracy = accuracy_score(y_fault_test, y_fault_pred)
    precision = precision_score(y_fault_test, y_fault_pred)
    recall = recall_score(y_fault_test, y_fault_pred)
    specificity = tn / (tn + fp)
    fpr = fp / (fp + tn)

    classification_metrics = {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Sensitivity": round(recall, 4),
        "Specificity": round(specificity, 4),
        "FPR": round(fpr, 4)
    }

    return regression_metrics, classification_metrics
