from tensorflow.keras.layers import  MaxPooling1D, Flatten
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from confusion_met import * # Assuming this is your custom confusion matrix function

def mkcnn_cls(xtrain, xtest,ytrain, ytest):
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)

    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    xtrain_cnn = np.expand_dims(xtrain, axis=2)
    xtest_cnn = np.expand_dims(xtest, axis=2)

    # CNN model
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(xtrain.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xtrain_cnn, ytrain_cat, epochs=5, batch_size=32,
              validation_split=0.2, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)
    y_pred_probs = model.predict(xtest_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(ytest_encoded, y_pred)
    met= multi_confu_matrix(ytest_encoded, y_pred)
    return y_pred, met


def lstm_ann_cls(xtrain, xtest, ytrain, ytest):
    # Label encoding + one-hot encoding
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standard scaling
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Reshape for LSTM: (samples, timesteps=1, features)
    xtrain_seq = np.expand_dims(xtrain_scaled, axis=1)
    xtest_seq = np.expand_dims(xtest_scaled, axis=1)

    # LSTM-ANN model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, xtrain.shape[1]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))   # ANN part
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))    # More ANN layers if needed
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(xtrain_seq, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    # Predict
    y_pred_probs = model.predict(xtest_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    met = multi_confu_matrix(ytest_encoded, y_pred)
    return y_pred, met


def cnn_lstm_cls(xtrain, xtest, ytrain, ytest):
    # Label encoding and one-hot transformation
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standardization
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Reshape for Conv1D-LSTM: (samples, timesteps, features)
    xtrain_seq = np.expand_dims(xtrain_scaled, axis=1)  # (samples, 1, features)
    xtest_seq = np.expand_dims(xtest_scaled, axis=1)

    # CNN-LSTM Model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, xtrain.shape[1])))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(xtrain_seq, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    y_pred_probs = model.predict(xtest_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    met = multi_confu_matrix(ytest_encoded, y_pred)
    return y_pred, met

def dnn_model_cls(xtrain, xtest, ytrain, ytest):
    # Label Encoding and One-Hot Encoding
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standard Scaling
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # DNN Model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(xtrain.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(xtrain_scaled, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    # Prediction
    y_pred_probs = model.predict(xtest_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluation
    met = multi_confu_matrix(ytest_encoded, y_pred)
    return y_pred, met
def mkcnn_reg(xtrain, xtest,ytrain, ytest):
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)

    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    xtrain_cnn = np.expand_dims(xtrain, axis=2)
    xtest_cnn = np.expand_dims(xtest, axis=2)

    # CNN model
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=(xtrain.shape[1], 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xtrain_cnn, ytrain_cat, epochs=5, batch_size=32,
              validation_split=0.2, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
              verbose=1)
    y_pred_probs = model.predict(xtest_cnn)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(ytest_encoded, y_pred)
    met= regression_metrics(ytest_encoded, y_pred)
    return y_pred, met


def lstm_ann_reg(xtrain, xtest, ytrain, ytest):
    # Label encoding + one-hot encoding
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standard scaling
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Reshape for LSTM: (samples, timesteps=1, features)
    xtrain_seq = np.expand_dims(xtrain_scaled, axis=1)
    xtest_seq = np.expand_dims(xtest_scaled, axis=1)

    # LSTM-ANN model
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, xtrain.shape[1]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))   # ANN part
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))    # More ANN layers if needed
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(xtrain_seq, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    # Predict
    y_pred_probs = model.predict(xtest_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    met = regression_metrics(ytest_encoded, y_pred)
    return y_pred, met


def cnn_lstm_reg(xtrain, xtest, ytrain, ytest):
    # Label encoding and one-hot transformation
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standardization
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # Reshape for Conv1D-LSTM: (samples, timesteps, features)
    xtrain_seq = np.expand_dims(xtrain_scaled, axis=1)  # (samples, 1, features)
    xtest_seq = np.expand_dims(xtest_scaled, axis=1)

    # CNN-LSTM Model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(1, xtrain.shape[1])))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(xtrain_seq, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    y_pred_probs = model.predict(xtest_seq)
    y_pred = np.argmax(y_pred_probs, axis=1)

    met = regression_metrics(ytest_encoded, y_pred)
    return y_pred, met

def dnn_model_reg(xtrain, xtest, ytrain, ytest):
    # Label Encoding and One-Hot Encoding
    all_labels = np.concatenate([ytrain, ytest])
    le = LabelEncoder()
    le.fit(all_labels)
    ytrain_encoded = le.transform(ytrain)
    ytest_encoded = le.transform(ytest)
    ytrain_cat = to_categorical(ytrain_encoded)
    ytest_cat = to_categorical(ytest_encoded)

    # Standard Scaling
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)

    # DNN Model
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(xtrain.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(ytrain_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit(xtrain_scaled, ytrain_cat, epochs=20, batch_size=32, validation_split=0.2,
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=1)

    # Prediction
    y_pred_probs = model.predict(xtest_scaled)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Evaluation
    met = regression_metrics(ytest_encoded, y_pred)
    return y_pred, met
