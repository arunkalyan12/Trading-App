import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# -------------------------------------------------------
# Data Preprocessing
# -------------------------------------------------------

def load_and_scale(path):
    df = pd.read_csv(path)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df[['Open', 'High', 'Low', 'Volume']] = feature_scaler.fit_transform(
        df[['Open', 'High', 'Low', 'Volume']]
    )
    df['Close'] = target_scaler.fit_transform(df[['Close']])

    return df, feature_scaler, target_scaler


# -------------------------------------------------------
# Sequence Builder
# -------------------------------------------------------

def create_sequences(df, seq_length, forecast_len, n_jobs=-1):
    def process(i):
        seq = df.iloc[i:i + seq_length].copy()
        target = df['Close'].iloc[i + seq_length:i + seq_length + forecast_len].values
        return seq, target

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process)(i)
        for i in range(len(df) - seq_length - forecast_len)
    )
    X, y = zip(*results)

    X = np.array([seq[['Open', 'High', 'Low', 'Volume']].values for seq in X])
    y = np.array(y)

    return X, y


# -------------------------------------------------------
# Splitting
# -------------------------------------------------------

def split_data(X, y, val_ratio=0.1):
    return train_test_split(X, y, test_size=val_ratio, shuffle=False)


# -------------------------------------------------------
# Training Helper
# -------------------------------------------------------

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64):
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.001
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr]
    )

    return history, model
