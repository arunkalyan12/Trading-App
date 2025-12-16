import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dropout, Layer, Dense, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

#Read data from "preprocessing.csv"
data = pd.read_csv("Preprocessed.csv")

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Separate scalers for features and target
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Normalize the feature columns
data[['Open', 'High', 'Low', 'Volume']] = feature_scaler.fit_transform(
    data[['Open', 'High', 'Low', 'Volume']]
)

# Normalize the target column
data['Close'] = target_scaler.fit_transform(data[['Close']])



from joblib import Parallel, delayed

def create_sequences(df, seq_length, forecast_length, n_jobs=-1):
    def process(i):
        seq = df.iloc[i:i + seq_length].copy()
        target = df['Close'].iloc[i + seq_length:i + seq_length + forecast_length].values
        return seq, target

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(process)(i)
        for i in range(len(df) - seq_length - forecast_length)
    )
    return results

# Usage
SEQ_LENGTH = 60
FORECAST_LENGTH = 25
sequences = create_sequences(data, SEQ_LENGTH, FORECAST_LENGTH)

# Split into X and y
X, y = zip(*sequences)

X = np.array([
    seq[['Open', 'High', 'Low', 'Close', 'Volume']].values
    for seq in X
])

y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)


del data, sequences, X, y

# Define the Attention layer
@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

def cnn_block(x):
    for _ in range(3):
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.3)(x)
    return x

def tcn_block(x, filters=64, kernel_size=3, dilations=[1, 2, 4, 8]):
    for dilation in dilations:
        res = x
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu"
        )(x)
        x = Dropout(0.25)(x)

        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu"
        )(x)

        # Residual connection
        if res.shape[-1] != x.shape[-1]:
            res = Conv1D(filters, kernel_size=1)(res)

        x = x + res
        x = Dropout(0.25)(x)
    return x


def build_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    attention = Attention()(x)

    outputs = Dense(FORECAST_LENGTH)(attention)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

model = build_lstm_model((SEQ_LENGTH, 5))
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.001
)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save('lstm_model.h5')

# Evaluate the model
mae = model.evaluate(X_val, y_val)[1]
rmse = model.evaluate(X_val, y_val)[2]
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")


# Build the model
def build_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    attention = Attention()(x)

    outputs = Dense(FORECAST_LENGTH)(attention)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

model = build_bilstm_model((SEQ_LENGTH, 5))
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.001
)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save('bilstm_model.h5')

# Evaluate the model
mae = model.evaluate(X_val, y_val)[1]
rmse = model.evaluate(X_val, y_val)[2]
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    # Replace LSTM with Bidirectional GRU layers
    x = GRU(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = GRU(32, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    # Attention layer
    attention = Attention()(x)

    outputs = Dense(FORECAST_LENGTH)(attention)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])
    return model

model = create_model((SEQ_LENGTH, 5))
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=15,
    min_lr=0.001
)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val),
                    callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save('gru_model.h5')

# Evaluate the model
mae = model.evaluate(X_val, y_val)[1]
rmse = model.evaluate(X_val, y_val)[2]
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")

def create_tcn_model(input_shape):
    inputs = Input(shape=input_shape)

    x = cnn_block(inputs)

    # ---- TCN Block (stacked dilated convolutions + residuals) ----
    x = tcn_block(
        x,
        filters=64,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16]   # very important for TCN!
    )

    # ---- second TCN stage (deeper model) ----
    x = tcn_block(
        x,
        filters=32,
        kernel_size=3,
        dilations=[1, 2, 4, 8]
    )

    # ---- Attention Layer ----
    attention_out = Attention()(x)

    # ---- Output Layer (forecast next N timesteps) ----
    outputs = Dense(FORECAST_LENGTH)(attention_out)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(),
            tf.keras.metrics.RootMeanSquaredError()
        ]
    )

    return model


# Build model
model = create_tcn_model((SEQ_LENGTH, 5))
model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr]
)

# Save
model.save('tcn_model.h5')

# Evaluate
mae = model.evaluate(X_val, y_val)[1]
rmse = model.evaluate(X_val, y_val)[2]
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")