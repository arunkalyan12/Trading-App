import sys
import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, LSTM, GRU, Bidirectional, Dropout, Dense, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.model_helper import load_and_scale, create_sequences, split_data, train_model

# -------------------------------------------------------
# Custom Attention Layer
# -------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        return K.sum(context, axis=1)

# -------------------------------------------------------
# Multi-Head Self Attention Layer
# -------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads=4, head_dim=32, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        self.Wq = self.add_weight(
            shape=(feature_dim, self.proj_dim), initializer="glorot_uniform", name="Wq"
        )
        self.Wk = self.add_weight(
            shape=(feature_dim, self.proj_dim), initializer="glorot_uniform", name="Wk"
        )
        self.Wv = self.add_weight(
            shape=(feature_dim, self.proj_dim), initializer="glorot_uniform", name="Wv"
        )

        self.dense = Dense(feature_dim)

        super(MultiHeadSelfAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=(0, 2, 1, 3))  # (B, H, T, D)

    def call(self, x):
        batch_size = tf.shape(x)[0]

        Q = tf.matmul(x, self.Wq)
        K = tf.matmul(x, self.Wk)
        V = tf.matmul(x, self.Wv)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        score = tf.matmul(Q, K, transpose_b=True)
        score = score / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        weights = tf.nn.softmax(score, axis=-1)

        attention_output = tf.matmul(weights, V)  # (B, H, T, D)

        attention_output = tf.transpose(attention_output, perm=(0, 2, 1, 3))
        concat = tf.reshape(attention_output, (batch_size, -1, self.proj_dim))

        return self.dense(concat)  # project back to model dimension

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        })
        return config

# -------------------------------------------------------
# Base CNN Block
# -------------------------------------------------------

def cnn_block(x):
    for _ in range(3):
        x = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.3)(x)
    return x


# -------------------------------------------------------
# TCN BLOCK
# -------------------------------------------------------

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


# -------------------------------------------------------
# Model Variants
# -------------------------------------------------------

def build_lstm_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    for _ in range(3):
        x = LSTM(200, return_sequences=True)(x)
        x = Dropout(0.3)(x)

    x = Attention()(x)
    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(),
                 tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


def build_bilstm_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    for _ in range(3):
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)

    x = Attention()(x)
    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(),
                 tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


def build_gru_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    for _ in range(3):
        x = GRU(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)

    x = Attention()(x)
    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(),
                 tf.keras.metrics.RootMeanSquaredError()]
    )
    return model


# -------------------------------------------------------
# NEW: TCN + ATTENTION MODEL
# -------------------------------------------------------

def build_tcn_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)

    # temporal CNN stack with dilation
    x = tcn_block(inputs, filters=128, kernel_size=3, dilations=[1, 2, 4, 8, 16])

    # attention over the resulting sequence
    x = Attention()(x)

    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(),
                 tf.keras.metrics.RootMeanSquaredError()]
    )

    return model

def build_lstm_mhsa_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=True)(x)
    x = Dropout(0.3)(x)

    x = MultiHeadSelfAttention(
        num_heads=4,
        head_dim=32,
        dropout_rate=0.1,
        use_causal_mask=True  # Ensures no future peeking
    )(x)

    outputs = Dense(FORECAST_LENGTH)(attention)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()])

    return model

def build_bilstm_mhsa_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    for _ in range(3):
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(0.3)(x)

    x = MultiHeadSelfAttention(num_heads=4, head_dim=32)(x)
    x = tf.reduce_mean(x, axis=1)

    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_gru_mhsa_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)
    x = cnn_block(inputs)

    for _ in range(3):
        x = GRU(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)

    x = MultiHeadSelfAttention(num_heads=4, head_dim=32)(x)
    x = tf.reduce_mean(x, axis=1)

    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.RootMeanSquaredError()])
    return model

def build_tcn_mhsa_model(input_shape, forecast_len):
    inputs = Input(shape=input_shape)

    x = tcn_block(inputs, filters=128, kernel_size=3, dilations=[1, 2, 4, 8, 16])

    x = MultiHeadSelfAttention(num_heads=4, head_dim=32)(x)

    x = tf.reduce_mean(x, axis=1)

    outputs = Dense(forecast_len)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(),
                           tf.keras.metrics.RootMeanSquaredError()])
    return model


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------
DATA_PATH = r"Data\Preprocessed\Preprocessed.csv"     # change to your path
SEQ_LENGTH = 128
FORECAST_HORIZON = 5
VAL_SPLIT = 0.1
EPOCHS = 30
BATCH_SIZE = 64


def run_all_models():
    # -------------------------------------------------------
    # Load & Preprocess Data
    # -------------------------------------------------------
    print("Loading + Scaling dataset...")
    df, feature_scaler, target_scaler = load_and_scale(DATA_PATH)
    print("Finished scaling.")

    print("Building sequences...")
    X, y = create_sequences(df, SEQ_LENGTH, FORECAST_HORIZON)
    print(f"X shape = {X.shape}, y shape = {y.shape}")

    print("Splitting train/validation...")
    X_train, X_val, y_train, y_val = split_data(X, y, val_ratio=VAL_SPLIT)

    print("Data ready.\n")

    # -------------------------------------------------------
    # List of models to run
    # -------------------------------------------------------
    model_list = [
        ("LSTM", build_lstm_model),
        ("BiLSTM", build_bilstm_model),
        ("GRU", build_gru_model),
        ("TCN", build_tcn_model),
        ("LSTM+MHSA", build_lstm_mhsa_model),
        ("BiLSTM+MHSA", build_bilstm_mhsa_model),
        ("GRU+MHSA", build_gru_mhsa_model),
        ("TCN+MHSA", build_tcn_mhsa_model)
    ]

    # -------------------------------------------------------
    # Run sequentially
    # -------------------------------------------------------
    for model_name, model_builder in model_list:
        print("\n===================================================")
        print(f" Running Model: {model_name} ")
        print("===================================================\n")

        # Build model
        model = model_builder(
            input_shape=X_train.shape[1:],
            forecast_len=FORECAST_HORIZON
        )

        # Train model
        history, trained_model = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )

        print(f"\n{model_name} Training Completed.")
        print("===================================================\n")

        # YOU will add model.save() inside builders (as you said)


if __name__ == "__main__":
    run_all_models()