import os
import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dropout, Layer, Dense, BatchNormalization, Bidirectional, GRU, LayerNormalization
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras import backend as K
import tensorflow as tf

# ==== Project Imports =====================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.data_ingestion_helper import ingest_data
from Components.preprocessing import preprocess_data
from Config.config_loader import load_config
from Logging.logging_config import setup_logging
from Utils.backtest_helper import backtest_regression_simple_centered_v4


# =======================================================================
# CUSTOM LAYERS
# =======================================================================

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
        context = K.sum(context, axis=1)
        return context


# ====== FIX: Missing imports for MHSA layer ==============================
from tensorflow.keras.layers import Dense, Dropout


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(Layer):
    def __init__(self, num_heads=4, head_dim=32, dropout_rate=0.1, use_causal_mask=True, **kwargs):
        """
        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension of each head (proj_dim = num_heads * head_dim).
            dropout_rate: Dropout probability.
            use_causal_mask: If True, applies a look-ahead mask (critical for forecasting).
        """
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.proj_dim = num_heads * head_dim
        self.dropout_rate = dropout_rate
        self.use_causal_mask = use_causal_mask

    def build(self, input_shape):
        feature_dim = input_shape[-1]

        # Linear Projections for Query, Key, Value
        self.Wq = self.add_weight(name="Wq", shape=(feature_dim, self.proj_dim),
                                  initializer="glorot_uniform")
        self.Wk = self.add_weight(name="Wk", shape=(feature_dim, self.proj_dim),
                                  initializer="glorot_uniform")
        self.Wv = self.add_weight(name="Wv", shape=(feature_dim, self.proj_dim),
                                  initializer="glorot_uniform")

        # Output Projection
        self.dense = Dense(feature_dim)

        # Dropout Layers
        self.att_dropout = Dropout(self.dropout_rate)
        self.output_dropout = Dropout(self.dropout_rate)

        super(MultiHeadSelfAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        # Reshape to (Batch, Seq_Len, Num_Heads, Head_Dim)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        # Transpose to (Batch, Num_Heads, Seq_Len, Head_Dim)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # 1. Project and Split Heads
        Q = self.split_heads(tf.matmul(x, self.Wq), batch_size)
        K = self.split_heads(tf.matmul(x, self.Wk), batch_size)
        V = self.split_heads(tf.matmul(x, self.Wv), batch_size)

        # 2. Scaled Dot-Product Attention
        # Shape: (Batch, Heads, Seq_Len, Seq_Len)
        score = tf.matmul(Q, K, transpose_b=True)

        # Scale scores to stabilize gradients
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        score = score / scale

        # 3. Apply Causal Mask (Look-ahead Mask)
        if self.use_causal_mask:
            # Create a lower triangular matrix of ones (1s in past/present, 0s in future)
            ones = tf.ones((seq_len, seq_len))
            mask = tf.linalg.band_part(ones, -1, 0)  # Keep lower triangle

            # Invert: 0s in past, 1s in future
            mask = 1.0 - mask

            # Add huge negative number to future positions so Softmax makes them 0
            # Shape broadcasting: (1, 1, Seq, Seq)
            mask = mask[tf.newaxis, tf.newaxis, :, :]
            score += (mask * -1e9)

        # 4. Softmax & Dropout
        weights = tf.nn.softmax(score, axis=-1)
        if training:
            weights = self.att_dropout(weights, training=training)

        # 5. Weighted Sum of Values
        attention_output = tf.matmul(weights, V)

        # 6. Concatenate Heads
        # Transpose back to (Batch, Seq_Len, Num_Heads, Head_Dim)
        attention_output = tf.transpose(attention_output, perm=(0, 2, 1, 3))
        # Flatten to (Batch, Seq_Len, Proj_Dim)
        concat = tf.reshape(attention_output, (batch_size, -1, self.proj_dim))

        # 7. Final Projection
        output = self.dense(concat)
        if training:
            output = self.output_dropout(output, training=training)

        return output[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "use_causal_mask": self.use_causal_mask
        })
        return config

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

def tcn_block_safe(*args, **kwargs):
    return tcn_block(*args, **kwargs)

# =======================================================================
# MULTI-MODEL BACKTEST SCRIPT
# =======================================================================
def main():

    config = load_config()
    logger = setup_logging(config['logging']['log_file_path'])
    logger.info("=== MULTI MODEL BACKTEST STARTED ===")

    df = pd.read_csv(r'C:/Users/arunm/Documents/Projects/Trading-App/Data/Raw/Pipeline_raw.csv')

    preprocessed_df = preprocess_data(df)
    preprocessed_df.dropna(inplace=True)

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    preprocessed_df[['Open', 'High', 'Low', 'Volume']] = feature_scaler.fit_transform(
        preprocessed_df[['Open', 'High', 'Low', 'Volume']]
    )
    preprocessed_df['Close'] = target_scaler.fit_transform(preprocessed_df[['Close']])

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    X_test = preprocessed_df[features]

    SEQ_LENGTH = 60
    FORECAST_LENGTH = 25

    def create_sequences(df, seq_length, forecast_length):
        X, y = [], []
        for i in range(len(df) - seq_length - forecast_length):
            seq = df.iloc[i:i + seq_length].values
            tgt = df['Close'].iloc[i + seq_length:i + seq_length + forecast_length]
            X.append(seq)
            y.append(tgt)
        return X, y

    X, y = create_sequences(X_test, SEQ_LENGTH, FORECAST_LENGTH)
    X = np.array(X, dtype=np.float32)


    # ===================================================================
    # FIXED: ADD MHSA to CUSTOM OBJECTS
    # ===================================================================
    custom_objects = {
        "Attention": Attention,
        "MultiHeadSelfAttention": MultiHeadSelfAttention,
        "Conv1D": Conv1D,
        "Dropout": Dropout,
        "Dense": Dense,
        "LayerNormalization": LayerNormalization,
        "BatchNormalization": BatchNormalization,
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
        "tcn_block": tcn_block_safe
    }


    # ===================================================================
    # 8 MODELS
    # ===================================================================
    MODEL_PATHS = [
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/lstm_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/bilstm_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/gru_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/tcn_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/lstm_mhsa_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/bilstm_mhsa_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/gru_mhsa_model.h5",
        r"C:/Users/arunm/Documents/Projects/Trading-App/Components/models/tcn_mhsa_model.h5",
    ]


    results_list = []

    for model_path in MODEL_PATHS:
        model_name = os.path.basename(model_path)
        logger.info(f"=== Running model: {model_name} ===")

        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=False)

            y_pred = model.predict(X)
            y_pred_flat = y_pred.reshape(-1, 1)
            unscaled = target_scaler.inverse_transform(y_pred_flat)
            y_pred_unscaled = unscaled.reshape(y_pred.shape)

            results = backtest_regression_simple_centered_v4(
                df=df,
                y_pred=y_pred_unscaled,
                config=config,
                horizon=50,
                quantile=0.98,
                max_hold=50,
                invert_signal=True,
                side_mode="both",
                max_loss_cap=-3.0,
                decay_factor=0.1,
            )

            results_list.append({
                "model": model_name,
                "final_balance": results["final_balance"],
                "profit_pct": results["profit_pct"],
                "trades": len(results["trades"]),
                "win_rate": results["diagnostics"]["win_rate"],
                "avg_pnl": results["diagnostics"]["avg_pnl"],
                "max_dd": results["max_drawdown"],
            })

        except Exception as e:
            logger.error(f"Error running model {model_name}: {e}")


    results_df = pd.DataFrame(results_list)
    results_df.sort_values(by="profit_pct", ascending=False, inplace=True)

    save_dir = r"C:\Users\arunm\Documents\Projects\Trading-App\Data\Results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model_backtest_results.csv")
    results_df.to_csv(save_path, index=False)

    print(results_df)


if __name__ == "__main__":
    main()
