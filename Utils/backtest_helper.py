import numpy as np
import pandas as pd


def backtest_regression_simple_centered_v4(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    config,
    horizon: int = 50,
    quantile: float = 0.90,
    max_hold: int = None,
    invert_signal: bool = True,
    side_mode: str = "both",   # "both", "long_only", "short_only"
    max_loss_cap: float = -3.5,
    decay_factor: float = 0.5,  # exit when |pred_now| < decay_factor * |pred_at_entry|
):
    """
    Simple but enhanced regression backtester:

      - Centered predicted returns
      - Quantile threshold -> signals
      - Optional inversion (model anti-directional)
      - Optional side filter: both / long_only / short_only
      - Fixed-horizon exit + predictive exits:
          * HARD_STOP (pnl <= max_loss_cap)
          * MODEL_FLIP (pred sign flips vs entry)
          * DECAY (pred magnitude collapses vs entry)
          * TIME (max_hold reached)
      - Diagnostics for analysis

    All trades are implemented as long-only in PnL math (position > 0),
    but side_mode controls which signal directions are allowed to enter.
    """

    df = df.copy()

    # -----------------------------
    # 1. Align predictions
    # -----------------------------
    if y_pred.ndim > 1:
        # Average first `horizon` steps if shape is (N, H)
        future_pred = np.mean(y_pred[:, :horizon], axis=1)
    else:
        future_pred = y_pred

    # Align df to prediction length
    df = df.iloc[-len(future_pred):].copy()
    df["future_pred"] = future_pred

    # -----------------------------
    # 2. Predicted return (raw)
    # -----------------------------
    df["pred_ret_raw"] = (df["future_pred"] - df["Close"]) / df["Close"]

    # Model seems anti-directional → invert by default
    if invert_signal:
        df["pred_ret_raw"] = -df["pred_ret_raw"]

    # -----------------------------
    # 3. Centering (CRITICAL)
    # -----------------------------
    center_lb = 200  # rolling median window
    df["center"] = df["pred_ret_raw"].rolling(center_lb).median().fillna(0.0)
    df["pred_ret"] = df["pred_ret_raw"] - df["center"]

    # -----------------------------
    # 4. Quantile threshold on centered predictions
    # -----------------------------
    thr = df["pred_ret"].abs().quantile(quantile)
    if (not np.isfinite(thr)) or thr == 0:
        thr = df["pred_ret"].abs().mean()  # fallback

    df["signal"] = 0
    df.loc[df["pred_ret"] > thr, "signal"] = 1
    df.loc[df["pred_ret"] < -thr, "signal"] = -1

    # Only enter when signal changes (avoid spam)
    df["entry"] = (df["signal"] != 0) & df["signal"].ne(df["signal"].shift(1))

    # -----------------------------
    # 5. Backtest core
    # -----------------------------
    if max_hold is None:
        max_hold = horizon

    initial_balance = config["backtesting"]["initial_balance"]
    risk_pct = config["risk_management"]["risk_percentage"]

    balance = initial_balance
    position = 0.0
    entry_price = 0.0
    hold = 0

    entry_signal = None
    entry_pred = None

    trades: list[dict] = []
    equity_curve: list[float] = []

    for i, row in df.iterrows():
        price = float(row["Close"])
        pred_now = float(row["pred_ret"])
        equity_curve.append(balance + position * price)

        # --------------------------------
        # Decide if this signal side is allowed
        # --------------------------------
        side_ok = True
        if side_mode == "long_only":
            side_ok = (row["signal"] == 1)
        elif side_mode == "short_only":
            side_ok = (row["signal"] == -1)

        # --------------------------------
        # ENTRY
        # --------------------------------
        if position == 0 and row["entry"] and side_ok:
            size = balance * (risk_pct / 100.0)
            if size <= 0:
                continue

            position = size / price
            balance -= size
            entry_price = price
            hold = 0
            entry_signal = row["signal"]
            entry_pred = pred_now

            trades.append({
                "type": "ENTRY",
                "signal": entry_signal,
                "price": price,
                "index": i,
                "pred_ret_entry": entry_pred,
            })

        # --------------------------------
        # EXIT logic
        # --------------------------------
        elif position > 0:
            hold += 1
            pnl = (price - entry_price) * position
            exit_reason = None

            # 1) HARD_STOP
            if pnl <= max_loss_cap:
                exit_reason = "HARD_STOP"

            # 2) MODEL_FLIP (prediction sign flips vs entry)
            elif entry_pred is not None and entry_signal is not None:
                if np.sign(pred_now) * np.sign(entry_pred) < 0:
                    exit_reason = "MODEL_FLIP"

            # 3) DECAY (prediction magnitude has collapsed vs entry)
            if exit_reason is None and entry_pred is not None:
                if abs(pred_now) < decay_factor * abs(entry_pred):
                    exit_reason = "DECAY"

            # 4) TIME EXIT
            if exit_reason is None and hold >= max_hold:
                exit_reason = "TIME"

            if exit_reason is not None:
                balance += position * price
                trades.append({
                    "type": "EXIT",
                    "price": price,
                    "index": i,
                    "pnl": pnl,
                    "hold": hold,
                    "reason": exit_reason,
                })
                position = 0.0
                entry_signal = None
                entry_pred = None

    # -----------------------------
    # 6. Final liquidation if needed
    # -----------------------------
    if position > 0:
        balance += position * df["Close"].iloc[-1]
        position = 0.0

    final_balance = balance
    profit_pct = 100.0 * (final_balance - initial_balance) / initial_balance

    trades_df = pd.DataFrame(trades)
    trades_df["pnl"] = trades_df.get("pnl", np.nan)

    if len(equity_curve) > 0:
        equity = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
        max_dd = (equity / equity.cummax() - 1).min()
    else:
        equity = pd.Series(dtype=float)
        max_dd = 0.0

    # -----------------------------
    # 7. Diagnostics
    # -----------------------------
    diagnostics = {}

    diagnostics["pred_ret_raw_stats"] = df["pred_ret_raw"].describe()
    diagnostics["pred_ret_centered_stats"] = df["pred_ret"].describe()
    diagnostics["signal_counts"] = df["signal"].value_counts(dropna=False)
    diagnostics["threshold_value"] = thr
    diagnostics["center_median"] = df["center"].median()
    diagnostics["entry_rate_pct"] = 100.0 * df["entry"].sum() / len(df)
    diagnostics["trade_count"] = len(trades_df)
    diagnostics["win_rate"] = (trades_df["pnl"] > 0).mean() if len(trades_df) else np.nan
    diagnostics["avg_pnl"] = trades_df["pnl"].mean() if len(trades_df) else np.nan
    diagnostics["max_loss_cap"] = max_loss_cap
    diagnostics["decay_factor"] = decay_factor
    diagnostics["side_mode"] = side_mode

    if "hold" in trades_df:
        diagnostics["hold_distribution"] = trades_df["hold"].describe()

    if "reason" in trades_df:
        diagnostics["exit_reason_counts"] = trades_df["reason"].value_counts(dropna=False)

    if "pnl" in trades_df and len(trades_df) > 0:
        diagnostics["top_5_winners"] = trades_df.nlargest(5, "pnl")
        diagnostics["top_5_losers"] = trades_df.nsmallest(5, "pnl")

        joined = trades_df.join(df["pred_ret"], on="index", rsuffix="_pred")
        diagnostics["pred_ret_pnl_corr"] = joined[["pred_ret", "pnl"]].corr().iloc[0, 1]
    else:
        diagnostics["top_5_winners"] = pd.DataFrame()
        diagnostics["top_5_losers"] = pd.DataFrame()
        diagnostics["pred_ret_pnl_corr"] = np.nan

    return {
        "final_balance": final_balance,
        "profit_pct": profit_pct,
        "trades": trades_df,
        "equity_curve": equity,
        "max_drawdown": max_dd,
        "diagnostics": diagnostics,
    }
