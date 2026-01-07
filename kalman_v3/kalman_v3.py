"""
kalman_v3.py
-----------------
Adaptive Kalman Filter with CONFIDENCE-GATED forecasting.

Design:
- Kalman estimates FAIR PRICE + VELOCITY
- Forecasting is CONDITIONAL
- Persistence is the safe fallback
- No unconditional extrapolation

This version combines:
PATH 1: Fair-price estimation
PATH 2: Confidence-gated forecasting
"""

import time
import csv
import argparse
from datetime import datetime, timezone

import numpy as np
import ccxt
import os





def save_state(state_file, x, P, Q_scale, R_scale, last_ts):
    np.savez(
        state_file,
        x=x,
        P=P,
        Q_scale=Q_scale,
        R_scale=R_scale,
        last_ts=last_ts.timestamp() if last_ts else None
    )


def load_state(state_file):
    if not os.path.exists(state_file):
        return None

    data = np.load(state_file, allow_pickle=True)
    x = data["x"]
    P = data["P"]
    Q_scale = float(data["Q_scale"])
    R_scale = float(data["R_scale"])
    ts_val = data["last_ts"].item()

    last_ts = (
        datetime.fromtimestamp(ts_val, tz=timezone.utc)
        if ts_val is not None
        else None
    )

    return x, P, Q_scale, R_scale, last_ts



# ===============================
# Arguments
# ===============================
def parse_args():
    p = argparse.ArgumentParser(description="Kalman v3 – Confidence-Gated Estimator")
    p.add_argument("--symbol", type=str, default="BTC/USDT")
    p.add_argument("--timeframe", type=str, default="1m")
    p.add_argument("--exchange", type=str, default="binance")
    p.add_argument("--horizon", type=int, default=5, help="forecast horizon in minutes")
    p.add_argument("--out", type=str, default="results_v3.csv")
    return p.parse_args()


# ===============================
# Exchange
# ===============================
def init_exchange(name):
    ex = getattr(ccxt, name)()
    ex.load_markets()
    return ex


# ===============================
# Kalman initialization
# ===============================
def init_kalman():
    # State: [price, velocity]
    x = np.array([[0.0], [0.0]])
    P = np.eye(2) * 10.0

    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    Q_base = np.array([[0.01, 0.0],
                       [0.0, 0.001]])
    R_base = np.array([[0.1]])

    return x, P, F, H, Q_base, R_base


# ===============================
# Confidence function
# ===============================
def compute_confidence(innovation, velocity, R_scale):
    inv_score = np.exp(-innovation / 50.0)      # smooth decay
    vel_score = np.tanh(abs(velocity) / 50.0)   # saturates naturally
    noise_score = 1.0 / (1.0 + R_scale)

    return 0.5 * inv_score + 0.3 * vel_score + 0.2 * noise_score



# ===============================
# CSV logger
# ===============================
def init_csv(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "symbol",
            "raw_price",
            "fair_price",
            "fair_velocity",
            "confidence",
            "forecast",
            "forecast_mode",
            "Q_scale",
            "R_scale"
        ])


def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ===============================
# Main loop
# ===============================
def run():
    args = parse_args()
    ex = init_exchange(args.exchange)

    symbol = args.symbol
    tf = args.timeframe
    horizon = args.horizon
    out_csv = args.out
    CONF_THRESHOLD = 0.55
    save_counter = 0
    state_file = f"kalman_state_v3_{symbol.replace('/', '_')}_{tf}.npz"



    init_csv(out_csv)

    x, P, F, H, Q_base, R_base = init_kalman()
    Q_scale = 1.0
    R_scale = 1.0
    last_ts = None

    loaded = load_state(state_file)
    if loaded is not None:
        x, P, Q_scale, R_scale, last_ts = loaded
        print("[INFO] Kalman state restored from disk")
    else:
        print("[INFO] Starting with fresh Kalman state")


    print("[INFO] Kalman v3 started (confidence-gated forecasting)")
    print("[INFO] Press Ctrl+C to stop")

    while True:
        try:
            ohlcv = ex.fetch_ohlcv(symbol, tf, limit=2)
            ts, _, _, _, close, _ = ohlcv[-1]
            close = float(close)

            close_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
            if last_ts == close_dt:
                time.sleep(1)
                continue
            last_ts = close_dt

            z = np.array([[close]])

            # ---------- PREDICT ----------
            Q = Q_base * Q_scale
            x = F @ x
            P = F @ P @ F.T + Q

            # ---------- UPDATE ----------
            y = z - (H @ x)
            S = H @ P @ H.T + R_base * R_scale
            K = P @ H.T @ np.linalg.inv(S)

            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P

            innovation_mag = abs(float(y[0, 0]))

            # ---------- ADAPT NOISE ----------
            R_scale = 0.95 * R_scale + 0.05 * min(10.0, innovation_mag)
            Q_scale = 0.98 * Q_scale + 0.02 * min(5.0, innovation_mag)

            # ---------- FAIR PRICE ----------
            fair_price = float(x[0, 0])
            fair_velocity = float(np.clip(x[1, 0], -2000, 2000))
            x[1, 0] = fair_velocity

            # ---------- CONFIDENCE ----------
            confidence = compute_confidence(
                innovation_mag,
                fair_velocity,
                R_scale
            )

            # ---------- GATED FORECAST ----------
            if confidence > CONF_THRESHOLD:
                forecast = fair_price + horizon * fair_velocity
                mode = "KALMAN"
            else:
                forecast = close
                mode = "PERSIST"

            # ---------- LOG ----------
            append_csv(out_csv, [
                close_dt.isoformat(),
                symbol,
                close,
                fair_price,
                fair_velocity,
                confidence,
                forecast,
                mode,
                Q_scale,
                R_scale
            ])

            # ---------- CONSOLE ----------
            if mode == "KALMAN":
                print(
                    f"[{close_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"raw={close:,.2f} | fair={fair_price:,.2f} | "
                    f"vel={fair_velocity:,.2f} | "
                    f"conf={confidence:.3f} | "
                    f"FORECAST(+{horizon})={forecast:,.2f}"
                )
            else:
                print(
                    f"[{close_dt.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"raw={close:,.2f} | fair={fair_price:,.2f} | "
                    f"vel={fair_velocity:,.2f} | "
                    f"conf={confidence:.3f} | mode=PERSIST"
                )
                
            save_counter += 1
            if save_counter % 5 == 0:  # every 5 minutes
                save_state(state_file, x, P, Q_scale, R_scale, last_ts)




        except KeyboardInterrupt:
            save_state(state_file, x, P, Q_scale, R_scale, last_ts)
            print("\n[INFO] State saved. Stopped by user.")
            break
        except Exception as e:
            print("[WARN]", str(e))
            time.sleep(5)


if __name__ == "__main__":
    run()
