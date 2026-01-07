#!/usr/bin/env python3
"""
“residual-based adaptive Kalman filter” with limitations

"""

import argparse
import ccxt
import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime, timedelta, timezone


import numpy as np
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)


# ---------------- utils ---------------- #
def now_utc():
    return datetime.now(timezone.utc)


def to_dt(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


# ---------------- RSI ---------------- #
def compute_rsi(prices, period=14):
   
    if len(prices) < period + 1:
        return None
    p = np.asarray(prices, dtype=float)
    deltas = np.diff(p)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        rs = np.inf
        rsi = 100.0
    else:
        rs = up / down
        rsi = 100.0 - (100.0 / (1.0 + rs))

    up_ema = up
    down_ema = down
    for delta in deltas[period:]:
        up_val = max(delta, 0.0)
        down_val = -min(delta, 0.0)
        up_ema = (up_ema * (period - 1) + up_val) / period
        down_ema = (down_ema * (period - 1) + down_val) / period
        if down_ema == 0:
            rsi = 100.0
        else:
            rs = up_ema / down_ema
            rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


# ---------------- Kalman Filter ---------------- #
def ensure_col_vec(x):
    arr = np.atleast_2d(x)
    if arr.shape[0] == 1 and arr.shape[1] > 1:
        arr = arr.T
    return arr
    

def ema(series, alpha):
    if not series:
        return None
    v = series[0]
    for x in series[1:]:
        v = alpha * x + (1 - alpha) * v
    return float(v)
"""
last_price = close_price
ema10 = ema(state["price_history"][-10:], 2/(10+1))
ema20 = ema(state["price_history"][-20:], 2/(20+1))
"""






import numpy as np

"""
    alpha_s — window of innovation variance (0.05–0.2 typical). Larger → smoother, slower reaction.

    gamma_R — how fast R follows observed residual variance. Try 0.01–0.1.

    gamma_Q — usually very small (0.001–0.02) because Q should adapt slowly.

    q_min, r_min — avoid exact zeros (numerical problems).

"""

def kalman_predict_update(
    x, P, z, Q_scale, R_scale, S_ema,
    alpha_s=0.05,     # innovation variance smoothing
    gamma_Q=0.005,    # very slow Q adaptation
    gamma_R=0.05,     # moderate R adaptation
    q_min=1e-8, q_max=1e-2,
    r_min=1e-8, r_max=1e-1
):
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    Q = np.eye(2) * Q_scale
    R = np.array([[R_scale]])

    # Predict
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Innovation
    y_pred = float((H @ x_pred).item())
    e = z - y_pred
    e2 = e * e

    S = float((H @ P_pred @ H.T + R).item())
    K = P_pred @ H.T / S

    # Update
    x_new = x_pred + K * e
    I = np.eye(2)
    P_new = (I - K @ H) @ P_pred

    # Innovation variance tracking
    if S_ema is None:
        S_ema = e2
    else:
        S_ema = (1 - alpha_s) * S_ema + alpha_s * e2

    # Adapt R from residual variance ONLY
    R_target = max(r_min, S_ema - float((H @ P_pred @ H.T).item()))
    R_scale = (1 - gamma_R) * R_scale + gamma_R * R_target

    # Adapt Q very slowly
    Q_scale = (1 - gamma_Q) * Q_scale + gamma_Q * abs(e)

    Q_scale = float(np.clip(Q_scale, q_min, q_max))
    R_scale = float(np.clip(R_scale, r_min, r_max))

    return x_new, P_new, Q_scale, R_scale, S_ema, y_pred, e



# ---------------- State persistence ---------------- #
def save_state(path, state):
    try:
        with open(path, "wb") as f:
            pickle.dump(state, f)
    except Exception as e:
        print("[WARN] Failed to save state:", e, flush=True)


def load_state(path, horizons):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            st = pickle.load(f)
        if not isinstance(st, dict):
            return None

        # Ensure required keys exist (backwards compatibility)
        defaults = {
            "x": np.array([[0.0], [0.0]]),
            "P": np.diag([1.0, 0.01]),
            "Q_scale": 1e-3,
            "R_scale": 1e-3,
     
            "last_minute_ts": None,
            "pending": {h: [] for h in horizons},
            "price_history": [],
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "volume_imbalance": 0.0,
            "RSI": None,
            "warmup": 0
        }
        for k, v in defaults.items():
            if k not in st:
                st[k] = v
        # ensure pending contains all horizons
        if "pending" not in st or not isinstance(st["pending"], dict):
            st["pending"] = {h: [] for h in horizons}
        else:
            for h in horizons:
                if h not in st["pending"]:
                    st["pending"][h] = []
        if "price_history" not in st or not isinstance(st["price_history"], list):
            st["price_history"] = []
        return st
    except Exception as e:
        print(f"[WARN] Failed to load state from {path}: {e}", flush=True)
        return None


# ---------------- CSV helpers ---------------- #
def init_csv(csv_path, horizons):
    headers = (
        ["row_type", "timestamp", "symbol", "raw", "smoothed"]
        + [f"pred_{h}" for h in horizons]
        + [f"check_{h}" for h in horizons]
        + ["Q_scale", "R_scale", "RSI", "avg_errs"]
    )
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    return headers


def append_candle_csv(csv_path, ts, symbol, raw_price, smoothed, preds, checks, Q_scale, R_scale, rsi=None):
    """
    Append one candle row (with predictions & checks) to CSV.
    Emojis are kept in terminal, but CSV uses plain OK/FAIL flags.
    """
    fieldnames = ["timestamp", "symbol", "raw_price", "smoothed"] + \
                 [f"pred_{h}" for h in preds.keys()] + \
                 [f"check_{h}" for h in preds.keys()] + \
                 ["Q_scale", "R_scale", "RSI"]

    row = {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "raw_price": raw_price,
        "smoothed": smoothed,
        "Q_scale": Q_scale,
        "R_scale": R_scale,
        "RSI": rsi if rsi is not None else ""
    }

    # Add predictions
    for h, p in preds.items():
        row[f"pred_{h}"] = p


    # Write row with UTF-8 encoding
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)



def append_summary_csv(csv_path, timestamp, Q_scale, R_scale, RSI, avg_errs):
    row = ["summary", timestamp.isoformat(), "SUMMARY", "", ""]
    # fill preds and checks with empties for summary
    # determine horizon count from avg_errs keys order (we just put JSON)
    # put JSON avg_errs into last column
    row += [""] * 0  # preds empty
    row += [""] * 0  # checks empty
    row += [Q_scale, R_scale, RSI, json.dumps(avg_errs)]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def init_csv(csv_path, horizons):
    # Create header if missing (mirror init in main)
    headers = (
        ["row_type", "timestamp", "symbol", "raw", "smoothed"]
        + [f"pred_{h}" for h in horizons]
        + [f"check_{h}" for h in horizons]
        + ["Q_scale", "R_scale", "RSI", "avg_errs"]
    )
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def append_user_interrupted(csv_path):
    row = ["interrupted", datetime.now(timezone.utc).isoformat(), "USER_INTERRUPTED"] + [""] * 20
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


# ---------------- Trades & RSI updater (5s heartbeat) ---------------- #
def fetch_trades_and_update(exchange, symbol, state, rsi_period, ohlcv_rsi_tf="1m", rsi_ohlcv_limit=100):
    """
    Fetch recent trades for last 5 seconds -> update buy/sell volumes and volume imbalance.
    Fetch recent closes to compute RSI -> update state's RSI and R_scale via RS formula.
    Returns (imbalance_delta, rsi, RS).
    """
    now_ms = int(time.time() * 1000)
    since = now_ms - 5000  # last 5 seconds
    imbalance_delta = 0.0
    try:
        trades = exchange.fetch_trades(symbol, since=since)
        # trades items usually have keys: 'side' ('buy'/'sell') and 'amount'
        buy_v = sum(float(t.get("amount", 0.0)) for t in trades if t.get("side") == "buy")
        sell_v = sum(float(t.get("amount", 0.0)) for t in trades if t.get("side") == "sell")
        state["buy_volume"] += buy_v
        state["sell_volume"] += sell_v
        imbalance_delta = buy_v - sell_v
        state["volume_imbalance"] += imbalance_delta
    except Exception:
        # fail gracefully: no trades -> no change
        pass

    # update price history with latest close to compute RSI (we'll fetch recent ohlcv)
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=ohlcv_rsi_tf, limit=rsi_ohlcv_limit)
        closes = [float(c[4]) for c in ohlcv if c and len(c) >= 5]
        state["price_history"] = closes[-rsi_ohlcv_limit :]
        rsi = compute_rsi(state["price_history"], period=rsi_period)
        state["RSI"] = rsi
    except Exception:
        rsi = state.get("RSI", None)

    RS = None
    if rsi is not None:
        # RS = (100/(100-rsi)) - 1, avoid divide by zero
        denom = 100.0 - rsi
        if denom <= 0.0:
            RS = float("inf")
        else:
            RS = (100.0 / denom) - 1.0
        # Fully replace R_scale with RSI-driven scale multiplicative factor:
        # apply factor (1 + RS) to base measurement noise; we keep a base R_base in state
        # Guarantee R_scale stays finite and positive
        rb = state.get("R_base", 1e-3)
        new_R_scale = rb * (1.0 + RS)
        # clamp to reasonable bounds
        new_R_scale = float(np.clip(new_R_scale, 1e-8, 1e2))
        state["R_scale"] = new_R_scale
    else:
        RS = None

    return imbalance_delta, rsi, RS

def run(symbol,timeframe,horizons,state_path,csv_path,print_cycle,tol,rsi_period):
    

    exchange = ccxt.okx()
    init_csv(csv_path, horizons)  # ensure file exists with header
    state = load_state(state_path, horizons)
    rsi_period = 14
    if state is None:
        state = {
            "x": np.array([[0.0], [0.0]]),
            "P": np.eye(2),
            "Q_scale": 1e-3,
            "R_base": 1e-3,
            "R_scale": 1e-3,
            "S_ema": 1e-3,
            "err": 0.0,
            "last_minute_ts": None,
            "pending": {h: [] for h in horizons},
            "price_history": [],
            "buy_volume": 0.0,
            "sell_volume": 0.0,
            "last_acc_volume": 0.0,
            "volume_imbalance": 0.0,
            "RSI": None,
            "RS": None,
            "warmup": 0
        }

    candle_counter = 0
    last_2s = 0
    last_rs = None

    print("[INFO] Starting loop. Press Ctrl+C to stop.", flush=True)
    try:
        while True:
            # heartbeat every 2 sec
            now_sec = int(time.time())
            if now_sec - last_2s >= 2:
                _, rsi, RS = fetch_trades_and_update(exchange, symbol, state, rsi_period)
                state["RSI"] = rsi
                state["RS"] = RS
                last_2s = now_sec

            # get candles
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=2)
            except Exception as e:
                print("[WARN] fetch_ohlcv failed:", e, flush=True)
                time.sleep(1)
                continue

            cand = ohlcv[-2] if len(ohlcv) >= 2 else ohlcv[-1]
            close_ts_ms = int(cand[0])
            close_dt = to_dt(close_ts_ms)
            close_price = float(cand[4])
            
            # ---------------- BASELINES ----------------
            last_price = close_price

            ema10 = None
            ema20 = None
            ph = state.get("price_history", [])

            if len(ph) >= 10:
                ema10 = ema(ph[-10:], alpha=2/(10+1))

            if len(ph) >= 20:
                ema20 = ema(ph[-20:], alpha=2/(20+1))

            if state["last_minute_ts"] is None or close_dt > state["last_minute_ts"]:
                # ---- Kalman update (unchanged) ----
                
                
                
                x, P, Q_scale, R_scale, S_ema, y_pred, err = kalman_predict_update(
                state["x"], state["P"], close_price,
                state["Q_scale"], state["R_scale"], state["S_ema"]
                )
                # --- velocity sanity clamp ---
                max_vel = 2000.0  # BTC $/minute (conservative physical bound)
                x[1, 0] = float(np.clip(x[1, 0], -max_vel, max_vel))

                state.update({"x": x, "P": P, "Q_scale": Q_scale, "R_scale": R_scale, "S_ema": S_ema})
                state["warmup"] += 1
                if state["warmup"] < 10:
                    print(f"[WARMUP] minute {state['warmup']}/10")
                    state["last_minute_ts"] = close_dt
                    save_state(state_path, state)
                    continue

                
                smoothed = float(x[0].item())

                # predictions
                preds, checks = {}, {}
                for h in horizons:
                    Fh = np.array([[1.0, float(h)], [0.0, 1.0]])
                    pred_price = float((Fh @ state["x"])[0].item())
                    preds[h] = pred_price
                    target_ts = close_dt + timedelta(minutes=h)
                    state["pending"][h].append({"t": target_ts,"kalman": pred_price,"last": last_price,"ema10": ema10,"ema20": ema20})


                # ---------------- VALIDATIONS ----------------
                matured_any = False

                for h in horizons:
                    for rec in list(state["pending"][h]):
                        if close_dt >= rec["t"]:

                            # Validate all models for this horizon
                            for model in ["kalman", "last", "ema10", "ema20"]:
                                p = rec.get(model)
                                if p is None:
                                    continue

                                err = close_price - p
                                abs_err = abs(err)
                                dir_ok = np.sign(err) == np.sign(close_price - last_price)

                                checks.setdefault(h, []).append({
                                    "model": model,
                                    "pred": p,
                                    "abs_err": abs_err,
                                    "sq_err": err * err,
                                    "dir_ok": dir_ok
                                })

                            # ---- PRINT VALIDATION (truthful, no judgment) ----
                            ts_print = close_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                            print(f"[{ts_print}] Validation +{h}min:")
                            for v in checks[h][-4:]:  # last batch (4 models)
                                d = "✓" if v["dir_ok"] else "✗"
                                print(
                                    f"    {v['model']:>7} | abs_err={v['abs_err']:.2f} | dir={d}"
                                )

                            state["pending"][h].remove(rec)
                            matured_any = True


                # ---- CLI PRINTING ----
                ts_print = close_dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

                # 1) Predictions line
                preds_line = " ".join([f"+{h}={preds[h]:.6f}" for h in horizons])
                print(f"[{ts_print}] {symbol} raw={close_price:.6f} smoothed={smoothed:.6f} | {preds_line}", flush=True)

                # 2) Validation line
                if matured_any:
                    parts = []
                    for h in horizons:
                        if h in checks:
                            for rec in list(state["pending"][h]):
                                if close_dt >= rec["t"]:
                                    for model in ["kalman", "last", "ema10", "ema20"]:
                                        p = rec.get(model)
                                        if p is None:
                                            continue

                                        err = close_price - p
                                        abs_err = abs(err)
                                        dir_ok = np.sign(err) == np.sign(close_price - last_price)

                                        metrics[h].append({
                                            "model": model,
                                            "abs_err": abs_err,
                                            "sq_err": err**2,
                                            "dir_ok": dir_ok
                                        })
                                    state["pending"][h].remove(rec)

                    
                    print(f"[{ts_print}] Validation: " + " ".join(parts), flush=True)
                else:
                    print(f"[{ts_print}] Validation: (no matured checks)", flush=True)

                # 3) Volume + RS line
                buy, sell = state["buy_volume"], state["sell_volume"]
                total_vol = buy + sell
                
                last_acc = state.get("last_acc_volume", None)
                if last_acc is None:
                    delta_vol = 0.0
                else:
                    delta_vol = total_vol - last_acc
                # store for next minute
                state["last_acc_volume"] = total_vol

                # --- Volume coloring ---
                if total_vol > 0:
                    vol_str = f"{Fore.GREEN}{total_vol:>10.2f}{Style.RESET_ALL}"
                elif total_vol < 0:
                    vol_str = f"{Fore.RED}{total_vol:>10.2f}{Style.RESET_ALL}"
                else:
                    vol_str = f"{total_vol:>10.2f}"

                if delta_vol > 0:
                    delta_vol_str = f"{Fore.GREEN}{delta_vol:>10.2f}{Style.RESET_ALL}"
                elif delta_vol < 0:
                    delta_vol_str = f"{Fore.RED}{delta_vol:>10.2f}{Style.RESET_ALL}"
                else:
                    delta_vol_str = f"{delta_vol:>10.2f}"

                # --- RS and delta(RS) ---
                rs_val = state.get("RS", None)
                last_rs = state.get("last_rs", None)

                if rs_val is None:
                    rs_str = f"{'RS=N/A':>12}"
                    delta_rs_str = f"{'ΔRS=N/A':>12}"
                else:
                    rs_color = Fore.GREEN if rs_val >= 1.016 else Fore.RED
                    rs_str = f"{rs_color}{rs_val:>12.4f}{Style.RESET_ALL}"

                    if last_rs is None:
                        delta_rs_str = f"{'ΔRS=N/A':>12}"
                    else:
                        delta_rs = rs_val - last_rs
                        if delta_rs >= 0.1:
                            delta_rs_str = f"{Fore.GREEN}{delta_rs:>+12.4f}{Style.RESET_ALL}"
                        elif delta_rs <= -0.1:
                            delta_rs_str = f"{Fore.RED}{delta_rs:>+12.4f}{Style.RESET_ALL}"
                        else:
                            delta_rs_str = f"{delta_rs:>+12.4f}"

                    # update stored last_rs
                    state["last_rs"] = rs_val

                # --- Print formatted row ---
                print(f"[{ts_print}] Vol={vol_str} ΔVol={delta_vol_str} | RS={rs_str} ΔRS= {delta_rs_str}", flush=True)
                

                # 4) Q/R line
                print(f"[{ts_print}] Q={state['Q_scale']:.3e}, R={state['R_scale']:.3e}", flush=True)
                
                print(f"====================================================================================================================", flush=True) 

                # ---- CSV + state save (unchanged) ----
                flat_checks = {h: "" for h in horizons}
                
                append_candle_csv(csv_path, close_dt, symbol, close_price, smoothed, preds, flat_checks,
                                  state["Q_scale"], state["R_scale"], state.get("RSI", None))

                #candle_counter += 1
                #if candle_counter % print_cycle == 0:
                #    avg_errs = {h: (np.mean([c["err"] for c in checks.get(h, [])]) if checks.get(h) else None)
                #                for h in horizons}
                #    append_summary_csv(csv_path, close_dt, state["Q_scale"], state["R_scale"], state.get("RSI", None), avg_errs)

                state["buy_volume"] = state["sell_volume"] = state["volume_imbalance"] = 0.0
                state["last_minute_ts"] = close_dt
                save_state(state_path, state)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted — saving state.", flush=True)
        append_user_interrupted(csv_path)
        save_state(state_path, state)
        sys.exit(0)




# ---------------- CLI & run ---------------- #



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Kalman with RSI-driven R and 5s buy/sell updater")
    p.add_argument("--symbol", type=str, default="ETH/USDT")
    p.add_argument("--timeframe", type=str, default="1m")
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--state", type=str, default="kalman_state_v3.pkl")
    p.add_argument("--csv", type=str, default="results.csv")
    p.add_argument("--print-cycle", type=int, default=10)
    p.add_argument("--tol", type=float, default=0.01, help="relative error tolerance (e.g. 0.01 = 1%)")
    p.add_argument("--rsi-period", type=int, default=14, help="RSI period used for R scaling (fetched every 5s)")
    args = p.parse_args()

    run(
        args.symbol,
        args.timeframe,
        args.horizons,
        args.state,
        args.csv,
        args.print_cycle,
        args.tol,
        args.rsi_period,
    )
