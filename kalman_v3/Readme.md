## Forecasting is performed conditionally using estimator confidence; in low-confidence regimes the system explicitly falls back to persistence.


### run it like that "python kalman_v3.py --symbol BTC/USDT --timeframe 1m --horizon 5"

## State Persistence and Warm-Start Behavior (v3)

The estimator in **v3** is designed to operate as a **continuous online system**, not as a stateless script.

To avoid repeated warm-up transients and unstable behavior on restart, the Kalman filter **persists its internal state to disk** and restores it automatically when the process is restarted.

### Persisted State

The following internal quantities are saved periodically and on graceful shutdown:

- **State vector (`x`)** — latent fair price and velocity  
- **Covariance matrix (`P`)** — estimator uncertainty  
- **Process noise scale (`Q_scale`)** — model mismatch adaptation  
- **Measurement noise scale (`R_scale`)** — observation reliability adaptation  
- **Last processed timestamp** — prevents duplicate updates  

This represents the minimal sufficient state required to resume estimation consistently.

---

### State File Handling

State is stored using a NumPy `.npz` file with a name derived from the data stream:


## Results (v3 – Confidence-Gated Estimation)

Version 3 evaluates the behavior of a **confidence-gated Kalman estimator** operating online under highly noisy, non-stationary conditions. The system was run continuously for short live intervals (~30 minutes), reflecting realistic uptime constraints rather than idealized backtests.

### Estimator Behavior

Across all runs, the Kalman filter consistently produced a stable **fair-price estimate** that was smoother than raw observations while remaining responsive to genuine price movements. Adaptive noise scaling prevented estimator divergence during volatility spikes and regime changes.

No warm-up instability was observed due to state persistence across restarts.

---

### Confidence-Gated Forecasting

Forecasting was **explicitly conditional**. The system defaulted to persistence and allowed Kalman extrapolation only when internal confidence exceeded a predefined threshold.

Observed behavior:
- Kalman extrapolation occurred **rarely**, typically during short-lived coherent intervals
- Most time steps correctly reverted to persistence under high noise
- Kalman deactivated immediately when confidence deteriorated

This behavior confirms that the gating mechanism is functioning as intended: **preventing extrapolation in chaotic regimes rather than maximizing forecast count**.

---

### Example Event

During one live run, the system detected a brief coherent interval and produced a single short-horizon forecast:

conf = 0.565 → Kalman extrapolation enabled
FORECAST(+5) ≈ fair_price + 5 × velocity
