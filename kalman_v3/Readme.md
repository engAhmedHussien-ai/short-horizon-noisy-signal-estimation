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


