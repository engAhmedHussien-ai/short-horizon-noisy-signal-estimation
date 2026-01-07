# Prediction of Noisy Data in Short Time Intervals

This project studies how different modeling choices behave when predicting or estimating signals in **highly noisy, non-stationary environments**, 
using BTC price data purely as a **proxy for a chaotic sensor signal**.

---

## v1 — Direct Short-Horizon Prediction

Version 1 applied a Kalman filter to directly extrapolate short-horizon values and benchmarked it against simple baselines such as persistence and EMA.  
Empirical results showed that persistence consistently outperformed Kalman on error metrics, while Kalman extrapolation amplified noise as horizons increased.  
This behavior revealed a fundamental limitation: **linear state-space models are poorly suited for direct prediction in high-frequency chaotic data**.  
The outcome was expected and highlighted a model-role mismatch rather than an implementation error.

---

## v2 — Fair-Value Estimation Under Noise (Current)

Version 2 reframes the Kalman filter as a **fair-value estimator**, not a predictor.  
Instead of extrapolating noisy measurements, Kalman is used to estimate the latent signal and its uncertainty in real time.  
Persistence is treated as the default short-horizon behavior, while Kalman contributes structure and noise suppression.  
This design aligns with established practice in control systems and sensor fusion, where estimation precedes any decision or prediction layer.

---

## Key Difference Between v1 and v2

| Aspect | v1 | v2 |
|------|----|----|
| Kalman role | Direct predictor | Latent signal estimator |
| Use of extrapolation | Always | Removed |
| Baseline handling | Competes with persistence | Defers to persistence |
| Behavior under noise | Error amplification | Noise suppression |
| Intended use | Forecasting | Estimation / preprocessing |
