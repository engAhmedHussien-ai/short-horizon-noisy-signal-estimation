# Adaptive Online Kalman Estimator — v2  
**Fair-Price Estimation Under Non-Stationarity**

## Overview

This repository implements **v2** of an adaptive Kalman filter system, explicitly reframed as a **fair-price estimator** rather than a price forecaster.

The estimator operates **online**, under noisy and non-stationary conditions, and is designed to suppress microstructure noise while preserving responsiveness to genuine price movements.

> This version intentionally avoids direct price forecasting.

---

## Motivation (Why v2 Exists)

In v1, the Kalman filter was used to extrapolate prices over short horizons and benchmarked against persistence (random walk).

Empirical results showed that:

- Persistence consistently outperformed Kalman on MAE/RMSE
- Kalman extrapolation amplified noise at high frequency
- Errors increased rapidly with horizon length

This behavior is **expected** in high-noise regimes and revealed a **model-role mismatch**, not a coding error.

**v2 corrects that mismatch.**

---

## Core Idea

> **Use the Kalman filter to estimate structure and uncertainty,  
> not to force short-horizon price prediction.**

In v2:

- Kalman estimates:
  - latent fair price
  - short-term velocity
  - adaptive uncertainty
- Forecasting is removed entirely
- Persistence remains the default forecast when needed

This aligns with how Kalman filters are used in control systems, sensor fusion, and industrial monitoring.

---

## Model Formulation

### State Vector

The latent state is defined as:

\[
x_t =
\begin{bmatrix}
p_t \\
v_t
\end{bmatrix}
\]

Where:
- \(p_t\) is the estimated fair price  
- \(v_t\) is the estimated short-term velocity  

---

### State Transition Model (Constant Velocity)



\[
x_{t+1} =
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
x_t + w_t
\]

Design intent:
- Locally linear approximation
- Minimal and interpretable dynamics
- Avoids overfitting in high-noise regimes

Velocity is physically bounded to prevent unrealistic behavior.

---

### Measurement Model

\[
z_t = [1 \;\; 0] \, x_t + v_t
\]

Only the price component is observed.  
Measurement noise is adapted online using innovation magnitude.

---

## Adaptive Noise Handling

The estimator adapts noise statistics online:

- Measurement noise \(R\) reflects observation reliability
- Process noise \(Q\) adapts slowly to model mismatch

Constraints:
- No indicators
- No look-ahead
- No retrospective fitting

A warm-up period prevents initialization artifacts.

---

## What v2 Evaluates

### Evaluated

- Noise reduction vs raw price
- Lag vs EMA during sharp price moves
- Stability across volatility regimes
- Estimator behavior (not prediction accuracy)

### Explicitly Not Evaluated

- Directional accuracy
- Trading profitability
- Horizon-based forecasting

Those belong to higher-level, gated systems (planned in later versions).

---

## Baselines for Comparison

v2 compares **estimation quality**, not forecasts:

- Raw price
- EMA(10)
- EMA(20)

The Kalman estimator is expected to:
- Reduce noise more than raw price
- React faster than EMA during regime changes
- Avoid excessive lag

Persistence remains the default forecast outside this estimator.

---


