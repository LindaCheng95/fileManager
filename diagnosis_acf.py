import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# --------------------------------------------------------------------------
# Guard: ensure df exists and has the expected column
# --------------------------------------------------------------------------
if "df" not in dir() or not isinstance(df, pd.DataFrame):
    raise NameError("'df' is not defined. Please load your DataFrame before running this script.")
if "filtered_series" not in df.columns:
    raise KeyError("Column 'filtered_series' not found in df.")

y = pd.to_numeric(df["filtered_series"], errors="coerce").dropna().reset_index(drop=True)

if y.empty:
    raise ValueError("'filtered_series' is empty after coercion and dropna().")

# --------------------------------------------------------------------------
# ACF computation
# --------------------------------------------------------------------------
nlags = 60
acf_vals, confint = acf(y, nlags=nlags, fft=True, alpha=0.05)

acf_df = pd.DataFrame({
    "lag": np.arange(len(acf_vals)),
    "acf": acf_vals,
    "ci_lower": confint[:, 0] - acf_vals,  # deviation from ACF value
    "ci_upper": confint[:, 1] - acf_vals,
})

# --------------------------------------------------------------------------
# Successive ACF ratios: rho_{k+1} / rho_k  (fixed range: 1 .. nlags-1)
# --------------------------------------------------------------------------
ratios = []
for k in range(1, len(acf_vals) - 1):  # produces ratios for lags 1..nlags-1
    if np.isclose(acf_vals[k], 0):
        ratios.append(np.nan)
    else:
        ratios.append(acf_vals[k + 1] / acf_vals[k])

ratio_df = pd.DataFrame({
    "lag": np.arange(1, len(acf_vals) - 1),
    "acf_ratio_next_over_current": ratios,
})

print(acf_df.to_string(index=False))
print(ratio_df.to_string(index=False))

# --------------------------------------------------------------------------
# Plot 1: ACF with 95% confidence bands
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Confidence interval shading (centred on 0, not on acf value)
ci_upper = confint[:, 1] - acf_vals  # distance above zero line
ci_lower = confint[:, 0] - acf_vals  # distance below zero line
ax.fill_between(acf_df["lag"], ci_lower, ci_upper, alpha=0.2, color="blue", label="95% CI")

ax.axhline(0, color="black", linewidth=0.8)
ax.vlines(acf_df["lag"], 0, acf_df["acf"], colors="steelblue", linewidth=1.2)
ax.scatter(acf_df["lag"], acf_df["acf"], s=25, color="steelblue", zorder=3)

ax.set_title("Autocorrelation Function (ACF)")
ax.set_xlabel("Lag")
ax.set_ylabel("ACF")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# Plot 2: Successive ACF ratios
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.axhline(0, color="black", linewidth=0.8)
ax.plot(ratio_df["lag"], ratio_df["acf_ratio_next_over_current"],
        marker="o", markersize=4, linewidth=1.2, color="darkorange")

ax.set_title(r"Successive ACF Ratios: $\rho_{k+1} / \rho_k$")
ax.set_xlabel("Lag $k$")
ax.set_ylabel(r"$\rho_{k+1} / \rho_k$")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

def estimate_lambda_from_acf(acf_values, max_lag=30):
    """
    Estimate EMA decay parameter lambda using log-linear regression:
        log(rho(h)) = h * log(lambda)

    Parameters
    ----------
    acf_values : array-like
        ACF values starting at lag 0
    max_lag : int
        Number of lags to use in estimation (avoid noisy long lags)

    Returns
    -------
    results : dict
        lambda estimate, half-life, regression fit, diagnostics
    """

    acf_values = np.array(acf_values)

    # Use lags 1 to max_lag
    lags = np.arange(1, max_lag + 1)
    rho = acf_values[1:max_lag + 1]

    # Only keep positive values (log requires positive)
    mask = rho > 0
    lags = lags[mask]
    rho = rho[mask]

    log_rho = np.log(rho)

    # Regression: log_rho = beta * lag
    X = lags.reshape(-1, 1)
    y = log_rho

    reg = LinearRegression().fit(X, y)

    beta = reg.coef_[0]
    lambda_est = np.exp(beta)

    # Half-life
    half_life = np.log(0.5) / np.log(lambda_est)

    # R^2
    r2 = reg.score(X, y)

    # Fitted values
    fitted_log = reg.predict(X)
    fitted_rho = np.exp(fitted_log)

    print("=" * 80)
    print("EMA PARAMETER ESTIMATION")
    print("=" * 80)
    print(f"Estimated log(lambda): {beta:.6f}")
    print(f"Estimated lambda:      {lambda_est:.6f}")
    print(f"Half-life (days):      {half_life:.2f}")
    print(f"R^2 (log-linear fit):  {r2:.6f}")

    # Plot fit
    plt.figure(figsize=(10, 5))
    plt.scatter(lags, rho, label="Observed ACF")
    plt.plot(lags, fitted_rho, linestyle='--', label="Fitted exp decay")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title("EMA Fit to ACF")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "lambda": lambda_est,
        "half_life": half_life,
        "beta": beta,
        "r2": r2
    }