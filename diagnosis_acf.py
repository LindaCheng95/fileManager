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