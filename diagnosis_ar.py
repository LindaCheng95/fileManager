import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

y = pd.to_numeric(df["filtered_series"], errors="coerce").dropna().reset_index(drop=True)

max_p = 10
hold_back = max_p

rows = []

for p in range(1, max_p + 1):
    model = AutoReg(y, lags=p, old_names=False, trend="c", hold_back=hold_back).fit()
    row = {
        "p": p,
        "aic": model.aic,
        "bic": model.bic,
        "sigma2": np.mean(model.resid ** 2),
    }
    for i, name in enumerate(model.model.exog_names):
        row[f"coef_{name}"] = model.params[i]
        row[f"t_{name}"] = model.tvalues[i]
    rows.append(row)

summary_df = pd.DataFrame(rows)
print(summary_df)