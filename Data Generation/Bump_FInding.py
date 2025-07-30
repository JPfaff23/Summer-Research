#!/usr/bin/env python3
# scan_global.py – scan EPS_REL bump sizes
# Each loop iteration reproduces the cold-start behaviour of WOF_No_Step.py
# (fresh RNG seed, fresh scenario) and then prices one Monte-Carlo run.

import time, math
import numpy as np
import pandas as pd
import WOF_No_Step as wo                         # master engine

from WOF_No_Step import (
    fg_sample,        # draws one random scenario
    greeks_fd,        # finite-difference Greeks engine
    N_ASSETS,         # number of underlyings
    torch             # torch already imported inside WOF_No_Step
)

# ─────────────────────────────────────────────────────────────────────────────
SEED_BASE   = wo.SEED_BASE            # keep the same constant as master file
PATHS       = 100_000_000             # MC paths per run (prod setting)

# Only the bump sizes that appear in your new list / spreadsheet
BUMPS = [
    1.0,
    0.7,
    0.5,
    0.3,
    0.2,
    0.1,
    0.07,
    0.05,
    0.03,
    0.02,
    0.015,
    0.01,
    0.0075,
    0.005,
    0.0035,
    0.0025,
    0.001
]


RAW_CSV      = "scan_global.csv"
SUMMARY_CSV  = "scan_global_summary.csv"
# ─────────────────────────────────────────────────────────────────────────────


def reset_rng(seed: int = SEED_BASE) -> None:
    """Exactly replicate the seeding that happens when WOF_No_Step
    is started as a fresh process."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # seed every GPU identically


def greeks_fd_with_eps(params: dict, n_paths: int, eps: float):
    """Wrapper that monkey-patches EPS_REL before calling greeks_fd."""
    wo.EPS_REL = eps                   # override the module-level global
    return greeks_fd(params, n_paths)


# ───────────── main loop ────────────────────────────────────────────────────
records = []

for h in BUMPS:
    reset_rng()                        # cold-start RNGs for this bump
    params = fg_sample()               # new random scenario (like master)

    t0 = time.time()
    (pr, pr_se, d, d_se, v, v_se,
     g, g_se, rho, rho_se, th, th_se) = greeks_fd_with_eps(params, PATHS, h)

    rec = {'h': h}
    rec.update({f'delta_{i}': d[i]       for i in range(N_ASSETS)})
    rec.update({f'delta_se_{i}': d_se[i] for i in range(N_ASSETS)})
    rec.update({f'gamma_{i}': g[i]       for i in range(N_ASSETS)})
    rec.update({f'gamma_se_{i}': g_se[i] for i in range(N_ASSETS)})
    rec.update({f'vega_{i}': v[i]        for i in range(N_ASSETS)})
    rec.update({f'vega_se_{i}': v_se[i]  for i in range(N_ASSETS)})
    rec.update(dict(rho=rho, rho_se=rho_se,
                    theta=th, theta_se=th_se))
    records.append(rec)
    print(f"done h = {h:g}  ({time.time()-t0:.1f}s)")

# ───────────── dump raw results ─────────────────────────────────────────────
raw_df = pd.DataFrame.from_records(records).sort_values("h", ascending=False)
raw_df.to_csv(RAW_CSV, index=False)
print("wrote", RAW_CSV)

# ───────────── quick RMSE proxy & best-h table ─────────────────────────────
def rmse(col, se_col):
    vals = raw_df[col].values
    bias = np.abs(np.diff(vals, append=vals[-1]))  # |G(h)-G(next)|
    var  = raw_df[se_col].values ** 2
    return np.sqrt(bias**2 + var)

summary_rows = []
for greek, se in [("delta_0", "delta_se_0"),
                  ("gamma_0", "gamma_se_0"),
                  ("vega_0",  "vega_se_0"),
                  ("rho",     "rho_se"),
                  ("theta",   "theta_se")]:
    r = rmse(greek, se)
    best = r.argmin()
    summary_rows.append({
        "greek":  greek,
        "h_best": raw_df["h"].iloc[best],
        "rmse":   r[best]
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
print("\nchosen bumps\n", summary_df)
print("wrote", SUMMARY_CSV)
