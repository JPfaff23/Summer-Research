"""
WOF_FD+AAD_From_Existing.py
  • Read existing cleaned scenarios in chunks
  • Compute FD Δ/Vega and (optionally) AAD Δ/Vega for each scenario on GPU
  • Write two Parquet outputs incrementally (each including all original columns + Greeks)
  • Displays a tqdm progress bar with ETA for chunks
"""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
import torch
from tqdm import tqdm

# -----------------------------------------------------------------------------
# CONFIGURATION SWITCH
# -----------------------------------------------------------------------------
DO_AAD = True        # set to False to skip AAD block entirely

# Constants
PARQUET_IN   = "Test_clean_5k.parquet"
PARQUET_FD   = "Test_clean_5k_fd_greeks.parquet"
PARQUET_AAD  = "Test_clean_5k_aad_greeks.parquet"
N_PATHS      = 100_000_000  # use 1M paths per scenario
BASE_SEED    = 42
N_ASSETS     = 3
CHUNK_MAX    = 100_000

# ---- rebuild params (normalized K=1) ----
def rebuild_params(row):
    S0 = np.array([row[f"S0_{i}/K"] for i in range(N_ASSETS)])
    sigma = np.array([row[f"sigma_{i}"] for i in range(N_ASSETS)])
    rho = np.array([
        [1.0 if i == j else row[f"corr_{min(i,j)}_{max(i,j)}"]
         for j in range(N_ASSETS)]
        for i in range(N_ASSETS)
    ])
    return {"S0": S0, "sigma": sigma, "T": row["T"], "r": row["r"], "K": 1.0, "rho": rho}

# ---- FD Greeks via torch on GPU ----
def delta_vega_fd(p, n_paths, rel=1e-4, seed=None):
    device = torch.device('cuda')
    if seed is not None:
        torch.manual_seed(seed)

    S0  = torch.tensor(p['S0'],    dtype=torch.float64, device=device)
    sig = torch.tensor(p['sigma'], dtype=torch.float64, device=device)
    r   = torch.tensor(p['r'],     dtype=torch.float64, device=device)
    T   = torch.tensor(p['T'],     dtype=torch.float64, device=device)
    K   = torch.tensor(p['K'],     dtype=torch.float64, device=device)
    rho = torch.tensor(p['rho'],   dtype=torch.float64, device=device)

    # simulate correlated GBM
    L = torch.linalg.cholesky(rho)
    Z = torch.randn(n_paths, len(S0), dtype=torch.float64, device=device)
    Y = Z @ L.T
    drift = (r - 0.5 * sig**2) * T
    diff  = sig * torch.sqrt(T) * Y
    disc  = torch.exp(-r * T)

    delta = torch.zeros_like(S0)
    vega  = torch.zeros_like(sig)

    for i in range(len(S0)):
        # Delta (central)
        bump_s = rel * S0[i]
        S_up = S0.clone(); S_up[i] += bump_s
        S_dn = S0.clone(); S_dn[i] -= bump_s
        pay_up = torch.clamp((torch.exp(torch.log(S_up)+drift+diff)).min(dim=1).values - K, 0.0)
        pay_dn = torch.clamp((torch.exp(torch.log(S_dn)+drift+diff)).min(dim=1).values - K, 0.0)
        delta[i] = (disc*pay_up.mean() - disc*pay_dn.mean())/(2*bump_s)

        # Vega (central)
        bump_v = rel * sig[i]
        sig_up = sig.clone(); sig_up[i] += bump_v
        sig_dn = sig.clone(); sig_dn[i] -= bump_v

        drift_up = (r - 0.5*sig_up**2) * T
        diff_up  = sig_up * torch.sqrt(T) * Y
        pay_vup  = torch.clamp((torch.exp(torch.log(S0)+drift_up+diff_up)).min(dim=1).values - K, 0.0)

        drift_dn = (r - 0.5*sig_dn**2) * T
        diff_dn  = sig_dn * torch.sqrt(T) * Y
        pay_vdn  = torch.clamp((torch.exp(torch.log(S0)+drift_dn+diff_dn)).min(dim=1).values - K, 0.0)

        vega[i] = (disc*pay_vup.mean() - disc*pay_vdn.mean())/(2*bump_v)

    return delta.cpu().numpy(), vega.cpu().numpy()

# ---- AAD Greeks via autograd ----
def delta_vega_aad(p, n_paths, seed):
    device = torch.device('cuda')
    torch.manual_seed(seed)
    S0  = torch.tensor(p['S0'],    dtype=torch.float64, requires_grad=True, device=device)
    sig = torch.tensor(p['sigma'], dtype=torch.float64, requires_grad=True, device=device)
    r   = torch.tensor(p['r'],     dtype=torch.float64, device=device)
    T   = torch.tensor(p['T'],     dtype=torch.float64, device=device)
    K   = torch.tensor(p['K'],     dtype=torch.float64, device=device)
    rho = torch.tensor(p['rho'],   dtype=torch.float64, device=device)

    L = torch.linalg.cholesky(rho)
    Z = torch.randn(n_paths, N_ASSETS, dtype=torch.float64, device=device)
    Y = Z @ L.T
    drift = (r - 0.5*sig**2) * T
    diff  = sig * torch.sqrt(T) * Y

    logS   = torch.log(S0) + drift + diff
    ST     = torch.exp(logS)
    payoff = torch.clamp(ST.min(dim=1).values - K, 0.0)
    price  = torch.exp(-r*T) * payoff.mean()

    delta, vega = torch.autograd.grad(price, (S0, sig))
    return delta.cpu().numpy(), vega.cpu().numpy()

# ---- main loop ----
def main():
    pf = pq.ParquetFile(PARQUET_IN)
    total = pf.metadata.num_rows
    writer_fd = writer_aad = None

    for start in tqdm(range(0, total, CHUNK_MAX), desc="Chunks", unit="chunk"):
        end = min(start + CHUNK_MAX, total)
        df_chunk = pd.read_parquet(PARQUET_IN).iloc[start:end].copy()

        fd_res  = []
        aad_res = []

        for idx, row in df_chunk.iterrows():
            p = rebuild_params(row)

            # FD Greeks
            d_fd, v_fd = delta_vega_fd(p, N_PATHS, seed=BASE_SEED + start + idx)
            fd_res.append({f"delta_{i}": d_fd[i] for i in range(N_ASSETS)} |
                          {f"vega_{i}": v_fd[i] for i in range(N_ASSETS)})

            # AAD Greeks (if enabled)
            if DO_AAD:
                d_aad, v_aad = delta_vega_aad(p, N_PATHS, BASE_SEED + start + idx)
                aad_res.append({f"delta_{i}": d_aad[i] for i in range(N_ASSETS)} |
                               {f"vega_{i}": v_aad[i] for i in range(N_ASSETS)})

        # build FD table including original columns + Greeks
        df_fd = pd.concat([
            df_chunk.reset_index(drop=True),
            pd.DataFrame(fd_res)
        ], axis=1)
        tbl_fd = pa.Table.from_pandas(df_fd)
        if writer_fd is None:
            writer_fd = pq.ParquetWriter(PARQUET_FD, tbl_fd.schema, compression='zstd')
        writer_fd.write_table(tbl_fd)

        # build AAD table including original columns + Greeks
        if DO_AAD:
            df_aad = pd.concat([
                df_chunk.reset_index(drop=True),
                pd.DataFrame(aad_res)
            ], axis=1)
            tbl_aad = pa.Table.from_pandas(df_aad)
            if writer_aad is None:
                writer_aad = pq.ParquetWriter(PARQUET_AAD, tbl_aad.schema, compression='zstd')
            writer_aad.write_table(tbl_aad)

    writer_fd.close()
    if writer_aad is not None:
        writer_aad.close()

if __name__ == '__main__':
    main()
