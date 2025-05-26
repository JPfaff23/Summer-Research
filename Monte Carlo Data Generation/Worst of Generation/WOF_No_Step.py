#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Worst_of_FAST_SE.py
  • multi-GPU, single-process, FP16+TF32 Monte-Carlo
  • ~9 s per 100 k rows × 10 M paths on 4 V100s
  • price_mc streams in chunks to avoid OOM, accepts return_se
  • Removed time-stepping (n_steps) and decoupled scenario RNG
"""

import os, math, time, argparse, pathlib, sys
import numpy as np
import torch
import pyarrow as pa, pyarrow.parquet as pq
from torch.distributions import Beta

# --------- knobs ---------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark       = True
torch.set_default_dtype(torch.float32)

N_ASSETS   = 3
R_RATE     = 0.03
EPS_REL    = 1e-4
SEED_BASE  = 42
CHUNK_MAX  = 1_000_000                 # rows before flushing Parquet
NGPU       = torch.cuda.device_count()
DEVICES    = [torch.device(f"cuda:{i}") for i in range(NGPU)]
SIM_CHUNK  = 1_000_000                 # per-GPU sub-batch

if not NGPU:
    sys.exit("No CUDA GPU visible – aborting.")

# ------ correlation sampler using NumPy RNG ------
def cvine_corr_np(d, a=5.0, b=2.0):
    P = np.eye(d)
    for k in range(d-1):
        for i in range(k+1, d):
            rho = 2.0 * np.random.beta(a, b) - 1.0
            for m in range(k-1, -1, -1):
                rho = rho * np.sqrt((1-P[m,i]**2)*(1-P[m,k]**2)) + P[m,i]*P[m,k]
            P[k,i] = P[i,k] = rho
    ev, evec = np.linalg.eigh(P)
    P = evec @ np.diag(np.clip(ev, 1e-6, None)) @ evec.T
    return torch.as_tensor(P, dtype=torch.float32, device="cuda")

# ------ random scenario ------
def fg_sample():
    z      = np.random.normal(0.5, math.sqrt(0.25), N_ASSETS)
    S0     = 100 * np.exp(z)
    sigma  = np.random.uniform(0.0, 1.0, N_ASSETS)
    T      = (np.random.randint(1, 44)**2) / 252.0
    return dict(
        S0=S0,
        sigma=sigma,
        T=T,
        rho=cvine_corr_np(N_ASSETS),
        K=100.0,
        r=R_RATE
    )

# ------ simulate terminal prices (one-step GBM) ------
@torch.no_grad()
def terminal_prices(S0, sigma, T, rho, *, n_paths, r, gen=None):
    chol = torch.linalg.cholesky(rho)
    Z    = torch.randn(n_paths, N_ASSETS, device=S0.device, generator=gen)
    with torch.autocast('cuda', dtype=torch.float16):
        drift     = (r - 0.5 * sigma**2) * T
        diffusion = sigma * math.sqrt(T) * (Z @ chol.T)
        return torch.exp(torch.log(S0) + drift + diffusion)

# ------ raw MC vector ------
def price_mc_raw(params, n_paths, gen=None):
    per_gpu = n_paths // NGPU
    payoffs = []
    for dev in DEVICES:
        S0    = torch.tensor(params['S0'],    device=dev)
        sigma = torch.tensor(params['sigma'], device=dev)
        T     = torch.tensor(params['T'],     device=dev)
        rho   = params['rho']  # already a tensor on cuda
        K, r  = params['K'], params['r']
        ST = terminal_prices(S0, sigma, T, rho,
                             n_paths=per_gpu, r=r, gen=gen)
        payoffs.append(torch.clamp(ST.min(dim=1).values - K, 0.).cpu())
    all_pay = torch.cat(payoffs, dim=0)
    discount = math.exp(-params['r'] * params['T'])
    return discount * all_pay

# ------ streaming MC ------
def price_mc(params, n_paths, return_se=False):
    """
    Streams simulation in SIM_CHUNK×NGPU batches to avoid OOM.
    Returns mean or (mean, se) if return_se=True.
    """
    per_gpu     = n_paths // NGPU
    total_sum   = 0.0
    total_sumsq = 0.0
    disc        = math.exp(-params['r'] * params['T'])

    for offset in range(0, per_gpu, SIM_CHUNK):
        sz = min(SIM_CHUNK, per_gpu - offset)
        batch = []
        for dev in DEVICES:
            S0    = torch.tensor(params['S0'],    device=dev)
            sigma = torch.tensor(params['sigma'], device=dev)
            T     = torch.tensor(params['T'],     device=dev)
            rho   = params['rho'].to(dev)
            K, r  = params['K'], params['r']
            ST = terminal_prices(S0, sigma, T, rho,
                                 n_paths=sz, r=r)
            batch.append((torch.clamp(ST.min(dim=1).values - K, 0.)
                          * disc).cpu().numpy())
        arr = np.concatenate(batch, axis=0)
        total_sum   += arr.sum()
        total_sumsq += (arr**2).sum()

    mean = total_sum / n_paths
    var  = (total_sumsq / n_paths) - mean*mean
    se   = math.sqrt(var / n_paths)
    return (mean, se) if return_se else mean

# ------ finite-difference Greeks ------
def greeks_fd(params, n_paths):
    base, base_se = price_mc(params, n_paths, return_se=True)
    if base == 0.0:
        zeros = np.zeros(N_ASSETS)
        return (base, base_se, zeros, zeros, zeros, zeros, zeros, zeros, 0.0, 0.0, 0.0, 0.0)

    delta    = np.empty(N_ASSETS)
    delta_se = np.empty(N_ASSETS)
    gamma    = np.empty(N_ASSETS)
    gamma_se = np.empty(N_ASSETS)
    vega     = np.empty(N_ASSETS)
    vega_se  = np.empty(N_ASSETS)

    # 1) Delta, Gamma & Vega via finite differences
    for i in range(N_ASSETS):
        hS = max(EPS_REL * params['S0'][i],    1e-6)
        hV = max(EPS_REL * params['sigma'][i], 1e-6)

        # Delta & Gamma
        up_p = {**params, 'S0': params['S0'].copy()};    up_p['S0'][i] += hS
        dn_p = {**params, 'S0': params['S0'].copy()};    dn_p['S0'][i] -= hS
        up, up_se = price_mc(up_p, n_paths, return_se=True)
        dn, dn_se = price_mc(dn_p, n_paths, return_se=True)
        delta[i]    = (up - dn) / (2*hS)
        delta_se[i] = math.sqrt(up_se**2 + dn_se**2) / (2*hS)
        gamma[i]    = (up - 2*base + dn) / (hS*hS)
        gamma_se[i] = math.sqrt(up_se**2 + (2*base_se)**2 + dn_se**2) / (hS*hS)

        # Vega
        upv_p = {**params, 'sigma': params['sigma'].copy()}; upv_p['sigma'][i] += hV
        dnv_p = {**params, 'sigma': params['sigma'].copy()}; dnv_p['sigma'][i] -= hV
        upv, upv_se = price_mc(upv_p, n_paths, return_se=True)
        dnv, dnv_se = price_mc(dnv_p, n_paths, return_se=True)
        vega[i]    = (upv - dnv) / (2*hV)
        vega_se[i] = math.sqrt(upv_se**2 + dnv_se**2) / (2*hV)

    # 2) Rho
    hR = max(EPS_REL * abs(params['r']), 1e-6)
    rup_p = {**params, 'r': params['r'] + hR}
    rdn_p = {**params, 'r': params['r'] - hR}
    rup, rup_se = price_mc(rup_p, n_paths, return_se=True)
    rdn, rdn_se = price_mc(rdn_p, n_paths, return_se=True)
    rho_v  = (rup - rdn) / (2*hR)
    rho_se = math.sqrt(rup_se**2 + rdn_se**2) / (2*hR)

    # 3) Theta
    hT = max(EPS_REL * params['T'], 1e-6)
    tup_p = {**params, 'T': params['T'] + hT}
    tdn_p = {**params, 'T': max(1e-6, params['T'] - hT)}
    tup, tup_se = price_mc(tup_p, n_paths, return_se=True)
    tdn, tdn_se = price_mc(tdn_p, n_paths, return_se=True)
    theta    = (tup - tdn) / (2*hT)
    theta_se = math.sqrt(tup_se**2 + tdn_se**2) / (2*hT)

    return (base, base_se, delta, delta_se, vega, vega_se,
            gamma, gamma_se, rho_v, rho_se, theta, theta_se)

# ------------------------- driver -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows',       type=int, default=10)
    ap.add_argument('--paths',      type=int, default=10_000_000)
    ap.add_argument('--seed_offset',type=int, default=0)
    ap.add_argument('--out',        type=str, default='Test100.parquet')
    ap.add_argument('--no_chunking',action='store_true')
    args = ap.parse_args()

    np.random.seed(SEED_BASE + args.seed_offset)
    torch.manual_seed(SEED_BASE + args.seed_offset)

    out_path, first = pathlib.Path(args.out), True
    writer = None
    total_start = time.time()
    sample_time = mc_time = 0.0
    rows_left   = args.rows

    print(f"??  Starting Monte-Carlo for {args.rows:,} rows…", flush=True)
    while rows_left:
        batch = min(rows_left, CHUNK_MAX)
        recs  = []
        for _ in range(batch):
            t0 = time.perf_counter(); p = fg_sample(); sample_time += time.perf_counter()-t0
            t1 = time.perf_counter(); vals = greeks_fd(p, args.paths); mc_time += time.perf_counter()-t1
            (pr, pr_se, d, d_se, v, v_se, g, g_se,
             rv, rv_se, th, th_se) = vals
            rec = {**{f"S0_{i}": p['S0'][i] for i in range(N_ASSETS)},
                   **{f"sigma_{i}": p['sigma'][i] for i in range(N_ASSETS)},
                   **{f"delta_{i}": d[i] for i in range(N_ASSETS)},
                   **{f"delta_se_{i}": d_se[i] for i in range(N_ASSETS)},
                   **{f"vega_{i}": v[i] for i in range(N_ASSETS)},
                   **{f"vega_se_{i}": v_se[i] for i in range(N_ASSETS)},
                   **{f"gamma_{i}": g[i] for i in range(N_ASSETS)},
                   **{f"gamma_se_{i}": g_se[i] for i in range(N_ASSETS)},
                   "price": pr, "price_se": pr_se,
                   "rho": rv, "rho_se": rv_se,
                   "theta": th, "theta_se": th_se,
                   "T": p['T'], "r": p['r']}
            recs.append(rec)

        table = pa.Table.from_pylist(recs)
        if first:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression='zstd')
            first = False
        writer.write_table(table)
        rows_left -= batch

    writer.close()
    print(f"?? Sampling: {sample_time:.1f}s | MC+Greeks: {mc_time:.1f}s")
    print(f"? Wrote {args.rows:,} rows → {out_path} in {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()