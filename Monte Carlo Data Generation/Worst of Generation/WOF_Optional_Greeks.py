#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Worst_of_FAST_SE.py
  • multi-GPU, single-process, FP16+TF32 Monte-Carlo
  • price_mc streams in chunks to avoid OOM
  • Optionally skips finite-difference Greeks with --no-greeks
  • Always stores strike K and flattened correlation entries
"""

import os, math, time, argparse, pathlib, sys
import numpy as np
import torch
import pyarrow as pa, pyarrow.parquet as pq
from torch.distributions import Beta   # (still unused, kept in case)

# --------- global knobs ---------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark       = True
torch.set_default_dtype(torch.float32)

N_ASSETS   = 3
R_RATE     = 0.03
EPS_REL    = 1e-3
SEED_BASE  = 42
CHUNK_MAX  = 1_000_000                 # rows before flushing Parquet
NGPU       = torch.cuda.device_count()
DEVICES    = [torch.device(f"cuda:{i}") for i in range(NGPU)]
SIM_CHUNK  = 1_000_000                 # per-GPU sub-batch size

if not NGPU:
    sys.exit("No CUDA GPU visible – aborting.")

# --------- correlation sampler (NumPy RNG) ---------
def cvine_corr_np(d, a: float = 5.0, b: float = 2.0) -> torch.Tensor:
    P = np.eye(d)
    for k in range(d - 1):
        for i in range(k + 1, d):
            rho = 2.0 * np.random.beta(a, b) - 1.0
            for m in range(k - 1, -1, -1):
                rho = rho * np.sqrt((1 - P[m, i] ** 2) * (1 - P[m, k] ** 2)) + P[m, i] * P[m, k]
            P[k, i] = P[i, k] = rho
    ev, evec = np.linalg.eigh(P)
    P = evec @ np.diag(np.clip(ev, 1e-6, None)) @ evec.T
    return torch.as_tensor(P, dtype=torch.float32, device="cuda")

# --------- random scenario generator ---------
def fg_sample():
    z     = np.random.normal(0.5, math.sqrt(0.25), N_ASSETS)
    S0    = 100 * np.exp(z)
    sigma = np.random.uniform(0.0, 1.0, N_ASSETS)
    T     = (np.random.randint(1, 44) ** 2) / 252.0
    return dict(
        S0=S0,
        sigma=sigma,
        T=T,
        rho=cvine_corr_np(N_ASSETS),
        K=100.0,
        r=R_RATE
    )

# --------- terminal price simulation (one-step GBM) ---------
@torch.no_grad()
def terminal_prices(S0, sigma, T, rho, *, n_paths, r, gen=None):
    chol = torch.linalg.cholesky(rho)
    Z    = torch.randn(n_paths, N_ASSETS, device=S0.device, generator=gen)
    with torch.autocast('cuda', dtype=torch.float16):
        drift     = (r - 0.5 * sigma**2) * T
        diffusion = sigma * math.sqrt(T) * (Z @ chol.T)
        return torch.exp(torch.log(S0) + drift + diffusion)

# --------- MC payoff vector for an entire batch ---------
def price_mc_raw(params, n_paths, gen=None):
    per_gpu = n_paths // NGPU
    payoffs = []
    for dev in DEVICES:
        S0    = torch.tensor(params['S0'],    device=dev)
        sigma = torch.tensor(params['sigma'], device=dev)
        T     = torch.tensor(params['T'],     device=dev)
        rho   = params['rho']                 # already on CUDA
        K, r  = params['K'], params['r']
        ST = terminal_prices(S0, sigma, T, rho, n_paths=per_gpu, r=r, gen=gen)
        payoffs.append(torch.clamp(ST.min(dim=1).values - K, 0.).cpu())
    all_pay  = torch.cat(payoffs, dim=0)
    discount = math.exp(-params['r'] * params['T'])
    return discount * all_pay

# --------- streaming MC (avoids GPU OOM) ---------
def price_mc(params, n_paths, return_se=False):
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
            ST = terminal_prices(S0, sigma, T, rho, n_paths=sz, r=r)
            batch.append((torch.clamp(ST.min(dim=1).values - K, 0.) * disc)
                         .cpu().numpy())
        arr = np.concatenate(batch, axis=0)
        total_sum   += arr.sum()
        total_sumsq += (arr ** 2).sum()

    mean = total_sum / n_paths
    var  = (total_sumsq / n_paths) - mean * mean
    se   = math.sqrt(var / n_paths)
    return (mean, se) if return_se else mean

# --------- finite-difference Greeks ---------
def greeks_fd(params, n_paths):
    base, base_se = price_mc(params, n_paths, return_se=True)
    if base == 0.0:                       # option deep OTM
        zeros = np.zeros(N_ASSETS)
        return (base, base_se, zeros, zeros, zeros, zeros,
                zeros, zeros, 0.0, 0.0, 0.0, 0.0)

    delta    = np.empty(N_ASSETS)
    delta_se = np.empty(N_ASSETS)
    gamma    = np.empty(N_ASSETS)
    gamma_se = np.empty(N_ASSETS)
    vega     = np.empty(N_ASSETS)
    vega_se  = np.empty(N_ASSETS)

    # 1) Delta, Gamma & Vega via finite differences
    for i in range(N_ASSETS):
        # underlying shift
        hS = max(EPS_REL * params['S0'][i], 1e-6)
        # volatility shift
        hV = max(EPS_REL * params['sigma'][i], 1e-6)

        # Delta & Gamma
        up_p = {**params, 'S0': params['S0'].copy()}; up_p['S0'][i] += hS
        dn_p = {**params, 'S0': params['S0'].copy()}; dn_p['S0'][i] -= hS
        up, up_se = price_mc(up_p, n_paths, return_se=True)
        dn, dn_se = price_mc(dn_p, n_paths, return_se=True)
        delta[i]    = (up - dn) / (2 * hS)
        delta_se[i] = math.sqrt(up_se ** 2 + dn_se ** 2) / (2 * hS)
        gamma[i]    = (up - 2 * base + dn) / (hS * hS)
        gamma_se[i] = math.sqrt(up_se ** 2 + (2 * base_se) ** 2 + dn_se ** 2) / (hS * hS)

        # Vega
        upv_p = {**params, 'sigma': params['sigma'].copy()}; upv_p['sigma'][i] += hV
        dnv_p = {**params, 'sigma': params['sigma'].copy()}; dnv_p['sigma'][i] -= hV
        upv, upv_se = price_mc(upv_p, n_paths, return_se=True)
        dnv, dnv_se = price_mc(dnv_p, n_paths, return_se=True)
        vega[i]    = (upv - dnv) / (2 * hV)
        vega_se[i] = math.sqrt(upv_se ** 2 + dnv_se ** 2) / (2 * hV)

    # 2) Rho
    hR = max(EPS_REL * abs(params['r']), 1e-6)
    rup_p = {**params, 'r': params['r'] + hR}
    rdn_p = {**params, 'r': params['r'] - hR}
    rup, rup_se = price_mc(rup_p, n_paths, return_se=True)
    rdn, rdn_se = price_mc(rdn_p, n_paths, return_se=True)
    rho_v  = (rup - rdn) / (2 * hR)
    rho_se = math.sqrt(rup_se ** 2 + rdn_se ** 2) / (2 * hR)

    # 3) Theta
    hT = max(EPS_REL * params['T'], 1e-6)
    tup_p = {**params, 'T': params['T'] + hT}
    tdn_p = {**params, 'T': max(1e-6, params['T'] - hT)}
    tup, tup_se = price_mc(tup_p, n_paths, return_se=True)
    tdn, tdn_se = price_mc(tdn_p, n_paths, return_se=True)
    theta    = (tup - tdn) / (2 * hT)
    theta_se = math.sqrt(tup_se ** 2 + tdn_se ** 2) / (2 * hT)

    return (base, base_se, delta, delta_se, vega, vega_se,
            gamma, gamma_se, rho_v, rho_se, theta, theta_se)

# ----------------- main driver -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows',        type=int, default=5_000)
    ap.add_argument('--paths',       type=int, default=100_000_000)
    ap.add_argument('--seed_offset', type=int, default=0)
    ap.add_argument('--out',         type=str, default='Train_New_5M.parquet')
    ap.add_argument('--no_chunking', action='store_true')
    ap.add_argument('--no-greeks',   dest='compute_greeks', action='store_false',
                    default=False, help='skip finite-difference Greeks')
    args = ap.parse_args()

    np.random.seed(SEED_BASE + args.seed_offset)
    torch.manual_seed(SEED_BASE + args.seed_offset)

    out_path  = pathlib.Path(args.out)
    first     = True
    writer    = None
    total_t0  = time.time()
    sample_t  = mc_t = 0.0
    rows_left = args.rows

    print(f"Launching Monte-Carlo for {args.rows:,} rows …", flush=True)

    while rows_left:
        batch = min(rows_left, CHUNK_MAX)
        records = []

        for _ in range(batch):
            # ---- sample scenario
            t0 = time.perf_counter()
            p  = fg_sample()
            sample_t += time.perf_counter() - t0

            # ---- base price & SE
            t1 = time.perf_counter()
            price, price_se = price_mc(p, args.paths, return_se=True)
            mc_t += time.perf_counter() - t1

            if args.compute_greeks:
                (base, base_se,
                 delta, delta_se, vega, vega_se,
                 gamma, gamma_se, rho_v, rho_se,
                 theta, theta_se) = greeks_fd(p, args.paths)
            else:
                delta = delta_se = vega = vega_se = gamma = gamma_se = [None] * N_ASSETS
                rho_v = rho_se = theta = theta_se = None

            # ---- flatten correlation matrix
            corr_mat = p['rho'].cpu().numpy()
            corr_fields = {
                f"corr_{i}_{j}": float(corr_mat[i, j])
                for i in range(N_ASSETS) for j in range(i + 1, N_ASSETS)
            }

            # ---- assemble record
            rec = {
                **{f"S0_{i}":     p['S0'][i]     for i in range(N_ASSETS)},
                **{f"sigma_{i}":  p['sigma'][i]  for i in range(N_ASSETS)},
                **corr_fields,
                "K": p['K'],
                "r": p['r'],
                "T": p['T'],
                "price":    price,
                "price_se": price_se,
            }

            if args.compute_greeks:
                rec.update({
                    **{f"delta_{i}":    delta[i]    for i in range(N_ASSETS)},
                    **{f"delta_se_{i}": delta_se[i] for i in range(N_ASSETS)},
                    **{f"vega_{i}":     vega[i]     for i in range(N_ASSETS)},
                    **{f"vega_se_{i}":  vega_se[i]  for i in range(N_ASSETS)},
                    **{f"gamma_{i}":    gamma[i]    for i in range(N_ASSETS)},
                    **{f"gamma_se_{i}": gamma_se[i] for i in range(N_ASSETS)},
                    "rho_greek":  rho_v,
                    "rho_se":     rho_se,
                    "theta":      theta,
                    "theta_se":   theta_se,
                })

            records.append(rec)

        # ---- write Parquet
        table = pa.Table.from_pylist(records)
        if first:
            writer = pq.ParquetWriter(str(out_path), table.schema, compression='zstd')
            first = False
        writer.write_table(table)
        rows_left -= batch

    writer.close()
    print(f"Sampling: {sample_t:.1f}s | MC+Greeks: {mc_t:.1f}s")
    print(f"Wrote {args.rows:,} rows → {out_path} in {time.time() - total_t0:.1f}s")

if __name__ == "__main__":
    main()

