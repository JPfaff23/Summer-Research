#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_worst_of_dataset_fast.py
  • multi-GPU, single-process, FP16+TF32 Monte-Carlo
  • ~9 s per 100 k rows × 10 k paths × 64 steps on 4 V100s
"""

import os, math, time, argparse, pathlib, sys
import numpy as np, pandas as pd, torch, pyarrow as pa, pyarrow.parquet as pq
from torch.distributions import Beta

# ──────────────────────────── knobs ────────────────────────────
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

if not NGPU:
    sys.exit("No CUDA GPU visible – aborting.")

# ───────────────────── correlation sampler ─────────────────────
def cvine_corr(d, a=5.0, b=2.0):
    beta = Beta(torch.tensor([a], device="cuda"),
                torch.tensor([b], device="cuda"))
    P = torch.eye(d, device="cuda")
    for k in range(d-1):
        for i in range(k+1, d):
            rho = 2*beta.sample().item() - 1.0
            for m in range(k-1, -1, -1):
                rho = rho*math.sqrt((1-P[m,i]**2)*(1-P[m,k]**2)) + P[m,i]*P[m,k]
            P[k,i] = P[i,k] = rho
    ev, evec = torch.linalg.eigh(P)
    return evec @ torch.diag(torch.clamp(ev, min=1e-6)) @ evec.T


def fg_sample():
    z      = np.random.normal(0.5, math.sqrt(0.25), N_ASSETS)
    S0     = 100*np.exp(z)
    sigma  = np.random.uniform(0.0, 1.0, N_ASSETS)
    T      = (np.random.randint(1, 44)**2)/252.0
    return dict(S0=S0, sigma=sigma, T=T,
                rho=cvine_corr(N_ASSETS).cpu().numpy(),
                K=100.0, r=R_RATE)

# ────────────────── Monte-Carlo building blocks ─────────────────

@torch.no_grad()
def terminal_prices(S0, sigma, T, rho, *, n_paths, n_steps, r, Z=None):
    """Simulate terminal prices with path chunking to stay below GPU memory.``
    Each GPU handles `n_paths_per_gpu`; we further loop over smaller
    `chunk` blocks (default 5 million) so we never materialise a huge
    (n_paths×n_steps×N_ASSETS) tensor at once.
    Returns a [n_paths, N_ASSETS] FP16 tensor of terminal prices.
    """
    device = S0.device
    dt  = torch.tensor(T / n_steps, dtype=torch.float16, device=device)
    mu  = (r - 0.5 * sigma ** 2).to(torch.float16)
    sig = sigma.to(torch.float16)
    sqrt_dt = torch.sqrt(dt)
    chol = torch.linalg.cholesky(rho).to(torch.float16)

    # allocate output in chunks to avoid >12 GiB temp tensors
    chunk_size = 5_000_000  # 5 M paths ==> ~2.4 GiB for Z
    out = torch.empty(n_paths, N_ASSETS, dtype=torch.float16, device=device)

    for start in range(0, n_paths, chunk_size):
        end = min(start + chunk_size, n_paths)
        m = end - start  # paths in this chunk

        # If Z provided (antithetic shared across bumps) slice it; else sample
        Zi = (Z[start:end] if Z is not None else torch.randn(
                m, n_steps, N_ASSETS, device=device, dtype=torch.float16))

        # accumulate log‑prices path‑wise to avoid 3‑D temporary tensor
        logS = torch.log(S0).expand(m, N_ASSETS).clone().to(torch.float16)
        for k in range(n_steps):
            dW = Zi[:, k] @ chol.T  # [m, N_ASSETS]
            logS += mu * dt + sig * sqrt_dt * dW
        out[start:end] = torch.exp(logS)

    return out


def price_mc(params, n_paths, n_steps, Z=None):
    per_gpu = n_paths // NGPU
    payoffs = []
    for g, dev in enumerate(DEVICES):
        S0    = torch.tensor(params['S0'],    device=dev)
        sigma = torch.tensor(params['sigma'], device=dev)
        T     = torch.tensor(params['T'],     device=dev)
        rho   = torch.tensor(params['rho'],   device=dev)
        K, r  = params['K'], params['r']
        # if antithetic normals provided they may be only half‑paths long
        Zi    = None if Z is None else Z[g].to(dev)
        n_paths_gpu = Zi.shape[0] if Zi is not None else per_gpu

        ST = terminal_prices(
            S0, sigma, T, rho,
            n_paths=n_paths_gpu, n_steps=n_steps, r=r, Z=Zi
        )
        payoff = torch.clamp(ST.min(dim=1).values - K, 0.).cpu()
        payoffs.append(payoff)

    all_pay = torch.cat(payoffs, dim=0)
    discount = math.exp(-params['r'] * params['T'])
    return discount * all_pay.mean()
    

def greeks_fd(params, n_paths, n_steps):
    # split paths across GPUs and antithetic pairs
    per_gpu = n_paths // NGPU
    half    = per_gpu // 2
    Zpairs = [
        torch.randn(half, n_steps, N_ASSETS, device=d, dtype=torch.float16)
        for d in DEVICES
    ]

    base = price_mc(params, n_paths, n_steps, Zpairs)
    delta = np.empty(N_ASSETS)
    vega  = np.empty(N_ASSETS)
    gamma = np.empty(N_ASSETS)

    for i in range(N_ASSETS):
        hS = max(EPS_REL * params['S0'][i],    1e-6)
        hV = max(EPS_REL * params['sigma'][i], 1e-6)

        # delta & gamma bumps
        p_up = {**params, 'S0': params['S0'].copy()}; p_up['S0'][i] += hS
        p_dn = {**params, 'S0': params['S0'].copy()}; p_dn['S0'][i] -= hS
        up = price_mc(p_up, n_paths, n_steps, Zpairs)
        dn = price_mc(p_dn, n_paths, n_steps, Zpairs)
        delta[i] = (up - dn) / (2*hS)
        gamma[i] = (up - 2*base + dn) / (hS*hS)

        # vega bumps
        p_up = {**params, 'sigma': params['sigma'].copy()}; p_up['sigma'][i] += hV
        p_dn = {**params, 'sigma': params['sigma'].copy()}; p_dn['sigma'][i] -= hV
        vega[i] = (
            price_mc(p_up, n_paths, n_steps, Zpairs) -
            price_mc(p_dn, n_paths, n_steps, Zpairs)
        ) / (2*hV)

    # rho
    hR    = max(EPS_REL * abs(params['r']), 1e-6)
    rho_v = (
        price_mc({**params, 'r': params['r'] + hR}, n_paths, n_steps, Zpairs) -
        price_mc({**params, 'r': params['r'] - hR}, n_paths, n_steps, Zpairs)
    ) / (2*hR)

    # theta
    hT    = max(EPS_REL * params['T'], 1e-6)
    theta = (
        price_mc({**params, 'T': params['T'] + hT}, n_paths, n_steps, Zpairs) -
        price_mc({**params, 'T': max(1e-6, params['T'] - hT)},
                 n_paths, n_steps, Zpairs)
    ) / (2*hT)

    return base.item(), delta, vega, gamma, rho_v.item(), theta.item()
    
# ───────────────────────── driver ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows',         type=int, default=100000)
    ap.add_argument('--paths',        type=int, default=10000)
    ap.add_argument('--steps',        type=int, default=64)
    ap.add_argument('--seed_offset',  type=int, default=0)
    ap.add_argument('--out',          type=str, default='data.parquet')
    ap.add_argument('--no_chunking',  action='store_true',
                    help="run all rows in one chunk for timing")
    args = ap.parse_args()

    # auto-upgrade for VSCode runs with no args:
    if len(sys.argv) == 1:
        args.rows  = 1
        args.paths = 10000

    np.random.seed(SEED_BASE + args.seed_offset)
    torch.manual_seed(SEED_BASE + args.seed_offset)

    out_path = pathlib.Path(args.out)
    pa_writer, first = None, True

    chunk_size = args.rows if args.no_chunking else CHUNK_MAX
    total_start = time.time()
    sample_time = 0.0
    mc_time     = 0.0
    rows_left = args.rows

    print(f"⏱️  Starting Monte-Carlo for {args.rows:,} rows "
          f"(chunk_size={chunk_size})…", flush=True)

    while rows_left:
        batch = min(rows_left, chunk_size)
        recs  = []
        for _ in range(batch):
            ts = time.perf_counter();  p  = fg_sample(); sample_time += time.perf_counter() - ts
            tm = time.perf_counter();  pr, d, v, g, rho_v, th = greeks_fd(p, args.paths, args.steps); mc_time += time.perf_counter() - tm
            rec = {**{f"S0_{i}": p["S0"][i] for i in range(N_ASSETS)},
                   **{f"sigma_{i}": p["sigma"][i] for i in range(N_ASSETS)},
                   **{f"delta_{i}": d[i]         for i in range(N_ASSETS)},
                   **{f"vega_{i}":  v[i]         for i in range(N_ASSETS)},
                   **{f"gamma_{i}": g[i]         for i in range(N_ASSETS)},
                   "price": pr, "rho": rho_v, "theta": th,
                   "T": p["T"], "r": p["r"]}
            recs.append(rec)

        tbl = pa.Table.from_pylist(recs)
        if first:
            pa_writer = pq.ParquetWriter(str(out_path), tbl.schema, compression='zstd')
            first = False
        pa_writer.write_table(tbl)
        rows_left -= batch

    pa_writer.close()
    total_time = time.time() - total_start
    print(f"Total sampler time: {sample_time:.1f}s, MC+Greeks time: {mc_time:.1f}s", flush=True)
    print(f"Wrote {args.rows:,} rows → {out_path} in {total_time:.1f}s on {NGPU} GPU(s)")

if __name__ == "__main__":
    main()
