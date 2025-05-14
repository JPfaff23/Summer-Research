#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_worst_of_dataset_fast.py
  • multi‑GPU, single‑process, FP16+TF32 Monte‑Carlo
  • Chunk‑safe up to 100 M paths on 4×12 GiB GPUs
  • Exports price & Greeks with sampling‑error columns
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
CHUNK_MAX  = 1_000_000   # flush Parquet every X rows
CHUNK_PATH = 5_000_000   # inner GPU chunk size
NGPU       = torch.cuda.device_count()
DEVICES    = [torch.device(f"cuda:{i}") for i in range(NGPU)]

if not NGPU:
    sys.exit("No CUDA GPU visible – aborting.")

# ───────────────────── correlation sampler ─────────────────────

def cvine_corr(d, a=5.0, b=2.0):
    beta = Beta(torch.tensor([a], device="cuda"), torch.tensor([b], device="cuda"))
    P = torch.eye(d, device="cuda")
    for k in range(d - 1):
        for i in range(k + 1, d):
            rho = 2 * beta.sample().item() - 1.0
            for m in range(k - 1, -1, -1):
                rho = rho * math.sqrt((1 - P[m, i]**2) * (1 - P[m, k]**2)) + P[m, i]*P[m, k]
            P[k, i] = P[i, k] = rho
    ev, evec = torch.linalg.eigh(P)
    return evec @ torch.diag(torch.clamp(ev, min=1e-6)) @ evec.T


def fg_sample():
    z = np.random.normal(0.5, math.sqrt(0.25), N_ASSETS)
    return dict(
        S0=100 * np.exp(z),
        sigma=np.random.uniform(0.0, 1.0, N_ASSETS),
        T=(np.random.randint(1, 44)**2)/252.0,
        rho=cvine_corr(N_ASSETS).cpu().numpy(),
        K=100.0,
        r=R_RATE
    )

# ────────────────── Monte‑Carlo building blocks ─────────────────

@torch.no_grad()
def terminal_prices(S0, sigma, T, rho, *, n_paths, n_steps, r, Z=None):
    device = S0.device
    dt = torch.full((), T / n_steps, dtype=torch.float16, device=device)
    mu     = (r - 0.5 * sigma**2).to(torch.float16)
    sig    = sigma.to(torch.float16)
    sqrt_dt= torch.sqrt(dt)
    chol   = torch.linalg.cholesky(rho).to(torch.float16)

    out = torch.empty(n_paths, N_ASSETS, dtype=torch.float16, device=device)
    for start in range(0, n_paths, CHUNK_PATH):
        end = min(start + CHUNK_PATH, n_paths)
        m   = end - start
        Zi  = Z[start:end] if Z is not None else torch.randn(
              m, n_steps, N_ASSETS, device=device, dtype=torch.float16)
        logS = torch.log(S0).expand(m, N_ASSETS).clone().to(torch.float16)
        for k in range(n_steps):
            dW    = Zi[:, k] @ chol.T
            logS += mu * dt + sig * sqrt_dt * dW
        out[start:end] = torch.exp(logS)
    return out


def price_mc(params, n_paths, n_steps, Z=None, *, return_payoff=False, return_se=False):
    """Monte-Carlo worst-of price with optional payoff / SE outputs."""
    per_gpu = n_paths // NGPU
    payoffs = []
    for g, dev in enumerate(DEVICES):
        S0    = torch.tensor(params['S0'],    device=dev)
        sigma = torch.tensor(params['sigma'], device=dev)
        T     = torch.tensor(params['T'],     device=dev)
        rho   = torch.tensor(params['rho'],   device=dev)
        K, r  = params['K'], params['r']
        Zi    = Z[g] if Z is not None else None
        n_gpu = Zi.shape[0] if Zi is not None else per_gpu

        ST = terminal_prices(S0, sigma, T, rho,
                             n_paths=n_gpu, n_steps=n_steps, r=r, Z=Zi)
        payoffs.append(torch.clamp(ST.min(dim=1).values - K, 0.).cpu())

    pay = torch.cat(payoffs)
    disc_pay = math.exp(-params['r'] * params['T']) * pay

    # both payoff and SE requested
    if return_payoff and return_se:
        se = disc_pay.std(unbiased=True).item() / math.sqrt(n_paths)
        return disc_pay, se

    if return_payoff:
        return disc_pay

    price = disc_pay.mean().item()
    if return_se:
        se = disc_pay.std(unbiased=True).item() / math.sqrt(n_paths)
        return price, se
    return price

def greeks_fd(params, n_paths, n_steps):
    half = (n_paths // NGPU) // 2
    Zpairs = [torch.randn(half, n_steps, N_ASSETS, device=d, dtype=torch.float16)
              for d in DEVICES]

    pay, base_se = price_mc(params, n_paths, n_steps, Zpairs,
                            return_payoff=True, return_se=True)
    base = pay.mean().item()

    delta, delta_se = np.empty(N_ASSETS), np.empty(N_ASSETS)
    vega,  vega_se  = np.empty(N_ASSETS), np.empty(N_ASSETS)
    gamma, gamma_se = np.empty(N_ASSETS), np.empty(N_ASSETS)

    for i in range(N_ASSETS):
        hS = max(EPS_REL * params['S0'][i],    1e-6)
        hV = max(EPS_REL * params['sigma'][i], 1e-6)

        # Δ & Γ
        p_up = {**params, 'S0': params['S0'].copy()}; p_up['S0'][i] += hS
        p_dn = {**params, 'S0': params['S0'].copy()}; p_dn['S0'][i] -= hS
        pu = price_mc(p_up, n_paths, n_steps, Zpairs, return_payoff=True)
        pd = price_mc(p_dn, n_paths, n_steps, Zpairs, return_payoff=True)

        paths_d = (pu - pd)/(2*hS)
        delta[i]    = paths_d.mean().item()
        delta_se[i] = paths_d.std(unbiased=True).item()/math.sqrt(n_paths)

        paths_g = (pu - 2*pay + pd)/(hS*hS)
        gamma[i]    = paths_g.mean().item()
        gamma_se[i] = paths_g.std(unbiased=True).item()/math.sqrt(n_paths)

        # ν bump
        p_up = {**params, 'sigma': params['sigma'].copy()}; p_up['sigma'][i] += hV
        p_dn = {**params, 'sigma': params['sigma'].copy()}; p_dn['sigma'][i] -= hV
        pu = price_mc(p_up, n_paths, n_steps, Zpairs, return_payoff=True)
        pd = price_mc(p_dn, n_paths, n_steps, Zpairs, return_payoff=True)

        paths_v = (pu - pd)/(2*hV)
        vega[i]    = paths_v.mean().item()
        vega_se[i] = paths_v.std(unbiased=True).item()/math.sqrt(n_paths)

    # ρ
    hR = max(EPS_REL * abs(params['r']), 1e-6)
    pu = price_mc({**params, 'r': params['r']+hR}, n_paths, n_steps, Zpairs, return_payoff=True)
    pd = price_mc({**params, 'r': params['r']-hR}, n_paths, n_steps, Zpairs, return_payoff=True)
    paths_r = (pu-pd)/(2*hR)
    rho_v   = paths_r.mean().item()
    rho_se  = paths_r.std(unbiased=True).item()/math.sqrt(n_paths)

    # θ
    hT = max(EPS_REL*params['T'],1e-6)
    pu = price_mc({**params,'T':params['T']+hT}, n_paths, n_steps, Zpairs, return_payoff=True)
    pd = price_mc({**params,'T':max(1e-6,params['T']-hT)}, n_paths, n_steps, Zpairs, return_payoff=True)
    paths_t = (pu-pd)/(2*hT)
    theta   = paths_t.mean().item()
    theta_se= paths_t.std(unbiased=True).item()/math.sqrt(n_paths)

    return (base, base_se,
            delta, delta_se,
            vega,  vega_se,
            gamma, gamma_se,
            rho_v, rho_se,
            theta, theta_se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows',        type=int, default=100000)
    ap.add_argument('--paths',       type=int, default=10000)
    ap.add_argument('--steps',       type=int, default=64)
    ap.add_argument('--seed_offset', type=int, default=0)
    ap.add_argument('--out',         type=str, default='data.parquet')
    ap.add_argument('--no_chunking', action='store_true',
                    help="run all rows in one chunk for timing")
    args = ap.parse_args()

    np.random.seed(SEED_BASE + args.seed_offset)
    torch.manual_seed(SEED_BASE + args.seed_offset)

    out_path   = pathlib.Path(args.out)
    pa_writer, first = None, True
    chunk_size = args.rows if args.no_chunking else CHUNK_MAX

    total_start = time.time()
    sample_time = mc_time = 0.0
    rows_left   = args.rows

    print(f"⏱️  Starting Monte-Carlo for {args.rows:,} rows (chunk_size={chunk_size})…", flush=True)

    while rows_left:
        batch = min(rows_left, chunk_size)
        recs  = []
        for _ in range(batch):
            ts = time.perf_counter(); p = fg_sample(); sample_time += time.perf_counter()-ts
            tm = time.perf_counter()
            (pr,pr_se, d,d_se,v,v_se,g,g_se, rv,rv_se, th,th_se) = \
                greeks_fd(p, args.paths, args.steps)
            mc_time += time.perf_counter()-tm
            rec = {
                **{f"S0_{i}": p["S0"][i]   for i in range(N_ASSETS)},
                **{f"sigma_{i}": p["sigma"][i] for i in range(N_ASSETS)},
                "price":pr, "price_se":pr_se,
                **{f"delta_{i}":d[i]       for i in range(N_ASSETS)},
                **{f"delta_se_{i}":d_se[i] for i in range(N_ASSETS)},
                **{f"vega_{i}":v[i]        for i in range(N_ASSETS)},
                **{f"vega_se_{i}":v_se[i]  for i in range(N_ASSETS)},
                **{f"gamma_{i}":g[i]       for i in range(N_ASSETS)},
                **{f"gamma_se_{i}":g_se[i] for i in range(N_ASSETS)},
                "rho":rv, "rho_se":rv_se,
                "theta":th, "theta_se":th_se,
                "T":p["T"], "r":p["r"]
            }
            recs.append(rec)
        tbl = pa.Table.from_pylist(recs)
        if first:
            pa_writer=pq.ParquetWriter(str(out_path),tbl.schema,compression='zstd')
            first=False
        pa_writer.write_table(tbl)
        rows_left -= batch

    pa_writer.close()
    total_time = time.time() - total_start
    print(f"Total sampler time: {sample_time:.1f}s, MC+Greeks time: {mc_time:.1f}s",flush=True)
    print(f"Wrote {args.rows:,} rows → {out_path} in {total_time:.1f}s on {NGPU} GPU(s)")

if __name__ == "__main__":
    main()
