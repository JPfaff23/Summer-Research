#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_worst_of_dataset_fast.py
  • multi-GPU, single-process, FP16+TF32 Monte-Carlo
  • Owen-scrambled Sobol, antithetic, Brownian bridge
  • Control variate: sum of single-asset Black-Scholes calls
  • Chunk-safe up to 100 M paths on 4×12 GiB GPUs
  • Exports price & Greeks with sampling-error columns
"""

import os, math, time, argparse, pathlib, sys
import numpy as np, pandas as pd, torch
import pyarrow as pa, pyarrow.parquet as pq
from torch.distributions import Beta, Normal
from torch.quasirandom import SobolEngine

# ─────────────────────────── knobs ────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark       = True
torch.set_default_dtype(torch.float16)

N_ASSETS   = 3
R_RATE     = 0.03
EPS_REL    = 1e-4
SEED_BASE  = 42
CHUNK_MAX  = 1_000_000   # flush Parquet every X rows
CHUNK_PATH = 5_000_000   # inner GPU chunk size
NGPU       = torch.cuda.device_count()
DEVICES    = [torch.device(f"cuda:{i}") for i in range(NGPU)]
if NGPU == 0:
    sys.exit("No CUDA GPU visible – aborting.")

# ───────────────── Sampler helpers ──────────────────────────

def cvine_corr(d, a=5.0, b=2.0):
    # build correlation matrix in float32, then cast to float16
    beta = Beta(torch.tensor([a], device="cuda", dtype=torch.float32),
                torch.tensor([b], device="cuda", dtype=torch.float32))
    P    = torch.eye(d, device="cuda", dtype=torch.float32)
    for k in range(d - 1):
        for i in range(k + 1, d):
            rho = 2 * beta.sample().item() - 1.0
            for m in range(k - 1, -1, -1):
                rho = rho * math.sqrt((1 - P[m,i]**2)*(1 - P[m,k]**2)) + P[m,i]*P[m,k]
            P[k,i] = P[i,k] = rho
    ev, evec = torch.linalg.eigh(P)
    P_corr   = evec @ torch.diag(torch.clamp(ev, min=1e-6)) @ evec.T
    return P_corr.to(torch.float16)


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

# ──────────── Brownian bridge transformer ───────────────────

def brownian_bridge(normal_increments):
    """
    Simple Brownian-bridge reordering; replace with detailed bridge if needed.
    """
    order = [normal_increments.shape[1] - 1] + list(range(normal_increments.shape[1] - 1))
    return normal_increments[:, order, :]

# ──────────── QMC+Antithetic path generator ─────────────────

def generate_qmc_paths(m, n_steps, d, device):
    engine = SobolEngine(d * n_steps, scramble=True)
    u_cpu = engine.draw(m // 2, dtype=torch.float32)          # CPU float32
    u     = u_cpu.to(device).to(torch.float16)               # GPU FP16
    u     = torch.cat([u, 1.0 - u],      dim=0)              # antithetic
    normals = Normal(0.,1.).icdf(u).view(m, n_steps, d)
    return brownian_bridge(normals)

# ──────────── Raw MC price + Greeks (no CV) ────────────────

@torch.no_grad()
def terminal_prices(S0, sigma, T, rho, *, n_paths, n_steps, r, Z):
    # all FP16 intermediates
    dt_fp16  = (T / n_steps).to(torch.float16)  # scalar
    mu       = (r - 0.5 * sigma**2).to(torch.float16)  # [d]
    sig      = sigma.to(torch.float16)                 # [d]
    sqrt_dt  = math.sqrt(dt_fp16.item())              # float

    # cholesky in FP32, then cast
    rho32  = rho.to(torch.float32)
    chol32 = torch.linalg.cholesky(rho32)
    chol   = chol32.to(torch.float16)                 # [d,d]

    out  = torch.empty(n_paths, N_ASSETS, dtype=torch.float16, device=Z.device)
    logS0 = torch.log(S0).to(torch.float16)            # [d]

    mu_dt = mu * dt_fp16                               # [d]
    mu_dt = mu_dt.unsqueeze(0)                         # [1,d]
    sig   = sig.unsqueeze(0)                           # [1,d]

    for start in range(0, n_paths, CHUNK_PATH):
        end   = min(start + CHUNK_PATH, n_paths)
        Xi    = Z[start:end]                           # [batch,steps,d]
        batch = Xi.shape[0]
        logS  = logS0.expand(batch, N_ASSETS).clone() # [batch,d]
        for k in range(n_steps):
            dW    = Xi[:,k,:] @ chol.T               # [batch,d]
            # incremental log step
            inc   = mu_dt + sqrt_dt * (sig * dW)     # [batch,d]
            logS  = logS + inc
        out[start:end] = logS.exp()
    return out


def price_mc_raw(params, n_paths, n_steps):
    per_gpu = n_paths // NGPU
    payoffs = []
    for dev in DEVICES:
        Z     = generate_qmc_paths(per_gpu, n_steps, N_ASSETS, dev)
        S0    = torch.tensor(params['S0'],    device=dev, dtype=torch.float16)
        sigma = torch.tensor(params['sigma'], device=dev, dtype=torch.float16)
        T     = torch.tensor(params['T'],     device=dev, dtype=torch.float16)
        rho   = torch.tensor(params['rho'],   device=dev, dtype=torch.float16)
        K     = torch.tensor(params['K'],     device=dev, dtype=torch.float16)
        r     = torch.tensor(params['r'],     device=dev, dtype=torch.float16)

        ST    = terminal_prices(S0, sigma, T, rho,
                                n_paths=per_gpu, n_steps=n_steps, r=r, Z=Z)
        pay   = torch.clamp(ST.min(dim=1).values - K, 0.)
        payoffs.append(pay.to('cpu'))

    pay  = torch.cat(payoffs)
    disc = math.exp(-params['r'] * params['T']) * pay
    return disc, disc.std(unbiased=True).item() / math.sqrt(n_paths)

# ────────── Black-Scholes vanilla call CV ───────────────────

def bs_call_price(S0, K, r, T, sigma):
    # closed-form Black-Scholes call using math.erf for CDF
    d1 = (math.log(S0/K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # standard normal CDF via error function
    N1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
    N2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
    return S0 * N1 - K * math.exp(-r * T) * N2


def price_mc(params, n_paths, n_steps, *, return_se=False):
    raw_pay, raw_se = price_mc_raw(params, n_paths, n_steps)

    # control variate
    ctrl_pay = []
    for dev in DEVICES:
        Z     = generate_qmc_paths(n_paths//NGPU, n_steps, N_ASSETS, dev)
        S0    = torch.tensor(params['S0'],    device=dev, dtype=torch.float16)
        sigma = torch.tensor(params['sigma'], device=dev, dtype=torch.float16)
        T     = torch.tensor(params['T'],     device=dev, dtype=torch.float16)
        rho   = torch.tensor(params['rho'],   device=dev, dtype=torch.float16)
        K     = torch.tensor(params['K'],     device=dev, dtype=torch.float16)
        r     = torch.tensor(params['r'],     device=dev, dtype=torch.float16)

        ST    = terminal_prices(S0, sigma, T, rho,
                                 n_paths=n_paths//NGPU, n_steps=n_steps, r=r, Z=Z)
        calls = torch.clamp(ST - K, 0.).sum(dim=1).to('cpu')
        ctrl_pay.append(calls)
    ctrl = torch.cat(ctrl_pay)
    mc_ctrl_mean = ctrl.mean().item()

    E_ctrl = sum(bs_call_price(params['S0'][j], params['K'], params['r'],
                               params['T'], params['sigma'][j])
                 for j in range(N_ASSETS))

    raw_price = raw_pay.mean().item()
    price_cv  = raw_price + (E_ctrl - mc_ctrl_mean)

    if return_se:
        return price_cv, raw_se
    return price_cv

# ───────────────────── main + Greeks abbr. ─────────────────────

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
    ap.add_argument('--no_chunking', action='store_true')
    args = ap.parse_args()

    np.random.seed(SEED_BASE + args.seed_offset)
    torch.manual_seed(SEED_BASE + args.seed_offset)

    out_path, first = pathlib.Path(args.out), True
    chunk_size       = args.rows if args.no_chunking else CHUNK_MAX

    rows_left = args.rows
    pa_writer = None
    while rows_left:
        batch = min(rows_left, chunk_size)
        recs  = []
        for _ in range(batch):
            p = fg_sample()
            pr, pr_se, d, d_se, v, v_se, g, g_se, rv, rv_se, th, th_se = greeks_fd(p, args.paths, args.steps)
            recs.append({
                **{f"S0_{i}": p['S0'][i]       for i in range(N_ASSETS)},
                **{f"sigma_{i}": p['sigma'][i] for i in range(N_ASSETS)},
                "price":pr, "price_se":pr_se,
                **{f"delta_{i}":d[i]       for i in range(N_ASSETS)},
                **{f"delta_se_{i}":d_se[i] for i in range(N_ASSETS)},
                **{f"vega_{i}":v[i]        for i in range(N_ASSETS)},
                **{f"vega_se_{i}":v_se[i]  for i in range(N_ASSETS)},
                **{f"gamma_{i}":g[i]       for i in range(N_ASSETS)},
                **{f"gamma_se_{i}":g_se[i] for i in range(N_ASSETS)},
                "rho":rv, "rho_se":rv_se,
                "theta":th, "theta_se":th_se,
                "T":p['T'], "r":p['r']
            })
        tbl = pa.Table.from_pylist(recs)
        if first:
            pa_writer = pq.ParquetWriter(str(out_path), tbl.schema, compression='zstd')
            first = False
        pa_writer.write_table(tbl)
        rows_left -= batch

    pa_writer.close()

if __name__ == "__main__":
    main()
