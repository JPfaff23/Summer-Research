#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Worst_of_FAST_SE_FGAD_LRM.py — Fast 2nd-order Greeks via LRM + AAD
  • multi-GPU, single-process, FP16+TF32 Monte Carlo
  • computes 1st-order Greeks via AAD and 2nd-order (Gamma) via LRM×AAD
  • streams in chunks to avoid OOM
  • writes results (price, Δ, Γ, Vega, Θ, ρ) to Parquet
"""

import os, math, time, argparse, pathlib, sys
import numpy as np
import torch
import pyarrow as pa, pyarrow.parquet as pq

# global settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark       = True
torch.set_default_dtype(torch.float32)

N_ASSETS  = 3
R_RATE    = 0.03
CHUNK_MAX = 1_000_000
NGPU      = torch.cuda.device_count()
DEVICES   = [torch.device(f"cuda:{i}") for i in range(NGPU)]
SIM_CHUNK = 1_000_000
if not NGPU:
    sys.exit("No CUDA GPU visible – aborting.")

# sample correlated GBM parameters
def cvine_corr_np(d, a=5.0, b=2.0):
    P = np.eye(d)
    for k in range(d-1):
        for i in range(k+1,d):
            rho = 2*np.random.beta(a,b)-1
            for m in range(k-1,-1,-1):
                rho = rho*math.sqrt((1-P[m,i]**2)*(1-P[m,k]**2)) + P[m,i]*P[m,k]
            P[k,i]=P[i,k]=rho
    ev,evec=np.linalg.eigh(P)
    P = evec@np.diag(np.clip(ev,1e-6,None))@evec.T
    return torch.as_tensor(P,dtype=torch.float32,device="cuda:0")

def fg_sample():
    z     = np.random.normal(0.5, math.sqrt(0.25), N_ASSETS)
    S0    = 100*np.exp(z)
    sigma = np.random.uniform(0,1,N_ASSETS)
    T     = (np.random.randint(1,44)**2)/252.0
    return dict(S0=S0, sigma=sigma, T=T, rho=cvine_corr_np(N_ASSETS), K=100.0, r=R_RATE)

# compute Greeks with LRM+AAD
def fast_greeks(params, n_paths):
    per_gpu = n_paths//NGPU
    # CPU accumulators
    price_sum = 0.0
    delta_sum = np.zeros(N_ASSETS)
    gamma_sum = np.zeros(N_ASSETS)
    vega_sum  = np.zeros(N_ASSETS)  # per-asset vega
    theta_sum = 0.0
    rho_sum   = 0.0

    for offset in range(0,per_gpu,SIM_CHUNK):
        sz = min(SIM_CHUNK, per_gpu-offset)
        for dev in DEVICES:
            # load params on GPU
            S0_t    = torch.tensor(params['S0'],   device=dev, requires_grad=True)
            sigma_t = torch.tensor(params['sigma'],device=dev, requires_grad=True)
            r_t     = torch.tensor(params['r'],    device=dev, requires_grad=True)
            T_t     = torch.tensor(params['T'],    device=dev, requires_grad=True)
            rho_t   = params['rho'].to(dev)

            # simulate GBM
            chol = torch.linalg.cholesky(rho_t)
            Z    = torch.randn(sz, N_ASSETS, device=dev)
            with torch.autocast('cuda',dtype=torch.float16):
                drift     = (r_t - 0.5*sigma_t**2)*T_t
                diffusion = sigma_t*torch.sqrt(T_t)*(Z@chol.T)
                logS      = torch.log(S0_t) + drift + diffusion
                ST        = torch.exp(logS)

            # payoff & discount
            pay      = torch.clamp(ST.min(dim=1).values - params['K'], min=0.0)
            disc_pay = torch.exp(-r_t*T_t)*pay
            price_sum += disc_pay.sum().item()

            # AAD: first-order Delta paths
            path_delta = torch.autograd.grad(
                disc_pay,
                S0_t,
                grad_outputs=torch.ones_like(disc_pay),
                retain_graph=True
            )[0]  # [sz, N_ASSETS]

            # LRM weight for S0: ∂ log f/∂ S0
            m        = (r_t - 0.5*sigma_t**2)*T_t
            ln_ratio = logS - torch.log(S0_t) - m
            L0       = ln_ratio / (sigma_t**2 * T_t * S0_t)  # [sz, N_ASSETS]

            # Gamma via LRM×Delta
            gamma_sum += (L0 * path_delta).detach().cpu().numpy().sum(axis=0)
            delta_sum += path_delta.detach().cpu().numpy().sum(axis=0)

            # Vega, Theta, Rho via autograd
            total = disc_pay.sum()
            grad_vega, grad_theta, grad_rho = torch.autograd.grad(
                total,
                [sigma_t, T_t, r_t],
                retain_graph=False
            )
            vega_sum  += grad_vega.detach().cpu().numpy()      # vector
            theta_sum += grad_theta.detach().cpu().item()
            rho_sum   += grad_rho.detach().cpu().item()

    # normalize by total paths
    D = n_paths
    return (
        price_sum/D,
        delta_sum/D,
        gamma_sum/D,
        vega_sum /D,
        theta_sum/D,
        rho_sum  /D
    )

# main driver
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=1)
    ap.add_argument('--paths',type=int, default=100_000_000)
    ap.add_argument('--out',  type=str, default='Train_Fast2nd.parquet')
    args=ap.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    out=pathlib.Path(args.out)
    writer=None; first=True; rows=args.rows; start=time.time()
    print(f"Running Fast 2nd-order Greeks MC for {rows} rows …")
    while rows:
        batch=min(rows,CHUNK_MAX); recs=[]
        for _ in range(batch):
            p = fg_sample()
            pr, d, g, v, th, r = fast_greeks(p, args.paths)
            corr = p['rho'].cpu().numpy()
            corr_f = {f"corr_{i}_{j}": float(corr[i,j])
                      for i in range(N_ASSETS) for j in range(i+1,N_ASSETS)}
            rec = {f"S0_{i}": p['S0'][i] for i in range(N_ASSETS)}
            rec.update({f"sigma_{i}": p['sigma'][i] for i in range(N_ASSETS)})
            rec.update(corr_f)
            rec.update({"K": p['K'], "r": p['r'], "T": p['T'], "price": pr})
            rec.update({f"delta_{i}": d[i] for i in range(N_ASSETS)})
            rec.update({f"gamma_{i}": g[i] for i in range(N_ASSETS)})
            rec.update({f"vega_{i}": v[i] for i in range(N_ASSETS)})
            rec.update({"theta": th, "rho": r})
            recs.append(rec)
        tbl = pa.Table.from_pylist(recs)
        if first:
            writer = pq.ParquetWriter(str(out), tbl.schema, compression='zstd')
            first = False
        writer.write_table(tbl)
        rows -= batch
    writer.close()
    print(f"Done in {time.time()-start:.1f}s")

if __name__=="__main__": main()
