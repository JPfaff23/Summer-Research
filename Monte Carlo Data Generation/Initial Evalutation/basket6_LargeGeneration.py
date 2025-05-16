#!/usr/bin/env python3
"""
Generate a massive option-pricing dataset with price, Δ, Γdiag, ν, ρ, θ.

• First-order Greeks via AAD (pathwise) on GPU/CPU
• Diagonal Γ via finite differences, common random numbers
• Vectorised *exact* GBM step – no Euler loop
• Multi-process, chunked Parquet writer → minimal RAM
---------------------------------------------------------------------
Run example
-----------
python build_dataset.py \
    --rows        5_000_000 \
    --paths       30_000 \
    --chunk       25_000 \
    --jobs        16 \
    --out         greeks_5m.parquet \
    --device      cuda        # or "cpu"
"""

# ==============================================================
# 0. Imports & global knobs
# ==============================================================
import os, argparse, time, numpy as np, pandas as pd, torch
import pyarrow as pa, pyarrow.parquet as pq
from joblib import Parallel, delayed
from torch.distributions import Beta

torch.set_default_dtype(torch.double)

N_ASSETS = 6
R_RATE   = 0.03
SEED_BASE= 42

# ==============================================================
# 1. Ferguson–Green market sampler
# ==============================================================
def fg_sample(n_assets=N_ASSETS, *, rng_np=None, rng_torch=None):
    rng_np   = rng_np   or np.random
    rng_torch= rng_torch or torch
    Z      = rng_np.normal(0.5, np.sqrt(0.25), n_assets)
    S0     = 100*np.exp(Z)
    sigma  = rng_np.uniform(0, 1, n_assets)
    T      = (rng_np.randint(1, 44) ** 2) / 252.0
    beta   = Beta(torch.tensor([5.]), torch.tensor([2.]))
    C      = torch.eye(n_assets, dtype=torch.double)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            C[i, j] = C[j, i] = 2*beta.sample().item() - 1
    ev, evec = torch.linalg.eigh(C)
    C = evec @ torch.diag(torch.clamp(ev, min=1e-4)) @ evec.T
    return dict(S0=S0, sigma=sigma, T=T, rho=C.numpy(), K=100.0)

# ==============================================================
# 2. Vectorised exact-GBM Monte-Carlo (CUDA/CPU)
# ==============================================================
def mc_paths_exact(S0, sigma, T, rho, n_paths, *, r=R_RATE, device="cpu"):
    """
    Returns mean basket price across paths (n_paths, ).
    Uses one-step exact GBM; no n_steps loop.
    """
    if device.startswith("cuda"):
        S0, sigma, T, rho = map(
            lambda x: torch.as_tensor(x, device=device, dtype=torch.double),
            (S0, sigma, T, rho)
        )
    else:
        S0, sigma, T, rho = map(
            lambda x: torch.as_tensor(x, dtype=torch.double),
            (S0, sigma, T, rho)
        )
    d      = S0.numel()
    chol   = torch.linalg.cholesky(rho)
    Z      = torch.randn(n_paths, d, device=S0.device) @ chol.T
    drift  = (r - 0.5 * sigma**2) * T
    vol    = sigma * torch.sqrt(T)
    S_T    = S0 * torch.exp(drift + vol * Z)
    return S_T.mean(dim=1)                           # shape (n_paths,)

# ==============================================================
# 3. Greeks
# ==============================================================
def greeks_aad_exact(p, *, n_paths, device="cpu"):
    """price, Δ, ν, ρ, θ via AAD; no Gamma"""
    S0    = torch.tensor(p['S0'],    requires_grad=True, device=device)
    sigma = torch.tensor(p['sigma'], requires_grad=True, device=device)
    T     = torch.tensor(p['T'],     requires_grad=True, device=device)
    r     = torch.tensor(R_RATE,     requires_grad=True, device=device)
    rho_m = torch.tensor(p['rho'],   device=device)
    K     = p['K']

    basket = mc_paths_exact(S0, sigma, T, rho_m, n_paths, r=r.item(), device=device)
    price  = (torch.exp(-r*T) * torch.clamp(basket - K, 0)).mean()

    dS0, dSig, dr, dT = torch.autograd.grad(
        price, (S0, sigma, r, T), create_graph=False
    )

    return dict(price=price.item(),
                delta=dS0.detach().cpu().numpy(),
                vega=dSig.detach().cpu().numpy(),
                rho=dr.item(),
                theta=dT.item())

def gamma_fd_exact(p, *, n_paths, eps=1e-4, device="cpu"):
    """diagonal Gamma via central FD, common random numbers"""
    S0    = torch.tensor(p['S0'], device=device)
    sigma = torch.tensor(p['sigma'], device=device)
    T     = torch.tensor(p['T'], device=device)
    rho_m = torch.tensor(p['rho'], device=device)
    K, r  = p['K'], R_RATE
    d     = S0.numel()

    n_batch = 1 + 2*d
    S0_mat  = S0.repeat(n_batch, 1)
    for i in range(d):
        h = eps * S0[i]
        S0_mat[1 + 2*i,   i] += h
        S0_mat[1 + 2*i+1, i] -= h

    Z = torch.randn(n_paths, d, device=device)
    prices = torch.empty(n_batch, device=device)

    for j in range(n_batch):
        basket = mc_paths_exact(S0_mat[j], sigma, T, rho_m,
                                n_paths, r=r, device=device, )
        prices[j] = (torch.exp(-r*T) * torch.clamp(basket - K, 0)).mean()

    base  = prices[0].item()
    gamma = np.empty(d)
    for i in range(d):
        h = eps * S0[i].item()
        up, dn = prices[1 + 2*i].item(), prices[1 + 2*i + 1].item()
        gamma[i] = (up - 2*base + dn) / (h*h)
    return gamma

# ==============================================================
# 4. Worker: generate one chunk → Arrow Table
# ==============================================================
def _gen_chunk(n_rows, n_paths, seed_offset, device):
    torch.manual_seed(SEED_BASE + seed_offset)
    np.random.seed(SEED_BASE + seed_offset)
    rows = []
    for _ in range(n_rows):
        q  = fg_sample()
        g1 = greeks_aad_exact(q, n_paths=n_paths, device=device)
        g2 = gamma_fd_exact(q, n_paths=n_paths, device=device)

        rows.append({ **{f"S0_{i}": q['S0'][i]     for i in range(N_ASSETS)},
                      **{f"sigma_{i}": q['sigma'][i] for i in range(N_ASSETS)},
                      "T": q['T'], "price": g1['price'],
                      "rho_greek": g1['rho'], "theta": g1['theta'],
                      **{f"delta_{i}": g1['delta'][i] for i in range(N_ASSETS)},
                      **{f"gamma_{i}": g2[i]          for i in range(N_ASSETS)},
                      **{f"vega_{i}":  g1['vega'][i]  for i in range(N_ASSETS)} })
    return pa.Table.from_pylist(rows)

# ==============================================================
# 5. Streamed Parquet generator
# ==============================================================
def generate_dataset(total_rows, *, chunk_size, n_paths, n_jobs,
                     out_file, device):
    writer = None
    produced = 0
    for global_idx in range(0, total_rows, chunk_size*n_jobs):
        rows_left = total_rows - global_idx
        n_chunks  = (rows_left + chunk_size - 1) // chunk_size
        worksize  = min(n_chunks, n_jobs)

        tables = Parallel(n_jobs=worksize, backend="loky")(
            delayed(_gen_chunk)(min(chunk_size, rows_left - i*chunk_size),
                                n_paths,
                                seed_offset=global_idx + i,
                                device=device)
            for i in range(worksize)
        )

        for tbl in tables:
            if writer is None:
                writer = pq.ParquetWriter(out_file, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            produced += tbl.num_rows
            print(f"\r{produced:,}/{total_rows:,} rows written", end="")
    if writer:
        writer.close()
    print("\nDone.")

# ==============================================================
# 6. CLI
# ==============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Massive option Greeks dataset builder")
    p.add_argument("--rows",   type=int, required=True, help="total rows to generate")
    p.add_argument("--paths",  type=int, default=30_000, help="Monte-Carlo paths per row")
    p.add_argument("--chunk",  type=int, default=25_000, help="rows per worker chunk")
    p.add_argument("--jobs",   type=int, default=os.cpu_count(), help="parallel workers")
    p.add_argument("--out",    type=str, required=True, help="output Parquet file")
    p.add_argument("--device", type=str, choices=["cpu","cuda"], default="cpu",
                   help="compute device (use 'cuda' if GPU available)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available – falling back to CPU.")
        args.device = "cpu"

    t0 = time.time()
    generate_dataset(args.rows,
                     chunk_size=args.chunk,
                     n_paths=args.paths,
                     n_jobs=args.jobs,
                     out_file=args.out,
                     device=args.device)
    print(f"Total elapsed: {time.time() - t0:.2f} s")
