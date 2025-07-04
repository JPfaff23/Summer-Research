# Monte Carlo & Adjoint-Mode Greeks  
**Jackson Pfaff – May 2025**

---

## 1. Monte Carlo Setup (One-Step GBM)

We simulate \(N\) correlated assets over the interval \([0,T]\) in a single log-Euler step:

1. **Correlation and normal draws.**  
   Let \(\Sigma\in\mathbb{R}^{N\times N}\) be the assets’ correlation matrix.  
   Compute its lower-triangular Cholesky factor \(L\) so that  
   \[
     LL^\top = \Sigma.
   \]  
   Draw  
   \[
     Z \sim \mathcal{N}(0,I_N),
     \quad
     Y = L\,Z,
     \quad
     \mathrm{Cov}[Y] = \Sigma.
   \]

2. **Log-price increment.**  
   For each asset \(i=1,\dots,N\), compute  
   \[
     \mathrm{drift}_i = \bigl(r - \tfrac12\,\sigma_i^2\bigr)\,T,
     \quad
     \mathrm{diffusion}_i = \sigma_i\,\sqrt{T}\,Y_i.
   \]

3. **Terminal price.**  
   Define the log-argument  
   \[
     G_i = \ln S_{0,i} + \mathrm{drift}_i + \mathrm{diffusion}_i,
   \]  
   then  
   \[
     S_i(T) = e^{G_i}.
   \]

4. **Worst-of payoff.**  
   Let  
   \[
     S^* = \min_{1\le i\le N} S_i(T),
     \quad
     A = S^* - K,
     \quad
     h = \max(A,0),
     \quad
     D = e^{-rT},
     \quad
     P = D\,h.
   \]

5. **Monte Carlo estimator.**  
   Over \(M\) independent paths,  
   \[
     \widehat V
     = \frac1M\sum_{m=1}^M P^{(m)}
     = \frac{e^{-rT}}{M}\sum_{m=1}^M \bigl(\min_i S_i^{(m)}(T)-K\bigr)_+.
   \]

---

## 2. Adjoint-Mode Delta

We compute the pathwise sensitivity  
\(\displaystyle \Delta_i = \frac{\partial P}{\partial S_{0,i}}\) via reverse-mode differentiation [@Capriotti2010].

1. **Initialize.**  
   \[
     \bar P = 1.
   \]

2. **Back through discount.**  
   \(P = D\,h\) ⇒  
   \[
     \bar h = D\,\bar P,
     \quad
     \bar D = h\,\bar P.
   \]

3. **Back through payoff.**  
   \(h = \max(A,0)\) ⇒  
   \[
     \bar A = \mathbf{1}_{\{A>0\}}\,\bar h.
   \]

4. **Back through subtraction.**  
   \(A = S^* - K\) ⇒  
   \[
     \bar S^* = \bar A,
     \quad
     \bar K = -\,\bar A.
   \]

5. **Back through minimum.**  
   \(S^* = \min_i S_i(T)\) ⇒  
   \[
     \bar S_i(T)
     = \mathbf{1}_{\{i=i^*\}}\,\bar S^*,
     \quad
     i^* = \arg\min_i S_i(T).
   \]

6. **Back through exponential.**  
   \(S_i(T)=e^{G_i}\) ⇒  
   \[
     \bar G_i = S_i(T)\,\bar S_i(T).
   \]

7. **Back through log-Euler.**  
   \(G_i = \ln S_{0,i} + (r-\tfrac12\sigma_i^2)T + \sigma_i\sqrt{T}\,Y_i\) ⇒  
   \[
     \bar S_{0,i}
     = \frac{1}{S_{0,i}}\,\bar G_i.
   \]

Hence  
\[
\boxed{
\Delta_i
= e^{-rT}\,\mathbf{1}_{\{A>0\}}\,\mathbf{1}_{\{i=i^*\}}\,
  S_i(T)\,\frac1{S_{0,i}}.
}
\]

---

## 3. Adjoint-Mode Vega

Continuing the reverse sweep to obtain  
\(\displaystyle \mathrm{Vega}_i = \frac{\partial P}{\partial \sigma_i}\):

8. **Back through log-Euler (σ-branch).**  
   From  
   \[
     G_i = \ln S_{0,i}
           + (r-\tfrac12\sigma_i^2)T
           + \sigma_i\sqrt{T}\,Y_i,
   \]  
   differentiate w.r.t.\ \(\sigma_i\):  
   \[
     \frac{\partial G_i}{\partial \sigma_i}
     = -\,\sigma_i\,T + \sqrt{T}\,Y_i.
   \]  
   Then  
   \[
     \bar\sigma_i
     = \frac{\partial G_i}{\partial \sigma_i}\,\bar G_i
     = \bigl(-\sigma_i\,T + \sqrt{T}\,Y_i\bigr)\,\bar G_i.
   \]

Substituting \(\bar G_i = S_i(T)\,\bar S_i(T)\) gives  
\[
\boxed{
\mathrm{Vega}_i
= e^{-rT}\,\mathbf{1}_{\{A>0\}}\,\mathbf{1}_{\{i=i^*\}}\,
  S_i(T)\,\bigl(-\sigma_i\,T + \sqrt{T}\,(LZ)_i\bigr).
}
\]

---

## References

- Capriotti, L. (2010) *Fast Greeks by Algorithmic Differentiation*, preprint.  
- Ferguson, S. & Green, J. (2018) “Deeply Learning Derivatives,” *arXiv:1809.02233*.  
- Glasserman, P. (2004) *Monte Carlo Methods in Financial Engineering*, Springer.  
