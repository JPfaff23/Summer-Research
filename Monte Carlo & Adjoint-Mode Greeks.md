# Monte Carlo & Adjoint-Mode Greeks  
*Jackson Pfaff – May 2025*  

---

## 1. Monte Carlo Setup (One-Step GBM)

We simulate \(N\) correlated assets over the interval \([0,T]\) in a single log-Euler step:

1. **Correlation and normal draws.**  
   Let \(\Sigma\in\mathbb{R}^{N\times N}\) be the assets’ correlation matrix.  
   Compute its lower-triangular Cholesky factor \(L\) so that \(LL^\top=\Sigma\).  
   Draw  
   \[
     Z \sim \mathcal{N}(\mathbf{0},I_N), 
     \quad 
     Y = L\,Z 
     \quad(\mathrm{Cov}[Y]=\Sigma).
   \]

2. **Log-price increment.**  
   For each asset \(i=1,\dots,N\), compute  
   \[
     \text{drift}_i = \bigl(r - \tfrac12\,\sigma_i^2\bigr)\,T,
     \qquad
     \text{diffusion}_i = \sigma_i\,\sqrt{T}\,Y_i.
   \]

3. **Terminal price.**  
   Define the log-argument  
   \[
     G_i = \ln S_{0,i} + \text{drift}_i + \text{diffusion}_i,
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
     = \frac{1}{M}\sum_{m=1}^M P^{(m)}
     = \frac{e^{-rT}}{M}\sum_{m=1}^M \bigl(\min_i S_i^{(m)}(T) - K\bigr)_+.
   \]

---

## 2. Adjoint-Mode Delta

We wish to compute the pathwise sensitivity  
\(\displaystyle \Delta_i = \frac{\partial P}{\partial S_{0,i}}\) via reverse-mode (adjoint) differentiation.

1. **Initialize.**  
   \[
     \bar P = 1.
   \]

2. **Back through discount.**  
   \(P = D\,h\) gives  
   \[
     \bar h = D\,\bar P,
     \qquad
     \bar D = h\,\bar P.
   \]

3. **Back through ReLU payoff.**  
   \(h = \max(A,0)\) gives  
   \[
     \bar A = \mathbf{1}_{\{A>0\}}\;\bar h.
   \]

4. **Back through subtraction.**  
   \(A = S^* - K\) gives  
   \[
     \bar S^* = 1\;\bar A,
     \qquad
     \bar K = -1\;\bar A.
   \]

5. **Back through minimum.**  
   \(S^* = \min_i S_i(T)\) gives, for each \(i\),  
   \[
     \bar S_i(T)
     = \mathbf{1}_{\{i=i^*\}}\;\bar S^*,
     \quad
     i^* = \arg\min_i S_i(T).
   \]

6. **Back through exponential.**  
   \(S_i(T) = e^{G_i}\) gives  
   \[
     \bar G_i = S_i(T)\,\bar S_i(T).
   \]

7. **Back through log-Euler step.**  
   \(G_i = \ln S_{0,i} + (r-\tfrac12\sigma_i^2)T + \sigma_i\sqrt{T}\,Y_i\) gives  
   \[
     \bar S_{0,i}
     = \frac{1}{S_{0,i}}\;\bar G_i,
   \]  
   which is exactly the pathwise Delta contribution.

Hence the pathwise Delta is  
\[
\boxed{
\Delta_i
= \frac{\partial P}{\partial S_{0,i}}
= e^{-rT}\;\mathbf{1}_{\{A>0\}}\;\mathbf{1}_{\{i=i^*\}}\;
  S_i(T)\;\frac{1}{S_{0,i}}.
}
\]

---

## 3. Adjoint-Mode Vega

We next compute  
\(\displaystyle \mathrm{Vega}_i = \frac{\partial P}{\partial \sigma_i}\) by continuing the reverse sweep:

8. **Back through log-Euler step (σ-branch).**  
   From  
   \[
     G_i = \ln S_{0,i}
           + \bigl(r - \tfrac12\,\sigma_i^2\bigr)\,T
           + \sigma_i\,\sqrt{T}\,Y_i,
   \]  
   we differentiate w.r.t.\ \(\sigma_i\):  
   \[
     \frac{\partial G_i}{\partial \sigma_i}
     = -\sigma_i\,T + \sqrt{T}\,Y_i.
   \]  
   Therefore the adjoint is  
   \[
     \bar\sigma_i
     = \frac{\partial G_i}{\partial \sigma_i}\,\bar G_i
     = \bigl(-\sigma_i\,T + \sqrt{T}\,Y_i\bigr)\,\bar G_i.
   \]

Substituting \(\bar G_i = S_i(T)\,\bar S_i(T)\) and \(\bar S_i(T)\) from above gives  
\[
\boxed{
\mathrm{Vega}_i
= e^{-rT}\;\mathbf{1}_{\{A>0\}}\;\mathbf{1}_{\{i=i^*\}}\;
  S_i(T)\;\bigl(-\sigma_i\,T + \sqrt{T}\,(LZ)_i\bigr).
}
\]

This completes the in-depth adjoint derivations for both Delta and Vega, following the framework of Capriotti (2010) :contentReference[oaicite:0]{index=0}.

---

## References

- Capriotti, L. (2010) “Fast Greeks by Algorithmic Differentiation,” *preprint* :contentReference[oaicite:1]{index=1}.  
- Glasserman, P. (2004) *Monte Carlo Methods in Financial Engineering*, Springer.  
- Ferguson, S. & Green, J. (2018) “Deeply Learning Derivatives,” *arXiv:1809.02233*.  
- Other references as needed.
