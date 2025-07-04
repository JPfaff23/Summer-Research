
---

## 1. Monte Carlo Setup (One-Step GBM)

We simulate $N$ correlated assets over the interval $[0,T]$ in a single log-Euler step:

1. **Correlation and normal draws.**
   $LL^\top = \Sigma$
   $Z \sim \mathcal{N}(\mathbf{0},I_N), \quad Y = LZ \quad (\text{Cov}[Y]=\Sigma)$

2. **Log-price increment.** For each asset $i = 1,\dots,N$:
   $\text{drift}_i = \left(r-\tfrac12\sigma_i^{2}\right)T, \qquad \text{diffusion}_i = \sigma_i\sqrt{T}\,Y_i$

3. **Terminal price.**
   $G_i = \ln S_{0,i} + \text{drift}_i + \text{diffusion}_i$
   $S_i(T) = e^{G_i}$

4. **Worst-of payoff.**
   $S^* = \min_{1\le i\le N} S_i(T), \qquad A = S^* - K, \qquad h = \max(A,0)$
   $D = e^{-rT}, \quad P = D\,h$

6. **Monte Carlo estimator.** Over $M$ independent paths:
   $\widehat V = \frac{1}{M}\sum_{m=1}^{M}P^{(m)} = \frac{e^{-rT}}{M}\sum_{m=1}^{M}\bigl(\min_i S^{(m)}_i(T)-K\bigr)^{+}$

---

## 2. Adjoint-Mode Delta

We need the pathwise sensitivity $\Delta_i = \partial P/\partial S_{0,i}$ via reverse-mode automatic differentiation.

1. **Initialize.**
   $\bar P = 1$

2. **Back through discount.**
   $\bar h = D\,\bar P, \qquad \bar D = h\,\bar P$

3. **Back through ReLU payoff.**
   $\bar A = \mathbf{1}_{\{A>0\}}\,\bar h$

4. **Back through subtraction.**
   $\bar S^* = \bar A, \qquad \bar K = -\,\bar A$

5. **Back through minimum.**
   $\bar S_{i}(T)=\mathbf{1}_{\{i=i^{*}\}}\;\bar S^{*},\quad i^{*}=\arg\min_{i} S_{i}(T)$

6. **Back through exponential.**
   $\bar G_i = S_i(T)\,\bar S_i(T)$

7. **Back through log-Euler step.**
   $\bar S_{0,i} = \frac{1}{S_{0,i}}\,\bar G_i$

> **Pathwise Delta**
> $\Delta_i = e^{-rT}\,\mathbf{1}_{\{A>0\}}\,\mathbf{1}_{\{i = i^{*}\}}\;\frac{S_i(T)}{S_{0,i}}$

---

## 3. Adjoint-Mode Vega

We now find $\mathrm{Vega}_i = \partial P/\partial \sigma_i$ by continuing the reverse sweep.

8. **Back through log-Euler step ($\sigma$-branch).**
   $G_i = \ln S_{0,i} + \left(r-\tfrac12\sigma_i^{2}\right)T + \sigma_i\sqrt{T}\,Y_i$
   $\frac{\partial G_i}{\partial \sigma_i} = -\,\sigma_i T + \sqrt{T}\,Y_i$
   $\bar\sigma_i = \left(-\,\sigma_i T + \sqrt{T}\,Y_i\right)\bar G_i$

> **Pathwise Vega**
> $\mathrm{Vega}_i = e^{-rT}\,\mathbf{1}_{\{A>0\}}\,\mathbf{1}_{\{i = i^{*}\}}\;S_i(T)\!\left(-\,\sigma_i T + \sqrt{T}\,(LZ)_i\right)$

---

