# Monte Carlo Simulation for Derivative Pricing  
### Rationale, Methodology, Parameter Choices, and Paths to Improvement  
*Jackson Pfaff – May 2025*  

---

## Abstract  

Monte Carlo (MC) simulation is the workhorse technique for valuing complex financial derivatives when analytic or lattice methods fail.  
This note (i) explains why MC is often the **best practical choice** in quantitative finance;  
(ii) reviews the mechanics of path generation under a risk-neutral measure;  
(iii) summarises exactly how Ferguson & Green (2018) sample model inputs; and  
(iv) lists concrete ideas for boosting efficiency and accuracy.

---

## 1 Why Monte Carlo?

| Criterion            | Monte Carlo                                   | PDE / Lattice                          | Closed-Form            |
|----------------------|-----------------------------------------------|----------------------------------------|------------------------|
| **Dimensionality**   | $O(N)$ in assets; extendable to 100 dims      | Grid explodes (curse of dimensionality) | Limited to 1–2 dims     |
| **Path dependence**  | Natural and exact                             | Requires path augmentation             | Rare                   |
| **Exotic pay-offs**  | Any payoff evaluable in $O(\text{paths})$     | Needs bespoke grid                     | Rare                   |
| **Model flexibility**| Swap in any SDE by changing generator         | Hard; coefficients enter PDE           | Very limited           |
| **Parallelism**      | Embarrassingly parallel on CPU/GPU            | Weak (tri-diagonal solves)             | N/A                    |
| **Error type**       | Statistical; CLT gives $SE \propto N^{-1/2}$  | Deterministic truncation error         | N/A                    |

MC is preferred when  
* the payoff depends on **many** risk factors (basket, hybrid, credit–equity);  
* the payoff depends on the **entire path** (Asian, look-back, worst-of); or  
* the model contains jumps, stochastic volatility, or local-vol surfaces.

---

## 2 Mechanics of Monte Carlo Pricing

1. **Risk-neutral dynamics**  
   $$dS_i(t) = r\,S_i(t)\,dt + \sigma_i\,S_i(t)\,dW_i(t),\quad
     \mathrm{Cov}\bigl[dW_i,dW_j\bigr] = \rho_{ij}\,dt.$$

2. **Discretise** time: $0 = t_0 < t_1 < \dots < t_M = T$ with step $\Delta t$.

3. **Generate paths** (log-Euler scheme)  
   $$S_i(t_{m+1})
     = S_i(t_m)\,
       \exp\!\Bigl[(r - \tfrac12\sigma_i^2)\Delta t
                   + \sigma_i\sqrt{\Delta t}\,Z_i\Bigr],
     \quad \mathbf Z\sim\mathcal N(\mathbf0,\rho).$$

4. **Evaluate** payoff  
   $$P\bigl(S(t_0),\dots,S(t_M)\bigr).$$

5. **Discount & average**  
   $$\hat V = e^{-rT}\,\frac1N\sum_{k=1}^N P^{(k)},\qquad
     SE = \hat\sigma / \sqrt N.$$

6. *(Optional)* Apply variance-reduction: antithetic variates, control variates, stratification, quasi-MC, etc.

> **Note:** Log-Euler discretisation bias is $O(\Delta t)$; Milstein or Ninomiya–Victoir schemes reduce this to $O(\Delta t^2)$.

---

## 3 Sampling Distributions (Ferguson & Green 2018)

Training inputs follow **Table 3** of Ferguson & Green (2018), for a six-asset basket call:

| Parameter type (dim)      | Distribution                                              | Rationale                                                                 |
|---------------------------|-----------------------------------------------------------|---------------------------------------------------------------------------|
| **Forward stock prices** (6) | $S_{0,i}=100\exp(Z_i),\;Z_i\sim\mathcal N(0.5,0.25)$    | Lognormal keeps $S_0>0$ and spans roughly \$65–\$250.                   |
| **Stock volatilities** (6)   | $\sigma_i\sim\mathcal U(0,1)$                          | Equal weight to calm/crisis regimes.                                      |
| **Maturity** (1)             | Draw $X\sim\mathcal U\{1,\dots,43\}$; set $T = X^2$ days | Squaring biases toward short tenors (high gamma regions).                 |
| **Pairwise correlations** (15) | $\rho_{ij} = 2\,\mathrm{Beta}(5,2)-1$; assemble via C-vine | C-vine guarantees a valid (positive-definite) matrix but allows negatives. |
| **Strike** (1)               | Fixed at $K=100$                                       | Focuses the network on learning other risk drivers.                       |

## 4 Upgrade Ideas

* **Quasi-MC (Sobol / Halton)** → error $O\bigl(N^{-1}\log^d N\bigr)$ vs.\ $O(N^{-1/2})$.  
* **Control variates** → reuse analytic prices (e.g., Black–Scholes) or low-variance replicating portfolios; F&G’s deep-learning control-variate is a modern twist.  
* **Antithetic / stratified** sampling.  
* **GPU kernels** for path generation and payoff loops.  
* **Higher-order schemes** (Milstein, Ninomiya–Victoir) to reduce time-stepping bias.  
* **Smarter input sampling** – Latin-hypercube, adaptive importance sampling, stochastic-vol dynamics.  
* **Hybrid CV–NN** – train a neural net on the residual after subtracting an analytic control variate.

---

## 5 Conclusion

Monte Carlo remains the most versatile tool for exotic derivative pricing.  
With careful input sampling (as in Ferguson & Green), variance-reduction, and modern hardware, accuracy scales almost linearly with compute.  
Surrogate ML models and control variates can further accelerate production workflows while retaining MC’s transparency and robustness.

---

## References

- Ferguson, S. & Green, J. (2018) **Deeply Learning Derivatives**, *arXiv:1809.02233*  
- Glasserman, P. (2004) **_Monte Carlo Methods in Financial Engineering_**, Springer  
- Lewandowski, D. et al. (2009) “Generating random correlation matrices based on vines and extended onion method,” *J. Multivariate Anal.* 100 (9)  
- Jensen, L. & Petersen, M. (2023) “Neural Control Variates for Monte Carlo Pricing,” SSRN 4299983  
