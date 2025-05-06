---
title: |
  Monte Carlo Simulation for Derivative Pricing:  
  Rationale, Methodology, Parameter Choices, and Paths to Improvement
author:  Jackson Pfaff  
date:    May 2025  
---

## Abstract
Monte Carlo (MC) simulation is the workhorse technique for valuing complex financial derivatives when analytic or lattice methods fail.  
This paper (i) justifies why MC is often the *best practical choice* in modern quantitative finance,  
(ii) reviews the mechanics of path generation under a risk‑neutral measure,  
(iii) explains the specific sampling distributions adopted in Ferguson & Green (2018) when generating training data for a deep‑learning surrogate, and  
(iv) outlines concrete avenues for improving efficiency and accuracy—both at the simulation layer and in the data‑generation strategy for machine‑learning pipelines.

---

## 1 Why Monte Carlo?

| Criterion | Monte Carlo | PDE / Lattice | Closed‑Form |
|-----------|-------------|---------------|-------------|
| **Dimensionality** | $O(N)$ in assets; trivial to extend to 100‑dim | Curse‑of‑dim. (grid explodes) | Limited to 1–2 dims |
| **Path dependence** | Natural and exact | Requires path augmentation | Rare |
| **Exotic pay‑offs** | Any payoff evaluable in $O(\text{paths})$ | Needs bespoke grid | Rare |
| **Model flexibility** | Swap in any SDE by changing path‑generator | Hard; coeffs enter PDE | Very limited |
| **Parallelism** | Embarrassingly parallel across CPUs/GPUs | Weak (tri‑diagonal solves) | N/A |
| **Error control** | CLT: $\text{SE}\propto 1/\sqrt{N}$ | Stability/consistency issues | None |

Hence MC is *dominant* whenever  

* the payoff depends on many risk factors (basket, hybrid, credit‑equity);  
* the payoff depends on the entire path (Asian, lookback, worst‑of);  
* the model includes jumps, stochastic vol, or local‑vol surfaces.

---

## 2 Mechanics of Monte Carlo Pricing

1. **Risk‑neutral dynamics**  

   $$
   dS_i(t) = r\,S_i\,dt + \sigma_i S_i\,dW_i(t),\quad 
   \langle dW_i\,dW_j\rangle = \rho_{ij}\,dt.
   $$

2. **Discretise** time into $0=t_0<t_1<\dots<t_M=T$ with $\Delta t$.

3. **Generate paths**  

   * Draw $\mathbf Z_m \sim \mathcal N(\mathbf 0,\boldsymbol\rho)$.  
   * Propagate  

     $$
     S_i(t_{m+1}) = S_i(t_m)\,
       \exp\!\bigl[(r-\tfrac12\sigma_i^{2})\Delta t
                   +\sigma_i\sqrt{\Delta t}\,Z_{m,i}\bigr].
     $$

4. **Evaluate payoff** $P(\{S(t_m)\}_{m=0}^M)$ for each path.

5. **Discount & average**

   $$
   \hat V = e^{-rT}\,\frac1N\sum_{k=1}^{N} P^{(k)}, 
   \qquad 
   \text{SE} = \frac{\hat\sigma}{\sqrt{N}}.
   $$

6. *(Optional)* variance‑reduction: antithetic variates, control variates, stratification, quasi‑MC.

---

## 3 Why These Sampling Distributions? *(Ferguson & Green 2018)*

| Input | Distribution | Justification |
|-------|--------------|---------------|
| **Spot prices** | $S_{0,i}=100\,e^{Z_i},\; Z_i\sim\mathcal N(0.5,0.25)$ | Log‑normal respects positivity and typical equity scale. 95 % of draws $\in[65,250]$. |
| **Volatilities** | $\sigma_i \sim \mathrm U(0,1)$ (0–100 % p.a.) | Equal weight to calm and crisis regimes; prevents over‑focus on 20–30 % band. |
| **Maturity** | $T \sim \mathrm U(1,4)\ \text{yrs}$ with oversampling at 1 yr | Short expiries $\Rightarrow$ high gamma; network sees steep regions. |
| **Correlations** | $x\sim\operatorname{Beta}(5,2)$, map $2x-1$, build via C‑vine | Skew toward positive (empirical), yet 15 % mass $<0$. Vine guarantees $\mathbf\rho\succ 0$. |
| **Strike $K$** | Set so $\approx50\%$ scenarios near/ITM | Avoids trivial zero‑payoff samples. |

---

## 4 Improvement Opportunities

### 4.1  Simulation layer

| Technique | Benefit |
|-----------|---------|
| Quasi‑MC (Sobol) | Error $\sim O(N^{-1}\log^d N)$ vs $N^{-1/2}$ |
| Control variates | Variance slashed via correlated analytic payoff |
| Antithetic / stratified | Cheap 30–50 % variance drop |
| GPU kernels | $10^2$–$10^3\times$ speed‑up |
| Milstein / NV schemes | Lower discretisation bias |

### 4.2  Data‑generation / ML layer

* Latin‑hypercube & adaptive sampling  
* Importance sampling near strike/high‑vega  
* Augment with stochastic‑vol models  
* Label denoising via high‑path subset  
* Physics‑informed neural nets

### 4.3  Hybrid ideas

Control‑variate neural networks that learn the residual between an analytic approximation and MC.

---

## 5 Conclusion
Monte Carlo remains the most versatile framework for exotic derivative pricing.  
Thoughtful parameter sampling maximises information per CPU/GPU cycle.  
Future gains lie in variance‑reduction, smarter sampling, and hybrid ML–MC architectures.

---

## References
* Ferguson, R. & Green, A. (2018). **Deeply Learning Derivatives**. *arXiv:1809.02233*.  
* Glasserman, P. (2004). **Monte Carlo Methods in Financial Engineering**. Springer.  
* Lewandowski, D. et al. (2009). *J. Multivariate Anal.* 100 (9).  
* Jensen, K. & Petersen, M. (2023). *SSRN 4299983*.
