# Monte Carlo Simulation for Derivative Pricing  
### Rationale, Methodology, Parameter Choices, and Paths to Improvement  
*Jackson Pfaff – May 2025*  

---

## Abstract  

Monte Carlo (MC) simulation is the workhorse technique for valuing complex financial derivatives when analytic or lattice methods fail.  
This note (i) explains why MC is often the **best practical choice** in quantitative finance;  
(ii) reviews the mechanics of path generation under a risk‑neutral measure;  
(iii) discusses the specific sampling distributions adopted in Ferguson & Green (2018) when producing training data for a deep‑learning surrogate; and  
(iv) lists concrete ideas for boosting efficiency and accuracy—both in the simulation itself and in data‑generation pipelines for machine learning.

---

## 1 Why Monte Carlo?

| Criterion | Monte Carlo | PDE / Lattice | Closed‑Form |
|-----------|-------------|---------------|-------------|
| **Dimensionality** | $O(N)$ in assets; trivial to extend to 100 dims | Curse‑of‑dimensionality (grid explodes) | Limited to 1–2 dims |
| **Path dependence** | Natural and exact | Requires path augmentation | Rare |
| **Exotic pay‑offs** | Any payoff evaluable in $O(\text{paths})$ | Needs bespoke grid | Rare |
| **Model flexibility** | Swap in any SDE by changing the path generator | Hard; coefficients enter PDE | Very limited |
| **Parallelism** | Embarrassingly parallel on CPUs/GPUs | Weak (tri‑diagonal solves) | N/A |
| **Error control** | CLT ⇒ $\text{SE}\propto 1/\sqrt{N}$ | Stability / consistency issues | None |

Monte Carlo is therefore preferred when

* the payoff depends on **many** risk factors (basket, hybrid, credit–equity);
* the payoff depends on the **entire path** (Asian, look‑back, worst‑of); or
* the model contains jumps, stochastic volatility, or local‑vol surfaces.

---

## 2 Mechanics of Monte Carlo Pricing

1. **Risk‑neutral dynamics**

   $$
   dS_i(t) = r\,S_i\,dt \;+\; \sigma_i\,S_i\,dW_i(t), \qquad
   \langle dW_i\,dW_j \rangle = \rho_{ij}\,dt.
   $$

2. **Discretise** time from $0$ to $T$: $0 = t_0 < t_1 < \dots < t_M = T$ with $\Delta t$.

3. **Generate paths**

   * Draw $\mathbf Z_m \sim \mathcal N(\mathbf 0,\boldsymbol\rho)$.
   * Propagate  

     $$
     S_i(t_{m+1}) \;=\; S_i(t_m)\,
       \exp\!\Bigl[(r - \tfrac12 \sigma_i^{2})\,\Delta t
                   + \sigma_i \sqrt{\Delta t}\,Z_{m,i}\Bigr].
     $$

4. **Evaluate** the payoff $P\bigl(\{S(t_m)\}_{m=0}^{M}\bigr)$ for each path.

5. **Discount & average**

   $$
   \hat V \;=\; e^{-rT}\,\frac{1}{N}\sum_{k=1}^{N} P^{(k)}, 
   \qquad
   \text{SE} \;=\; \frac{\hat\sigma}{\sqrt{N}}.
   $$

6. *(Optional)* add variance‑reduction: antithetic variates, control variates, stratification, quasi‑MC.

---

## 3 Why These Sampling Distributions? (Ferguson & Green 2018)

| Input | Distribution | Rationale |
|-------|--------------|-----------|
| **Spot prices** | $S_{0,i}=100\,e^{Z_i}$ with $Z_i\sim\mathcal N(0.5,\,0.25)$ | Log‑normal keeps prices positive; 95 % of draws fall in \$65–\$250. |
| **Volatilities** | $\sigma_i \sim \mathrm U(0,\,1)$ (0–100 % p.a.) | Equal weight to calm and crisis regimes; avoids over‑fitting to the 20–30 % band. |
| **Maturity** | $T \sim \mathrm U(1,\,4)\text{ yrs}$, with extra weight at 1 yr | Short expiries have high gamma; the network sees steep regions more often. |
| **Correlations** | $x\sim\operatorname{Beta}(5,2)\!\to\! 2x-1$, then C‑vine build | Skew to positive (empirical) but ~15 % mass $<0$; vine ensures $\boldsymbol\rho\succ0$. |
| **Strike $K$** | Chosen so ≈ 50 % scenarios are near/ITM | Avoids a sea of zero‑payoff samples. |

---

## 4 Improvement Opportunities

### 4.1 Simulation layer

| Technique | Benefit |
|-----------|---------|
| **Quasi‑MC (Sobol)** | Error $\sim O(N^{-1}\!\log^{d}\!N)$ vs $O(N^{-1/2})$ |
| **Control variates** | Slashes variance via a correlated analytic payoff |
| **Antithetic / stratified** | Cheap 30–50 % variance drop |
| **GPU kernels** | $10^{2}\!–10^{3}\times$ speed‑up |
| **Milstein / NV schemes** | Lower discretisation bias |

### 4.2 Data‑generation / ML layer

* Latin‑hypercube and adaptive sampling  
* Importance sampling near strike / high‑vega  
* Add stochastic‑vol dynamics (Heston, SABR)  
* Label denoising via a high‑path subset  
* Physics‑informed neural nets (PDE residual penalty)

### 4.3 Hybrid ideas

*Control‑variate neural networks*—learn the residual between an analytic approximation and Monte Carlo.

---

## 5 Conclusion  

Monte Carlo remains the most versatile framework for exotic derivative pricing.  
Thoughtful parameter sampling maximises information per CPU/GPU cycle.  
Future gains lie in variance‑reduction, smarter sampling, and hybrid ML–MC architectures.

---

## References  

* Ferguson, R. & Green, A. (2018). **Deeply Learning Derivatives**. *arXiv 1809.02233*.  
* Glasserman, P. (2004). **Monte Carlo Methods in Financial Engineering**. Springer.  
* Lewandowski, D. et al. (2009). *J. Multivariate Anal.* 100 (9).  
* Jensen, K. & Petersen, M. (2023). *SSRN 4299983*.
