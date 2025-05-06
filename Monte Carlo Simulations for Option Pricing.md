# Monte Carlo Simulation for Derivative Pricing  
### Rationale, Methodology, Parameter Choices, and Paths to Improvement  
*Jackson Pfaff – May 2025*  

---

## Abstract  

Monte Carlo (MC) simulation is the workhorse technique for valuing complex financial derivatives when analytic or lattice methods fail.  
This note (i) explains why MC is often the **best practical choice** in quantitative finance;  
(ii) reviews the mechanics of path generation under a risk-neutral measure;  
(iii) discusses the specific sampling distributions adopted in Ferguson & Green (2018); and  
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
| **Error control**    | CLT gives $SE \propto N^{-1/2}$               | Stability / consistency issues         | None                   |

MC is preferred when  
* the payoff depends on **many** risk factors (basket, hybrid, credit–equity);  
* the payoff depends on the **entire path** (Asian, look-back, worst-of); or  
* the model contains jumps, stochastic volatility, or local-vol surfaces.

---

## 2 Mechanics of Monte Carlo Pricing

1. **Risk-neutral dynamics**  
   $dS_i(t)=r\,S_i(t)\,dt+\sigma_i\,S_i(t)\,dW_i(t),\quad \mathrm{Cov}\bigl(dW_i,dW_j\bigr)=\rho_{ij}\,dt.$

2. **Discretise** time: $0 = t_0 < t_1 < \dots < t_M = T$ with $\Delta t$.

3. **Generate paths**  
   Draw $\mathbf Z\sim\mathcal N(\mathbf{0},\rho)$ and update  
   $S_i(t_{m+1}) = S_i(t_m)\exp\bigl[(r-\tfrac12\sigma_i^2)\Delta t + \sigma_i\sqrt{\Delta t}\,Z_i\bigr].$

4. **Evaluate** payoff $P\bigl(S(t_0),\dots,S(t_M)\bigr)$.

5. **Discount & average**  
   $\hat V = e^{-rT}\frac{1}{N}\sum_{k=1}^{N}P^{(k)},\quad SE = \hat\sigma/\sqrt{N}.$

6. *(Optional)* add variance-reduction: antithetic variates, control variates, stratification, quasi-MC.

---

## 3 Sampling Distributions (Ferguson & Green 2018)

| Input            | Distribution                                                 | Why                                           |
|------------------|--------------------------------------------------------------|-----------------------------------------------|
| **Spot** $S_{0,i}$   | $100\,e^{Z_i}$ with $Z_i\sim\mathcal N(0.5,0.25)$            | Positive; spans about \$65–\$250.             |
| **Volatility** $\sigma_i$ | $\mathrm U(0,1)$                                           | Equal weight to calm & crisis regimes.        |
| **Maturity** $T$     | $T\sim\mathrm U(1,4)\,\text{yrs}$, with extra draws at 1 yr | Short tenors have high gamma.                 |
| **Correlation** $\rho$ | $x\sim\mathrm{Beta}(5,2)\to2x-1$, then C-vine               | Mostly positive but some negatives; vine ensures $\rho>0$. |
| **Strike** $K$       | Chosen so $\approx50\%$ of cases are near/ITM               | Avoids all-zero payoffs.                      |

---

## 4 Upgrade Ideas

* **Quasi-MC (Sobol)** → error $O\bigl(N^{-1}\log^{d}N\bigr)$.  
* **Control variates** → re-use analytic or low-variance pricers.  
* **Antithetic / stratified** sampling.  
* **GPU kernels** for path generation & payoff loops.  
* **Higher-order schemes** (Milstein, Ninomiya–Victoir) to cut discretization bias.  
* **Smarter data generation** – Latin-hypercube, adaptive sampling, stochastic-vol models.  
* **Hybrid CV-NN** – neural net learns residual to analytic control variate.

---

## 5 Conclusion

Monte Carlo remains the most versatile tool for exotic derivative pricing.  
With thoughtful parameter sampling and variance-reduction, accuracy scales with hardware, and surrogate ML models can accelerate production workflows.

---

## References

* Ferguson & Green (2018) **Deeply Learning Derivatives**, arXiv 1809.02233  
* Glasserman (2004) **Monte Carlo Methods in Financial Engineering**  
* Lewandowski et al. (2009) *J. Multivariate Anal.* 100 (9)  
* Jensen & Petersen (2023) SSRN 4299983  
