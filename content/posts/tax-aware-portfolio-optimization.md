---
title: "Convex Envelopes"
date: 2026-07-09
draft: false
tags: ["convex-optimization", "cvxpy", "portfolio-optimization"]
#summary: "One nonconvex term, and the trick for getting rid of it."
showtoc: false
---

I came across the idea of using convex envelopes while reading Moehle, Kochenderfer, Boyd & Ang's [tax-aware rebalancing problem](https://arxiv.org/abs/2008.04985) paper.
Tax-aware rebalancing adds realized tax to a portfolio rebalancing problem. The optimizer can harvest losses and defer gains while trading toward a benchmark.

Adding tax-loss harvesting to a standard Markowitz-style objective introduces a nonconvex kink. The paper handles it with a two-step optimization program, starting with a convex envelope.
Selling a stock lot can realize either a tax liability (sold at a gain) or a tax benefit (sold at a loss).
Buying has no tax cost. That asymmetry shows up at zero i.e. at the sign flip of the trade, making the tax term a nonconvex function of the traded amount.

### Non-convexity of tax term

Assuming a simple benchmark-relative setup with no return forecast, the objective is to minimise risk, transaction cost, and tax together:

$$\min_{u}\; \gamma\,\underbrace{(h - h^{b} + u)^\top X \Sigma X^\top (h - h^{b} + u)}_{\text{factor risk (not separable)}} \;+\; \sum_i \Big[\, \gamma\, D_{ii}\,(h_i - h_i^{b} + u_i)^2 + \gamma_{\text{tc}}\,\kappa_i\,|u_i| + \gamma_{\text{tax}}\, L_i(u_i) \,\Big]$$

where $h$ is the pre-trade holdings and $u$ the trade. Separating into asset-specific terms, the cost of trading $u_i$ dollars of asset $i$ (with $u_i < 0$ a sale) becomes:

$$f_i(u_i) = \underbrace{\gamma\, D_{ii}\,(h_i - h_i^{b} + u_i)^2 + \gamma_{\text{tc}}\,\kappa_i\,|u_i|}_{\text{convex}} \;+\; \gamma_{\text{tax}}\, L_i(u_i).$$

All terms except the factor risk are separated out. In this asset-specific objective, everything but the tax term $L_i$ is convex.

For lot $j$ of asset $i$, let $q_{ij}$ be the number of shares, $b_{ij}$ its basis, and $p_i$ the current price. Its dollar value and tax rate per dollar sold are

$$v_{ij} = p_iq_{ij}, \qquad t_{ij} = \rho_{ij}\left(1 - \frac{b_{ij}}{p_i}\right),$$

where $\rho_{ij}$ is the short- or long-term tax rate. A loss lot has $t_{ij} < 0$. Order the lots least-tax-first, so $t_{i1} \leq \cdots \leq t_{im}$. The tax cost is then

$$L_i(u_i) =
\begin{cases}
0, & u_i \geq 0, \\
\displaystyle \sum_{j=1}^{m} t_{ij}\left[\min\left(-u_i, \sum_{\ell \leq j}v_{i\ell}\right) - \sum_{\ell < j}v_{i\ell}\right]_+, & u_i < 0,
\end{cases}$$

where $[x]_+ = \max(x, 0)$. This walks through the cheapest tax lots first, so $L_i$ is piecewise linear. On the selling side, it is convex.

At $u_i = 0$ the convexity can break. If the first lot sold is a loss lot, then

$$L_i'(0^-) = -t_{i1} > 0 = L_i'(0^+).$$

The slope drops as the trade crosses zero. That creates a concave kink, which makes the problem unsuitable for convex optimizers like cvxpy.

### Visualizing

Geometrically, the convex envelope is the lower edge of the convex hull of the graph and gives us the best convex under-estimate. 
If we picture the outline of the tax term and let a tight string wrap underneath it, the string sticks to the parts that are already convex. Around zero it stretches across the dent creating the convex envelope.  

First, we ignore the quadratic risk and look only at the tax term:

<img src="/images/envelope_string_tax_only.png" alt="Tax term only: the tax cost is flat on the buy side, kinked at zero, and its convex envelope pulls under the concave region like a string." width="78%" style="display:block; margin:0.5rem auto;" />

Separating out the per-asset term allows each asset's tax term to *borrow* curvature from the quadratic risk term. The quadratic supplies a lot of curvature, so the envelope only has to 
bridge the small concave region around zero and leaves everything else alone. Compared to enveloping the tax term by itself, the relaxation is much tighter.

### ConvexHull

Sample $f_i$ on a grid, take the convex hull, keep the lower chain:

```python
from scipy.spatial import ConvexHull
import numpy as np

def convex_envelope(x, f):
    pts = np.column_stack([x, f])
    hull = ConvexHull(pts)
    v = hull.vertices                        # hull points, counter-clockwise
    lo, hi = np.argmin(pts[v, 0]), np.argmax(pts[v, 0])
    v = np.roll(v, -lo)                       # start at the leftmost point
    lower = v[: ((hi - lo) % len(v)) + 1]     # ...walk round to the rightmost
    lower = lower[np.argsort(pts[lower, 0])]
    return np.interp(x, pts[lower, 0], pts[lower, 1])
```

The SciPy function `ConvexHull` constructs the convex envelope for us.
The vertices are returned counter-clockwise, so starting at the leftmost point and walking to the rightmost traces the 
bottom of the hull: the lower envelope we want. Interpolating at the points of the original curve gives the envelope values at the original `x` points. 

To test this, I used the mixed-lot asset from a toy six-asset account. The account is about 1.20M dollars including cash; this asset is a 222K dollar position. Its lots are:

| Lot | Dollar value | Effective tax rate |
| --- | ---: | ---: |
| Short-term loss | 66.7K dollars | -12.24% |
| Long-term gain | 66.7K dollars | 5.95% |
| Long-term gain | 89.0K dollars | 14.28% |

Run `ConvexHull` on the full per-asset cost $f_i$ (tax bundled with the quadratic risk term) and the envelope comes out almost on top of the original. The left panel is the cost scale; the right panel plots the dollar gap $f_i - f_i^{**}$ directly. The max correction is only about 31 dollars, right at the kink.

<img src="/images/convex_envelope_with_gap.png" alt="The full per-asset cost and its convex envelope overlap at normal scale; a second panel shows the small envelope gap localized near the kink." width="92%" style="display:block; margin:0.5rem auto;" />

### Takeaway

Swap every $f_i$ for its envelope and the problem is convex again. Because the envelope sits below $f_i$, the relaxed optimum is a lower bound on the true minimum.
Write $F$ for the original objective and $F^{**}$ for the relaxed one. The two-step method is

$$\hat{u} = \mathop{\arg\min}_u F^{**}(u), \qquad z_i = \operatorname{sign}(\hat{u}_i), \qquad u^+ = \mathop{\arg\min}_u F(u) \quad \text{subject to} \quad z_i u_i \geq 0.$$ 

The relaxed solution chooses a direction for each trade. Once buys and sells are fixed, the original problem is convex again since the kink at zero is gone. On my toy problem this two-step process recovered the true optimum exactly — checked against a brute-force search over every buy/sell combination.

The nonconvexity came from a buy-or-sell choice hiding at $u_i = 0$. 
I suspect this is the reusable part: when a portfolio constraint is only hard because a trade can go either direction, solve the envelope version first and use it to decide which side of zero each trade belongs on.  
