---
title: "Implementing Tax-Aware Portfolio Construction via Convex Optimization"
date: 2026-07-09
draft: true
tags: ["portfolio-optimization", "convex-optimization", "cvxpy", "finance"]
summary: "Working through Moehle, Kochenderfer, Boyd & Andersen's tax-aware rebalancing formulation, and implementing it with cvxpy."
showtoc: true
---

## Overview

Reproducing and implementing ["Tax-Aware Portfolio Construction via Convex
Optimization"](https://arxiv.org/abs/2103.14971) by Nicholas Moehle, Mykel J.
Kochenderfer, Stephen Boyd, and Andrew Ang.

The paper frames tax-aware rebalancing — deciding what to buy/sell in a
taxable account without triggering unnecessary capital-gains tax — as a
convex optimization problem, solvable at scale with off-the-shelf solvers.

## Problem setup

*(fill in: portfolio state, tax lots, the trade vector, and what "tax-aware"
means concretely — short-term vs. long-term gains, wash-sale constraints,
etc.)*

## The convex formulation

*(reproduce the paper's objective and constraints here, e.g.)*

$$
\begin{aligned}
\text{maximize} \quad & \mu^T w - \gamma \, \phi(w) - \kappa \, \tau(w) \\
\text{subject to} \quad & \mathbf{1}^T w = 1 \\
& w \in \mathcal{W}
\end{aligned}
$$

where $\phi(w)$ is a risk term, $\tau(w)$ is the realized tax-cost term, and
$\mathcal{W}$ encodes portfolio constraints (long-only, turnover limits,
sector bounds, etc.).

## Implementation

Using `cvxpy`, the paper's own tooling of choice.

```python
import cvxpy as cp
import numpy as np

# Placeholder — replace with the real tax-lot-aware formulation
w = cp.Variable(n_assets)
risk = cp.quad_form(w, Sigma)
tax_cost = ...  # piecewise-linear realized gain term

objective = cp.Maximize(mu @ w - gamma * risk - kappa * tax_cost)
constraints = [cp.sum(w) == 1, w >= 0]

prob = cp.Problem(objective, constraints)
prob.solve()
```

## Backtest / example

*(small synthetic portfolio, compare tax-aware vs. naive rebalancing,
report realized tax cost and tracking error)*

## Takeaways

*(what surprised you, what was hard to implement, what you'd do differently)*

## Code

Full implementation: *(link to separate GitHub repo once published)*
