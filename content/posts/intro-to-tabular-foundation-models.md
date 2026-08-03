---
title: "Tabular Foundation Models"
date: 2026-08-01
draft: false
tags: ["machine-learning", "tabular-data", "foundation-models"]
summary: "An introduction to how tabular foundation models use pretraining and in-context learning, illustrated through OLS and a TabPFN example."
showtoc: false
---

Tabular foundation models aim to do for tabular tasks what GPT-style models do for language: train one model that can be 
reused across new tasks by providing the right context.

At its core, a TFM is a neural network pretrained to solve prediction problems on unseen tabular tasks. 
Given a labeled dataset, models such as TabPFN and TabICL can predict the labels for test rows without
updating their model weights for that table. I will use TFM as a shorthand for these models going ahead. 

Readers who have trained neural networks on tabular datasets will know how dataset specific the setup can be: input dimensions, preprocessing decisions, 
architecture, etc. Given this, how can a pretrained model work with tables it has never encountered before? Where does the learning happen, and what do 
`fit()` and `predict()` mean in this setting? 
This post aims to answer some of these questions. We motivate this discussion using model benchmarking results, discuss linear regression as a baseline followed by   
TFM pretraining, and finally run TabPFN on a toy dataset for loan defaults. 

### Benchmark results
In a [peer-reviewed study of TabPFN](https://www.nature.com/articles/s41586-024-08328-6), the model's default configuration outperformed XGBoost, CatBoost, LightGBM and other baselines 
that were given four hours of tuning time. 
This covered 29 classification and 28 regression datasets from AutoML and OpenML CTR23 benchmarks. The datasets had up to 10,000 rows 
and 500 feature columns, with classification limited to 10 classes.

{{< figure
    src="/images/tabular-foundation-models/tabpfn-benchmark-performance.png"
    link="https://www.nature.com/articles/s41586-024-08328-6/figures/4"
    target="_blank"
    alt="Bar charts comparing the classification and regression performance of TabPFN with default and four hour tuned baselines"
    caption="TabPFN compared with default and four hour tuned baselines. Cropped from the original; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)."
    attr="Hollmann et al. (2025), Figure 4a."
    attrlink="https://www.nature.com/articles/s41586-024-08328-6/figures/4"
>}}

The paper reports an average fit + predict time of 2.8s for classification, and 4.8s for regression. In fairness, these numbers do not account for 
the one-time pretraining cost of 2 weeks on eight NVIDIA RTX 2080 GPUs for this experiment.  

This does not mean that TFMs win on every task. The same paper notes that boosted trees and AutoML can perform better on tasks 
involving larger datasets or highly non-smooth regression problems.
The independent [TabArena benchmark](https://arxiv.org/abs/2506.16791) reaches a similar conclusion that gradient boosted trees remain strong contenders for 
practical tabular tasks.

### Linear regression: fit and predict

Consider linear regression using OLS. Suppose the data is generated from a linear relationship:

$$ y = X \beta + \epsilon $$

Fitting this model means using the training data to estimate the unknown coefficient vector $\beta$. OLS chooses the 
estimator $\hat{\beta}$ such that it minimizes the sum of squared differences between the observed targets and the model's 
fitted values. Assuming $X$ has full column rank, the solution is:

$$
\hat{\beta}
= \underset{\beta}{\operatorname{arg\,min}}\;\lVert y-X\beta\rVert_2^2
= (X^\top X)^{-1}X^\top y.
$$

After `fit()` has stored $\hat{\beta}$, predicting targets for new feature rows $X_{\mathrm{test}}$ requires only:

$$
\hat{y}_{\mathrm{test}} = X_{\mathrm{test}}\hat{\beta}.
$$

For this model, the original training data can be discarded as $\hat{\beta}$ contains the information required for prediction.

![Ordinary model workflow: labeled data is used to fit dataset-specific parameters before making predictions](/images/tabular-foundation-models/ordinary_models.png)

### TFM pretraining
A TFM is pretrained on many tabular tasks. For each task, the model receives labeled data $D  = (X, y)$, query row $x_*$, 
and its target $y_*$. 
Its goal is to predict the target while adjusting the shared model weights $\phi$. <br>

Note: I intentionally use the term "labeled dataset" when used in context of TFMs to distinguish them from the synthetic datasets
used during pretraining. 

By repeating this process across many different tasks, the model learns a procedure for making predictions from labeled 
examples rather than learning a fixed relationship for a specific task. 

In a Bayesian model, we generally begin with a prior $p(\theta)$ over unknown model parameters. After observing labeled  
dataset $D$, we update our beliefs about possible parameter values to get the posterior $p(\theta | D)$:

$$
p(\theta | D) \propto p(D | \theta) p (\theta)
$$

The posterior predictive distribution then averages over the possible parameter values to get a distribution for $y_*$

$$
p(y_* | x_*, D) = \int p(y_* | x_*, \theta) p (\theta | D)  d\theta
$$

We can now extend this Bayesian view by introducing uncertainty over the task space. If we let $\tau$ denote a possible data generating task, 
then the synthetic task generator used in pretraining defines the prior $p(\tau)$ over these tasks. 
Under this prior, when the TFM sees a new labeled dataset $D$ at inference, some tasks become more plausible than others. 

$$
p(\tau | D) \propto p(D | \tau) p (\tau)
$$

Rewriting the earlier prediction equation to introduce this averaging over possible tasks, we get the distribution for $y_*$

$$
q_{\phi}(y_* | x_*, D) \approx \int_{\tau} \left[ \int_{\theta} p(y_* | x_*, \theta, \tau) p (\theta | D, \tau)  d\theta  \right] p(\tau | D)  d\tau
$$

The inner integral averages over parameter values within a task while the outer integral averages over tasks under the posterior $p(\tau \mid D)$. 
Reading the equation from right to left:
- $p(\tau \mid D)$ is the posterior distribution over possible tasks after observing the labeled dataset $D$
- $p(\theta \mid D,\tau)$ is the posterior distribution over the parameters within task $\tau$
- $p(y_* \mid x_*,\theta,\tau)$ is the conditional distribution of the target $y_*$ for query row $x_*$, given a task and 
its parameters
- $q_{\phi}(y_* \mid x_*, D)$ is the predictive distribution produced by the TFM, where $\phi$ denotes its pretrained weights

Note that this integral is not actually evaluated during inference. Under Bayesian interpretation, the model learns the weights 
$\phi$ that map $D$ and $x_*$ to a predictive distribution. 

![Tabular foundation model workflow: shared weights are pretrained across many tasks, then combined with a labeled table and query for in-context prediction](/images/tabular-foundation-models/tfm.png)

### Loan default example
Suppose we have a dataset on loan defaults:

| Income ($k) | Debt-to-income | Late payments | Default |
|------------:|---------------:|--------------:|:-------:|
| 82          | 0.21           | 0             | No      |
| 39          | 0.68           | 3             | Yes     |
| 64          | 0.32           | 0             | No      |
| 46          | 0.57           | 2             | Yes     |
| 71          | 0.29           | 1             | No      |
| **51**      | **0.48**       | **1**         | **?**   |

The first five rows form $D = (X, y)$. The last row is the query $x_*$ for which we want the probability of default.   

The interface looks identical to that used by regular machine-learning models:

```python
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

In this TabPFN run, `fit()` validates, preprocesses, and stores the labeled dataset as context without changing the pretrained weights.
The main step of inference happens when we call `predict()` which combines the query with the labeled dataset:

$$
q_{\phi} \left( y_* \mid x_*, X_{\mathrm{train}}, y_{\mathrm{train}} \right)
$$

This is the key contrast with the OLS example. TabPFN still needs the information in the labeled dataset when `predict()` runs, 
either as the original rows or through a cached representation. <br>

Using TabPFN on this task gives a default probability of 0.510 for the query row; close to a coin flip. <br>
Next, I flip the label for Row 3 from No to Yes. The default probability now rises to 0.674, a move of +0.164 towards default.<br> 
Only Row 3's label changed while the query itself and the rest of the context remained as-is. This process is in-context learning.<br>
The move in default probability seems reasonable. Row 3 has better financial health across all features compared to the query row, 
so flipping it to a default should intuitively increase the probability. 

<img src="/images/tabular-foundation-models/loan_example.png" alt="Terminal output: probability of default moves from 0.510 to 0.674 when one context label is flipped, with fit and predict timings" width="420" style="display:block; width:100%; max-width:420px; height:auto; margin:1rem auto;" />

On my machine, once the pretrained model was loaded, `fit()` took ~0.22s and `predict()` ~0.43s averaged over 10 runs. 
This is consistent with in-context computation happening during the prediction step, making the step slower.  

### Takeaway

Typically, working on a new tabular task involves fitting and tuning a model for that dataset. 
TFMs split this workflow into two stages: learning shared weights during pretraining across many tasks, then keeping those 
weights fixed while generating predictions on new tasks.

The result is a familiar `fit() / predict()` interface that makes it easy to test with your existing tabular workflows.  

Some resources to continue exploring tabular foundation models:

- [TabArena leaderboard](https://huggingface.co/spaces/TabArena/leaderboard)
- [Prior Fitted Networks paper](https://arxiv.org/abs/2112.10510)
- [TabPFN on GitHub](https://github.com/PriorLabs/TabPFN)
