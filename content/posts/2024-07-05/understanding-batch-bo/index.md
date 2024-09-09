---
date: "2024-07-06"
title: "Understanding batch Bayesian optimization"
slug: batch-bo
images:
- static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
description: A brief introduction to batch Bayesian optimization, comparing it against CMA-ES
---

By its nature, Bayesian optimization is _sequential_, evaluating 1 potentially good point in input space at a time. This is rather unfortunate, because several problems out there are well suited for evaluating large batches at once. Think for example of Reinforcement learning environments: an agent can be trained on several runs at the same time using e.g. [Isaac Gym]() or [Brax](). Other examples come from biology and chemistry, where practitioners evaluate their assays in plates that can hold up to 96 "experiments" at the same time.[^an-example]

[^an-example]: Check [this paper]() and [this technology]().

[Image of a plate, or image of several robots being trained with caption: sometimes we can parallelize the evaluation of our objective.]

Since there's a big practical need for batching BO, several methods have been proposed since the 2000s. In this blogpost I explain some of them, explore the tools already available in frameworks like `botorch`, and compare against the default choice for evolutionary strategies: CMA-ES.

The code used for this blogpost is shared on [the code companion](). To keep things interesting, I'll implement most of this blogpost in `jax`.

# Bayesian optimization: a recap

But first, let's start with a brief recap of Bayesian optimization to set up the notation. I'll go pretty fast here, but [one of my previous blogposts gives an introduction to BO if you're interested](/blogposts/2023-07-31/intro-to-bo).

There are three ingredients in BO:
- An objective function {{< katex >}}f\colon\mathbb{R}^D\to\mathbb{R}{{< /katex >}} which is expensive to query and potentially noisy, and which you're interested in maximizing.
- A surrogate model (usually Gaussian Process Regression) of the objective function {{< katex >}}\tilde{f}\sim\text{GP}(\mu, k){{< /katex >}}.
- An acquisition function {{< katex >}}\alpha(\bm{x};\tilde{f}){{< /katex >}} which uses the approximation of the objective function to propose new points to evaluate in. Acquisition functions are maximized at points {{< katex >}}\bm{x}\in\mathbb{R}^D{{< /katex >}} that have potential to maximize the objective.

With these three ingredients, we optimize the objective function {{< katex >}}f{{< /katex >}} **sequentially** by querying the best point {{< katex >}}x_{\text{next}}{{< /katex >}} according to the acquisition function {{< katex >}}\alpha{{< /katex >}}, and adding the pair {{< katex >}}(x_{\text{next}}, f(x_{\text{next}})){{< /katex >}} to the dataset.

Consider a two-dimensional example given by e.g. `cross-in-tray` with noisy observations. The original function is given by[^reference-to-toy-cont] 
{{< katex display >}}
f(\bm{x}) := \text{\texttt{Cross-in-tray}}(\bm{x}) = \left|\sin(x_1)\sin(x_2)\exp\left(\left|10 - \frac{\sqrt{x_1^2 + x_2^2}}{\pi}\right|\right)\right|^{0.1},
{{< /katex >}}
which has four optima at {{< katex >}}(\pm 1.349..., \pm 1.349){{< /katex >}}. Visually:

{{< figure src="/static/assets/batch_bo_blogpost/cross_in_tray_2d.jpg" alt="A timeline of high-dimensional Bayesian optimization." class="largeSize" title="Cross-in-tray as a surface, and as a contour plot. x marks the spot(s)." >}}

[^reference-to-toy-cont]: Taken from [here]().

We usually kickstart BO with either available data, or a collection of informative points. Assume we sample 10 different points at random from a SOBOL sequence[^whats-a-sobol-sequence], then one potential GP approximation using default mean and kernel choices would look like this:[^technical-details]

{{< figure src="/static/assets/batch_bo_blogpost/cross_in_tray_2d_gp.jpg" alt="A timeline of high-dimensional Bayesian optimization." class="largeSize" title="Some SOBOL samples used to approximate cross-in-tray for starters." >}}

[^whats-a-sobol-sequence]: ...reference to SOBOL sequences.

[^technical-details]: ...Check the code snippets.

Let's optimize this function using sequential Bayesian optimization. For starters, we can optimize `cross-in-tray` using Thompson Sampling as an acquisition function.

[gif]

With enough samples, we can usually find a suitable optimum for `cross-in-tray`.

As we have described it so far, this framework only allows us to query a single point {{< katex >}}x_{\text{next}}{{< /katex >}} at each iteration. The main reason: we choose to focus on the maximum of the acquisition function. To opimize in a batch, it's necessary to either (i) consider a different acquisition function that works on batches instead of single points, or (ii) squeeze more information out of the acquisition functions we have already defined.

I'll now dive deeper into versions of these modifications, starting with how some acq. functions are easily parallelizable, following up with batch versions of acq. functions, and finishing with alternatives that modify the acq. functions we already have.

# Easy batching with Thompson sampling

Thompson sampling (TS) consists of sampling from the surrogate model {{< katex >}}\tilde{f}\sim\text{GP}(\mu, k){{< /katex >}}, and optimizing said sample. Each sample from the GP posterior renders a function that *looks like* the objective function. Here are three samples of the objective after the first 10 SOBOL samples.

[Three samples in 2 and 3d]

We can simply sample several different posteriors, optimize them, and use their maxima as a batch. Here's a gif showcasing this with a batch size of 6.

<video width="600" height="auto" controls>
    <source src="/static/assets/batch_bo_blogpost/batch_ts.mp4", type="video/mp4">
</video>

According to [Kandasamy et al. (2017)](https://arxiv.org/abs/1705.09236), batch TS is almost just as good as sequential TS (indeed, the expected regret is equal up to a multiplicative constant which increases with the batch size). I've found people applying this version of batch TS in several settings (e.g. Chemistry and drug discovery by [Hernández-Lobato et al. 2017](https://arxiv.org/abs/1706.01825)). One important example is `TuRBO` by [Eriksson et al.](), a competitive alternative for high-dimensional Bayesian optimization which uses trust regions for optimizing the acquisition function.

This is **the simplest way** to batch BO. These samples can be taken either synchronously or asynchronously and fully in parallel (something that can't be said about the other methods presented here). Let's see how it compares against other methods in our running example.

However, TS might get stuck on local optima depending on our prior of the outputscale/global noise.

# Pseudo-sequential methods

## The original batch expected improvement (qEI)

To motivate this section, let's remember how we define Expected Improvement. If {{< katex >}}f\sim\text{GP}(\mu, k){{< /katex >}}, then we can quantify how promising a new point {{< katex >}}\bm{x}{{< /katex >}} by how much we expect it to improve on the current best value {{< katex >}}f(x_{\text{best}}){{< /katex >}}:

{{< katex display >}}
\alpha_{\text{EI}}(\bm{x}; f, \mathcal{D}) = \mathbb{E}_{f(\bm{x})\sim\text{GP post.}}\left[I(\bm{x};f, \mathcal{D})\right],
{{< /katex >}}
where we can define the **improvement** {{< katex >}}I(\bm{x};f,\mathcal{D}){{< /katex >}} as {{< katex >}}\max(0, f(\bm{x}) - f(\bm{x}_{\text{best}})){{< /katex >}}, and {{< katex >}}\mathcal{D}{{< /katex >}} denotes the dataset.

Ideally, we would be able to measure the potential of a _batch_ of points similarly. We could define a batch version of expected improvement by considering a batch of input points {{< katex >}}\{\bm{x}^{(1)}, \dots \bm{x}^{(B)}\}{{< /katex >}} (where {{< katex >}}B{{< /katex >}} is the batch size), we define the **improvement of a batch** by
{{< katex display >}}
I_{b}(\bm{x}^{(1)}, \dots \bm{x}^{(B)}) = \max(I(x^{(1)}), \dots, I(x^{(B)})),
{{< /katex >}}
where we ommited the dependence on the posterior of the GP and the dataset. In other words, we are measuring whether a batch is good according to the performance of the best element in it. We're being **elitist**.[^is-this-a-good-idea?]

[^is-this-a-good-idea?]: Is being elitist in our definition a good idea? I really like the take of e.g. the quality-diversity community, and I wonder whether we could get inspired by how they define the quality of a collective/batch.

With this new random variable {{< katex >}}I_{b}{{< /katex >}} we can compute a batch expected improvement:
{{< katex display >}}
\alpha_{\text{qEI}}(\bm{x}^{(1)},\dots,\bm{x}^{(b)}; f, \mathcal{D}) = \mathbb{E}_{f(\bm{x}^{(1)}), \dots, f(\bm{x}^{(B)})\sim\text{GP post.}}\left[I_b(\bm{x}^{(1)}, \dots, \bm{x}^{(B)};f, \mathcal{D})\right],
{{< /katex >}}

Quick question: Is it easy to compute this quantity analytically? The answer is **very much no**. [Ginsbourger et al.]() devote 4 pages of their paper to the case where {{< katex >}}B = 2{{< /katex >}}. Numerically, we would need to search for all the elements in the batch simultaneously, converting the search space from {{< katex >}}\mathbb{R}^D{{< /katex >}} to {{< katex >}}\mathbb{R}^D\times\dots\times \mathbb{R}^D = \mathbb{R}^{BD}{{< /katex >}}. From a first glance this space might seem pretty big, but it is possible to optimize directly in it using gradient methods as we will see later (indeed, we fit neural networks in spaces much, much larger).[^BoTorch-seems-to-do-it]

[^BoTorch-seems-to-do-it]: According to [their documentation](), `botorch` actually optimizes in this large space for all their batch versions of acquisition functions by default; you can disable this behavior to go into what we discuss at the moment.

The original authors propose two heuristics for approximating this expected value in a pseudo-sequential manner. They're called the *Kriging Believer* (KB) and the *constant liar* (CL). In both, we construct the batch by sequentially computing EI, finding its maximum, and "simulating" the objective function by either assuming that the GP prediction is correct, or by assuming it's always a constant value chosen beforehand.

Here's a gif showcasing qEI using the constant liar heuristic, replacing the maximum of the acquisition function with the worst performing element of the dataset:

<video width="600" height="auto" controls>
    <source src="/static/assets/batch_bo_blogpost/q_ei.mp4", type="video/mp4">
</video>

Comparing with batch TS, it is evident that the original presentation of qEI **is much slower and requires many more model fits**. One needs to update the surrogate model for each element in the batch. A slightly faster way of doing it would be to optimize all the elements in the batch at the same time, like `botorch` does.[^BoTorch-seems-to-do-it] Still, remember that we are defining the quality of the batch to be the predicted performance of the best element in it.

## Batch UCB

<!-- - Parallelizing exploration-exploitation tradeoffs in gaussian process bandit optimization -->

Similarly, but for the Upper Confidence Bound, [Desautels et al.]() realized you don't need to evaluate the objective to compute the posterior variance[^the-formula]. So one could iteratively update _only_ the posterior variance and use UCB to construct a batch in a pseudo-sequential way. The authors call this GP-BUCB.

[^the-formula]: Remember that the posterior variance is given by ...

Put in pseudocode, GP-BUCB looks like this: Given the current dataset {{< katex >}}\mathcal{D}{{< /katex >}}, compute the GP posterior with predictive mean and standard deviation {{< katex >}}\mu_{\mathcal{D}}(\bm{x}), \sigma_{\mathcal{D}}(\bm{x}){{< /katex >}} and construct the batch as follows:

0. Initialize an empty batch {{< katex >}}\mathcal{B} = \varnothing{{< /katex >}}.
1. Optimize the upper confidence bound at {{< katex >}}\mathcal{D} \cup \mathcal{B}{{< /katex >}}: {{< katex >}}\alpha_{\text{UCB}}(\bm{x}) = \mu_{\mathcal{D}\cup \mathcal{B}}(\bm{x}) + \beta \sigma_{\mathcal{D}\cup\mathcal{B}}(\bm{x}){{< /katex >}}, arriving at {{< katex >}}\bm{x}_{\text{next}}{{< /katex >}}.
2. Append {{< katex >}}(\bm{x}_{\text{next}}, \mu_{\mathcal{D}}(\bm{x}_{\text{next}})){{< /katex >}} to the batch {{< katex >}}\mathcal{B}{{< /katex >}}. In other words, hallucinate that the output at the next point is what the GP predicts on \mathcal{D}. _Believe_ in the original kriging.
3. Go back to optimizing the UCB until {{< katex >}}\mathcal{B}{{< /katex >}} is full.

Here's a video showing the selection of this batch in our running example:

[gif]

## Penalizing locality (González et al.)

[González et al.]() propose another way of building a batch pseudo-sequentially. At each iteration, the original acquisition function {{< katex >}}\alpha{{< /katex >}} is optimized arriving at a next candidate {{< katex >}}\bm{x}_{b_1}{{< /katex >}}. It would be ideal to **penalize** the acquisition function **locally** around this point, to see where other local maxima of the acquisition function lie.

[Fig. from the paper]

A **local penalizer** around a point {{< katex >}}\bm{x}_{b_i}{{< /katex >}} is defined as a function {{< katex >}}\varphi_{\bm{x}_{b_i}}(\bm{x}){{< /katex >}} bounded between 0 and 1 that is non-decreasing on the distance to $\bm{x}_{b_i}$.

The particular penalizers González et al propose are defined as
{{< katex display >}}
\begin{array}{rl}
\varphi_{b_i}(\bm{x}) &= \text{Prob}[\bm{x}\text{ is not close to }\bm{x}_{b_i}] \\
&= \text{Prob}[\bm{x}\notin B_{r_i}(\bm{x}_{b_i})] \\
&= 1 - \text{Prob}[\bm{x}\in B_{r_i}(\bm{x}_{b_i})]
\end{array}
{{< /katex >}}
where {{< katex >}}B_{r_i}(\bm{x}_{b_i}){{< /katex >}} is the ball of radius {{< katex >}}r_i = (f_\text{best so far} - f(\bm{x}_{b_i})) / L{{< /katex >}}, and {{< katex >}}L{{< /katex >}} is a Lipschitz constant. {{< katex >}}\varphi_{b_i}(\bm{x}){{< /katex >}} can be computed in closed form, but we'll skip the technical details for now. Check the footnotes if you're curious.[^the-details-on-penalizer]

[^the-details-on-penalizer]: ...

As a brief reminder of what we mean by a Lipschitz constant: we say that a function is **Lipschitz** (with constant {{< katex >}}L{{< /katex >}}) if... If the function is differentiable, then the smallest Lipschitz constant is given by {{< katex >}}\nabla f{{< /katex >}}. In this paper, we approximate this gradient using what the GP predicts.[^gradient-of-gps]

[^gradient-of-gps]: GPs are great because they also allow us to make statements about the derivative of the function we're approximating [TODO:ADD]. 

It's a little bit non-trivial to implement this in GPJax at the moment, since they haven't implemented gradients of GPs yet. We leave the comparison of this method in particular as an exercise to the reader.

One small comment: this is not the only penalizer one might construct around the current batch point. It's a nice one, in the sense that it is theoretically justified by the Lipschitz assumption. However, one could easily think of a generalization using functions with compact support.

# Methods that vary the batch size dynamically

- hybrid, budgeted

# Contemporary batching of acquisition functions

This original way of batching expected improvement is no longer the default way in which contemporary frameworks do it.

- the reparametrization trick for acquisition functions by Wilson et al.

# Some simple comparisons

## The silliest baseline: sampling entirely at random.

## A more powerful baseline: CMA-ES

