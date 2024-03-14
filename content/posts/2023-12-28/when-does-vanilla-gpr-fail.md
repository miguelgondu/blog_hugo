---
date: "2023-12-28"
title: "Do Gaussian Processes scale well with dimension?"
slug: how-gps-scale-with-dimension
image: /static/assets/bo_blogpost/bo_w_TS.gif
---

> This blogpost assumes you're already familiarized with the basics of Gaussian Processes and Bayesian Optimization. Check [my previous blogpost](../2023-07-31/intro-to-bo.md) for an introduction.

It's folk knowledge that Gaussian Processes (GPs) don't scale well with the dimensionality of their inputs.
Some people even claim that, if the problem goes beyond 20 input dimensions, then GPs fail to do regression well.[^folk-knowledge]

[^folk-knowledge]: (folk knowledge) As I discussed in the previous blogpost, [this Stack Exchange question](https://stats.stackexchange.com/questions/564528/why-does-bayesian-optimization-perform-poorly-in-more-than-20-dimensions) dives into the question. There seems to be some divide in the answers, though, with some people disputing the usual arguments of volume in high dimensions in the comments.

The main hypothesis is that GPs fail because of the curse of dimensionality: since the usual (stationary) kernels base their computation of correlation on distance, there's less signal in higher dimensions because *distances become meaningless*. Other reasons why GPs might fail could be having difficult loss landscapes or numerical instabilities.[^botorch-and-float64]

[^botorch-and-float64]: For example, `botorch` always suggests working on double precision when running Bayesian Optimization with their models and kernels.

Last month, [Carl Hvafner, Erik Orm Hellsten and Luigi Nardi released a paper called *Vanilla Bayesian Optimization Performs Great in High Dimensions*](https://arxiv.org/abs/2402.02229), in which they explain the main reasons why GPs fail in high dimensions, disputing and disproving this folk knowledge.

In this blogpost I discuss said paper, and I re-implement one of its experiments in a toy setting.[^I-got-scooped] I only assume that you're familiar with [my previous blogpost on GPs and Bayesian Optimization](../2023-07-31/intro-to-bo.md).

[^I-got-scooped]: Actually, I started writing this blogpost in Dec. of last year, wanting to explore the impact several design choices had on GP regression, but I got scooped :(.

## The curse of dimensionality and kernel methods

**Measuring correlation using kernels**

Recall that GPs assume the following: given a function {{< katex >}}f\colon\mathbb{R}^n\to\mathbb{R}{{< /katex >}} and two inputs {{< katex >}}\bm{x}, \bm{x}'{{< /katex >}}, then
{{< katex display>}}
\text{cov}(f(\bm{x}), f(\bm{x}')) = k(\bm{x}, \bm{x}')
{{< /katex >}}
where {{< katex >}}k\colon\mathbb{R}^D\times\mathbb{R}^D\to\mathbb{R}{{< /katex >}} is a kernel function (i.e. symmetric positive-definite). In other words, we make statements about how correlated two function evaluations are (the left side of the equation) using kernels as a proxy (the right side).

A large family of these kernels only depend on the distance between inputs {{< katex >}}\|\bm{x} - \bm{x}'\|{{< /katex >}}, such as the Radial Basis Function (RBF):

{{< katex display>}}
k_{\text{RBF}}(\bm{x}, \bm{x}';\,\sigma, \Lambda) = \sigma\exp\left(-\frac{1}{2}(\bm{x}-\bm{x}')^\top\Lambda^{-2} (\bm{x}-\bm{x}')\right),
{{< /katex >}}
where {{< katex >}}\sigma>0{{< /katex >}} is an output scale, and {{< katex >}}\Lambda{{< /katex >}} is a diagonal matrix with lengthscales.

These kernels are called *stationary*. An example of a kernel that is **not stationary** is the polynomial kernel:
{{< katex display>}}
k_{\text{p}}(\bm{x}, \bm{x}';\,\sigma, c, d) = \sigma(\bm{x}^{\top}\bm{x}' + c)^d,
{{< /katex >}}
where {{< katex >}}c{{< /katex >}} is an offset, and {{< katex >}}d{{< /katex >}} is the degree of the polinomial. The degree of the polynomial is usually specified by the user, and the offset is optimized through the marginal likelihood of the dataset.

<!-- [Remembering the curse of dimensionality] -->
**The curse of dimensionality**

Put shortly, the curse of dimensionality says that distances start to become meaningless in higher dimensions, and that space becomes more and more sparse.

We can actually quantify this with a simple experiment: consider a collection of points {{< katex >}}\{x_n\}_{n=1}^N{{< /katex >}} in {{< katex >}}[0,1]^D\subseteq\mathbb{R}^D{{< /katex >}}, sampled uniformly at random in the unit cube. What can we say about the distribution of the pair-wise distances?

Consider the following Python script for randomly sampling a collection of points and computing their pairwise distance (using `jax` for speed):

```python
import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def pairwise_distances(x):
    # Using the identity \|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x^T y
    distances = (
        jnp.sum(x**2, axis=1)[:, None] - 2 * x @ x.T + jnp.sum(x**2, axis=1)[None, :]
    )

    return distances


def compute_distribution_of_distances(
    n_points: int,
    n_dimensions: int,
):
    seed = np.random.randint(0, 10000)
    key = jax.random.PRNGKey(seed)

    # Warm-up for the JIT
    x = jax.random.uniform(key, (2, n_dimensions))
    distances = pairwise_distances(x)

    # Sampling from the unit cube and computing
    # the pairwise distances
    x = jax.random.normal(key, (n_points, n_dimensions))
    distances = pairwise_distances(x)

    # Keeping only the upper triangular part
    # of the distance matrix
    distances = jnp.triu(distances, k=1)

    return distances[distances > 0.0]


if __name__ == "__main__":
    N_POINTS = 1_000
    n_dimensions = [2**exp_ for exp_ in range(1, 8)]

    arrays = {}
    for n_dimensions_ in n_dimensions:
        distances = compute_distribution_of_distances(N_POINTS, n_dimensions_)
        arrays[n_dimensions_] = distances
```

Visualizing these distances renders this plot, where we color by different dimensions (from `2 ** 0` to `2 ** 7`):

{{< figure src="/static/assets/hdgp_blogpost/pairwise_distances_unit_cube.jpg" alt="Distribution of pairwise distances in several dimensions" class="largeSize" >}}

Notice how **the average distance between randomly sampled points starts to grow**, even if we restrict ourselves to the unit square.

Our stationary kernels should reflect this. By including lengthscales, our computation of correlation is actually mediated by hyperparameters that we tune during training. Lengthscales govern the "zone of influence" of a given training point: large values allow GPs to have higher correlation _further away_, and lower correlations mean that the zone of influence of a given training point is small, distance-wise.

## The toy-est of toy problems

A way to check _how far up we can go_ with vanilla GP Regression is to consider the most simple function to predict. We use a slightly shifted version of a sphere:
{{< katex display>}}
f_{\bm{r}}(\bm{x}) = \sum_{i=1}^D (x_i - r_i)^2
{{< /katex >}}
where we select the radius at random from a standard Gaussian.

Using {{< katex >}}f_{\mathbb{r}}{{< /katex >}}, we construct a noisy dataset {{< katex >}}\{(\bm{x}_n, y_n)\}_{n=1}^N{{< /katex >}} where each {{< katex >}}y{{< /katex >}} was perturbed by noisy samples from a tiny Gaussian
{{< katex display>}}
\begin{array}{ll}
\bm{x}_n \sim \mathcal{N}(\bm{0}, \bm{I}_D)&\text{(data dist.)}\\[0.2cm]
y_n = f_{\bm{r}}(\bm{x}_n) + \epsilon,\epsilon \sim \mathcal{N}(0, 0.25)&\text{(noisy output)}.
\end{array}
{{< /katex >}}

To check the impact of stationarity, we test with an RBF and a polinomial kernel separately. In both cases we go for a zero mean and we maximize the marginal likelihood w.r.t the data.[^training-details]

[^training-details]: A little bit more about the training itself: using `gpytorch`, we constructed a basic `ExactGP` and fitted the kernel hyperparameters using adam with a learning rate of {{< katex >}}0.05{{< /katex >}} and {{< katex >}}1000{{< /katex >}} iterations on the whole dataset.

## Measuring model performance beyond 2D

For the cases {{< katex >}}D=1,2{{< /katex >}}, the output of both the underlying function and the GP model can be visualized:

[...Images of both cases]

What can we do for {{< katex >}}D > 2{{< /katex >}}? It's common practice[^for-example-SAASBO] to consider the following: starting with a test dataset of 50 points sampled from the same distribution {{< katex >}}\bm{x}_m^*\sim\mathcal{N}(\bm{0}, \bm{I}_D){{< /katex >}}, we consider the actual function value {{< katex >}}f_\mathbb{r}(\bm{x}_m){{< /katex >}} and the model prediction {{< katex >}}\tilde{f}(\bm{x}_m){{< /katex >}}, and plot them against each other.

[...one example]

In an ideal world, the points in this scatter plot should lie exactly on the diagonal. Here's an example of a poor model's performance:

[...another example]

Numerically, we could consider metrics like mean absolute error, root mean squared error, or the correlation between predictions and actual values. We will focus on correlation.

[^for-example-SAASBO]: For example, check Fig.[...TODO:ADD] of [the `SAASBO` paper](TODO:ADD). 

## Several dimensions and dataset sizes

To check how far up we can go, let's sweep between the following dataset sizes {{< katex >}}N{{< /katex >}} and number of input dimensions {{< katex >}}D{{< /katex >}}:
{{< katex display>}}
N = \{50, 100, 500, 1000, 1500, 2000, 2500, 5000, 7500, 10000\},\\[0.2cm] D=\{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024\}
{{< /katex >}}


[Using the toy high-dimensional problems described above, we check whether the GP is able to learn _something_ in higher dimensions. We compare that against e.g. simple linear regression]

[The moneyshot: a plot of predicted vs. actual values for a test set, like the one in SAASBO]

## Experiment: high-dimensional Bayesian Optimization

[HDBO is even harder: not only do we need to worry about the GP actually learning stuff, but we also need to make sure that we can optimize the acquisition function]

[The experimental set-up: using the toy high-dimensional problems, we compare the maximum achieved in different dimensions. We also compare against simple evolutionary strategies like CMA-ES]


