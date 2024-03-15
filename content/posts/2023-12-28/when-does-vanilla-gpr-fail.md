---
date: "2023-12-28"
title: "Do Gaussian Processes scale well with dimension?"
slug: how-gps-scale-with-dimension
image: /static/assets/hdgp_blogpost/banner.jpg
---

> This blogpost assumes you're already familiarized with the basics of Gaussian Processes. Check [my previous blogpost](../2023-07-31/intro-to-bo.md) for an introduction.

It's folk knowledge that Gaussian Processes (GPs) don't scale well with the dimensionality of their inputs.
Some people even claim that, if the problem goes beyond 20 input dimensions, then GPs fail to do regression well.[^folk-knowledge]

[^folk-knowledge]: (folk knowledge) As I discussed in the previous blogpost, [this Stack Exchange question](https://stats.stackexchange.com/questions/564528/why-does-bayesian-optimization-perform-poorly-in-more-than-20-dimensions) dives into the question. There seems to be some divide in the answers, though, with some people disputing the usual arguments of volume in high dimensions in the comments.

The main hypothesis is that GPs fail because of the curse of dimensionality: since the usual (stationary) kernels base their computation of correlation on distance, there's less signal in higher dimensions because *distances become meaningless*. Other reasons why GPs might fail could be having difficult loss landscapes or numerical instabilities.[^botorch-and-float64]

[^botorch-and-float64]: For example, `botorch` always suggests working on double precision when running Bayesian Optimization with their models and kernels.

These hypotheses might be misguided. Last month, [Carl Hvafner, Erik Orm Hellsten and Luigi Nardi released a paper called *Vanilla Bayesian Optimization Performs Great in High Dimensions*](https://arxiv.org/abs/2402.02229), in which they explain the main reasons why GPs fail in high dimensions, disputing and disproving this folk knowledge.

In this blogpost I explore the failures of GPs to fit in high dimensions, following what Hvafner et al. propose in their recent paper.[^I-got-scooped]

I only assume that you're familiar with [my previous blogpost on GPs and Bayesian Optimization](../2023-07-31/intro-to-bo.md).

[^I-got-scooped]: Actually, I started writing this blogpost in Dec. of last year, wanting to explore the impact several design choices had on GP regression, but I got scooped :(.

# Vanilla GPs fail to fit really simple functions

## A really simple function

Consider the **shifted sphere** function {{< katex >}}f_{\bm{r}}:\mathbb{R}^D\to\mathbb{R}{{< /katex >}} given by
{{< katex display>}}
f_{\bm{r}}(\bm{x}) = \sum_{i=1}^D (x_i - r_i)^2,
{{< /katex >}}
where {{< katex >}}\bm{r}{{< /katex >}} is a random offset. This function is extremely simple. In 2 dimensions it's a parabola, in 3 it's a paraboloid, and so on. It's a second-degree polynomial on its inputs {{< katex >}}\bm{x} = (x_1, \dots, x_D){{< /katex >}}, and it is as smooth as functions come.

## Fitting a GP to it

Consider the most vanilla GP model:

```python
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        _, n_dimensions = train_x.shape

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dimensions)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

```

We should it to fit this function quite easily, even in high dimensions. Let me show you when it fails.

For our training set, we sample {{< katex >}}N{{< /katex >}} points from a unit Gaussian in {{< katex >}}D{{< /katex >}} dimensions and add a little bit of random noise:
{{< katex display>}}
\begin{array}{ll}
\bm{x}_n \sim \mathcal{N}(\bm{0}, \bm{I}_D)&\text{(data dist.)}\\[0.2cm]
y_n = f_{\bm{r}}(\bm{x}_n) + \epsilon,\epsilon \sim \mathcal{N}(0, 0.25)&\text{(noisy output)}.
\end{array}
{{< /katex >}}

And indeed, in the 1D case we get a pretty good fit:

{{< figure src="/static/assets/hdgp_blogpost/shifted_sphere_1d.jpg" alt="A shifted sphere, approximated using a Gaussian Process" class="largeSize" title="Fitting a vanilla GP to a shifted sphere" >}}

We can also quantify the quality of the fit by plotting the mean predicted values against the actual values for a small test set, sampled from the same distribution and corrputed in the same way. This plot is useful, because it can be computed regardless of the input dimension.

{{< figure src="/static/assets/hdgp_blogpost/comparison_shifted_sphere_1d.jpg" alt="Actual vs. predicted values in a model fitted on a 1 dimensional shifted sphere" class="largeSize" title="A good fit - predictions and actual values are highly correlated" >}}

**What happens if we go to higher dimensions?**, Let's try to fit this exact same function, but with {{< katex >}}D=64{{< /katex >}}. Since we can't visualize {{< katex >}}\mathbb{R}^{64+1}{{< /katex >}} space, we can only rely on these second plots I showed you, the ones that compare predictions to actual values... Immediately, we can see that vanilla GP fails and defaults to predicting just the mean:

{{< figure src="/static/assets/hdgp_blogpost/comparison_on_64d.jpg" alt="Actual vs. predicted values in a model fitted on a 1 dimensional shifted sphere" class="largeSize" title="A bad fit - predictions default to the mean on a 64D shifted sphere, even using 2000 training points" >}}

That is, the model didn't learn a thing. It's defaulting to a certain mean prediction. Let me try to find exactly **when** GPs start to fail. Folk knowldege says it's around 20 dimensions, but if we sweep for several values of $N$ and $D$, we get the following table:

{{< figure src="/static/assets/hdgp_blogpost/ExactGPModel_nice_table.jpg" alt="A table showing whether the model learned anything, sweeping across number of points and dimensions" class="largeSize" title="Table - nr. of training points & input dimension, and whether the model learned anything" >}}

where `yes :)` means that the model learned _something_, and didn't default to the mean. The breaking point for this function in particular seems to be around 32 dimensions, and we notice that the number of training points is of course important.

## Why?!

Why are GPs failing to fit even a simple function like a high-dimensional paraboloid? As I said in the introduction,the community argues that this is a failure of stationary kernels' use of distance for correlation. In the next section I talk a little bit about kernels and about the curse of dimensionality.

# The curse of dimensionality and kernel methods

## Measuring correlation using kernels

Recall that GPs assume the following: given a function {{< katex >}}f\colon\mathbb{R}^D\to\mathbb{R}{{< /katex >}} and two inputs {{< katex >}}\bm{x}, \bm{x}'\in\mathbb{R}^D{{< /katex >}}, then
{{< katex display>}}
\text{cov}(f(\bm{x}), f(\bm{x}')) = k(\bm{x}, \bm{x}')
{{< /katex >}}
where {{< katex >}}k\colon\mathbb{R}^D\times\mathbb{R}^D\to\mathbb{R}{{< /katex >}} is a kernel function (i.e. symmetric positive-definite).

In other words, we make statements about how correlated two function evaluations are (the left side of the equation) using kernels as a proxy (the right side).

A large family of these kernels only depend on the distance between inputs {{< katex >}}(x_i - x_i')^2{{< /katex >}}, such as the Radial Basis Function (RBF):

{{< katex display>}}
k_{\text{RBF}}(\bm{x}, \bm{x}';\,\sigma, \Theta) = \sigma\exp\left(-\frac{1}{2}(\bm{x}-\bm{x}')^\top\Theta^{-2} (\bm{x}-\bm{x}')\right),
{{< /katex >}}
where {{< katex >}}\sigma>0{{< /katex >}} is an output scale, and {{< katex >}}\Theta{{< /katex >}} is a diagonal matrix with lengthscales.

These distance-based kernels are called *stationary*. An example of a kernel that is **not stationary** is the polynomial kernel:
{{< katex display>}}
k_{\text{p}}(\bm{x}, \bm{x}';\,\sigma, c, d) = \sigma(\bm{x}^{\top}\bm{x}' + c)^d,
{{< /katex >}}
where {{< katex >}}c{{< /katex >}} is an offset, and {{< katex >}}d{{< /katex >}} is the degree of the polinomial. The degree of the polynomial is usually specified by the user, and the offset is optimized through the marginal likelihood of the dataset.

<!-- [Remembering the curse of dimensionality] -->
## The curse of dimensionality

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


## Lengthscales, lengthscales, lengthscales

Our stationary kernels should reflect this change in distance. By including lengthscales, our computation of correlation is actually mediated by hyperparameters that we tune during training. Lengthscales govern the "zone of influence" of a given training point: large values allow GPs to have higher correlation _further away_, and lower correlations mean that the zone of influence of a given training point is small, distance-wise. Fig. 2 of [Hvafner et al. 2023](https://arxiv.org/abs/2402.02229) exemplifies this beautifully.

{{< figure src="/static/assets/hdgp_blogpost/lengthscale_impact.png" alt="Impact of the lengthscale on GP regression, taken from Hvafner" class="largeSize" title="The impact of lengthscales on Gaussian Process regression. (Image source: Fig. 2 of Hvafner et al. 2023)" >}}

How are the lengthscales looking in our simple example? [TODO: do this analysis]

# A simple fix: Imposing larger lengthscales

[Hvafner et al.]()'s insight is this: **we should be encouraging larger lengthscales**. They phrase it in the language of model complexity, saying that the functions we might be fitting are not as complex as one may think. There's a mismatch, according to them, between the assumed complexity and the actual complexity of the functions we're fitting.

Larger lengthscales would allow the kernel to assume correlation between points that are further away, mitigating the curse of dimensionality.

How do we encourage larger lengthscales during training? The answer is easy: regularize the loss.

## MAP vs. MLE estimates

In the previous blogpost, I vaguely stated that kernel hyperparameters can be trained by maximizing the log-marginal likelihood w.r.t. the training points. Being a little bit more explicit, the function we are trying to maximize is this:

[log marginal likelihood of a GP]

Maximizing this quantity results in what is called the *Maximum Likelihood Estimate*, or MLE.

To encourage higher lengthscales, we can add a prior distribution for the lengthscales, and try to maximize the *a posteriori* distribution, which is proportional to the product of the likelihood and the prior. The new function we would try to maximize would then be:

[the log marginal likelihood of a GP plus the prior]

Maximizing this renders the *Maximum a posteriori* (MAP) estimate.

This was all just a fancy way of saying "add a term to your loss that encourages high lengthscales". Phrasing this in a probabilistic language is useful, because we can be more precise about which values we'd like our lengthscales to take.

## Some priors for likelihoods

By default, `gpytorch` considers no prior on the lengthscales. `botorch`'s default `SingleTaskGP` has a {{< katex >}}\lambda_i \sim \text{Gamma}(3, 6){{< /katex>}} prior, which has an average value of {{< katex >}}\mathbb{E}[\lambda_i] = 18{{< /katex >}} and a standard deviation of around {{< katex >}}\sigma_{\lambda_i} = 10.4{{< /katex >}}. Here's the density of this prior:

[The density of a Gamma(3, 6)]

Notice how this is entiery independent of the dimensionality of the input space. As our plot above shows, for dimensionalities above 64, the expected distances between randomly sampled points is already above 20.

[Hvafner et al.]() propose a simple prior that does depend on the dimension of the input space:
{{< katex display >}}
p(\lambda_i) = \text{logNormal}(\mu_0 + \log(D)/2, \sigma_0).
{{< /katex >}}
Let's visualize this prior for different dimensions

[The plot]

# Applying this fix

Applying this fix is as simple as adding a single line to our torch models:

```python
# Check the previous blogpost for an implementation of a vanilla GP.
class ExactGPModelWithLogNormalPrior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelWithLogNormalPrior, self).__init__(
            train_x, train_y, likelihood
        )
        _, n_dimensions = train_x.shape

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_dimensions,
                # THE ONLY CHANGE IS THE FOLLOWING LINE:
                lengthscale_prior=gpytorch.priors.LogNormalPrior(
                    np.log(n_dimensions) / 2, 1.0
                ),
            )
        )
```

Adding this prior has a dramatic impact on whether the model is able to fit above 32 dimensions in the toy example discussed previously. Here're the model's predictions in, say, 512 dimensions:

[The plot]

We can compute the same table we showed for `ExactGPModel`, but for this one:

[The table]

Now we can fit up to 512 dimensions easily! With enough data, we might be able to fit 1024 dimensions as well.


