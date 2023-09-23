---
date: "2023-07-31"
title: "Bayesian Optimization using Gaussian Processes: an introduction"
slug: intro-to-bo
image: /static/assets/bo_blogpost/bo_w_TS.gif
---

> This blogpost is an adaptation of Chap. 3 in [my dissertation](https://www.miguelgondu.com/assets/phdthesis.pdf). Check it out if you're interested!

Bayesian Optimization (BO) is a tool for black-box optimization. By black-box, we mean functions to which we only have access by querying: expensive simulations of the interaction between a molecule and a folded protein, users interacting with our websites, or the accuracy of a Machine Learning algorithm with a given configuration of hyperparameters.

The gist of BO is to approximate the function we are trying to optimize with a regression model that is uncertainty aware, and to use these uncertainty estimates to determine what our next test point should be. It is usual practice to do BO using Gaussian Processes (GPs), and this blogpost starts with an introduction to GP regression.

This blogpost introduces (Gaussian Process based) Bayesian Optimization, and provides code snippets for the experiments performed. The tools used include GPyTorch and BoTorch as the main engines for Gaussian Processes and BO, and EvoTorch for the evolutionary strategies.

## Running examples: x sin(x), Easom, and Cross-in-tray

In this blogpost we will use three running examples to explain GPs and BO. The first one is the function {{< katex >}}f(x) = x\sin(x){{< /katex >}}, which is one-dimensional in its inputs and allows us to visualize how GPs handle uncertainty. The two remaining ones are part of a plethora of test functions that are usually included in black-box optimization benchmarks:[^optimization-benchmarks] `Easom` and `Cross-in-tray`. Their formulas are given by

{{< katex display >}}
\text{\texttt{Easom}}(\bm{x}) = \cos(x_1)\cos(x_2)\exp\left(-(x_1-\pi)^2 - (x_2 - \pi)^2\right),
{{< /katex >}}

{{< katex display >}}
\text{\texttt{Cross-in-tray}}(\bm{x}) = \left|\sin(x_1)\sin(x_2)\exp\left(\left|10 - \frac{\sqrt{x_1^2 + x_2^2}}{\pi}\right|\right)\right|^{0.1}.
{{< /katex >}}

We take the 2D versions of these functions for visualization, but notice how these can easily be extended to higher dimensions. The `Easom` test function has its optimum at 
{{< katex >}}(\pi, \pi){{< /katex >}}, and the `Cross-in-tray` has 4 equally good optima at {{< katex >}}(\pm 1.35, \pm 1.35){{< /katex >}}, approximately.

{{< figure src="/static/assets/bo_blogpost/all_three_test_functions.jpg" alt="Three test functions: x * sin(x), Easom and Cross-in-tray" >}}

[^optimization-benchmarks]: (Optimization benchmarks) There are plenty more test functions for optimization in [this link](https://www.sfu.ca/~ssurjano/optimization.html).

## An introduction to Gaussian Processes

Gaussian Processes are a **probabilistic regression method**. This means that they approximate a function {{< katex >}}f\colon\mathcal{X}\to\mathbb{R}{{< /katex >}} while quantifying values like **expectations** and **variances**. More formally,

**Definition:** A function {{< katex >}}f{{< /katex >}} is a **Gaussian Process** (denoted {{< katex >}}f\sim\text{GP}(\mu_0, k){{< /katex >}}) if any finite collection of evaluations {{< katex >}}\{f(\bm{x}_1), \dots, f(\bm{x}_N)\}{{< /katex >}} is normally distributed like

{{< katex display >}}
\begin{bmatrix}
f(\bm{x}_1) \\
\vdots \\
f(\bm{x}_N)
\end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix}
\mu_0(\bm{x}_1) \\
\vdots \\
\mu_0(\bm{x}_N)
\end{bmatrix}, \begin{bmatrix}
k(\bm{x}_1, \bm{x}_1) & \cdots & k(\bm{x}_1, \bm{x}_N) \\
\vdots & \ddots & \vdots \\
k(\bm{x}_N, \bm{x}_1) & \cdots & k(\bm{x}_N, \bm{x}_N) 
\end{bmatrix}\right){{< /katex >}}

The matrix {{< katex >}}\bm{K} = [k(\bm{x}_i, \bm{x}_j)]{{< /katex >}} is known as the **Gram matrix**, and since we're using it as a covariance matrix, this imposes some restrictions on our choice of **kernel** (or covariance function) {{< katex >}}k{{< /katex >}}: it must be symmetric positive-definite, in the sense that the Gram matrices it spans should be symmetric positive definite. The function {{< katex >}}\mu_0{{< /katex >}} is called a **prior**, and it allows us to inject expert knowledge in our modeling (if e.g. we know that our function should be something like a line or a parabola, we can set such {{< katex >}}\mu_0{{< /katex >}}). It's pretty common to set {{< katex >}}\mu_0 \equiv 0{{< /katex >}} (or a constant), and let the data speak for itself.

Assuming {{< katex >}}f\sim \text{GP}(\mu, k){{< /katex >}} allows for computing **an entire distribution** over previously unseen points. As an example, let's say you have a dataset {{< katex >}}\mathcal{D} = \{(x_1, f(x_1)), \dots, (x_N, f(x_N))\}{{< /katex >}} for our test function {{< katex >}}f(x)=x\sin(x){{< /katex >}}, measured with a little bit of noise: 

{{< figure src="/static/assets/bo_blogpost/xsinx_w_noisy_samples.jpg" alt="Noisy samples from x * sin(x)" class="midSize" >}}

Given a new point {{< katex >}}x_*{{< /katex >}} that's not on the dataset, you can consider the joint distribution of {{< katex >}}\{f(x_1), \dots, f(x_N), f(x_*)\}{{< /katex >}} and compute the conditional distribution of {{< katex >}}f(x_*){{< /katex >}} given the dataset:


{{< katex display >}}
f(\bm{x}_*) \sim \mathcal{N}(\mu_0(\bm{x}_*) + \bm{k}_*^\top \bm{K}^{-1}(\bm{f} - \bm{\mu_0}), k(\bm{x}_*, \bm{x}_*) - \bm{k}_*^\top \bm{K}^{-1} \bm{k}_*)
{{< /katex >}}

where {{< katex >}}\bm{k}^* = [k(\bm{x}_*, \bm{x}_i)]_{i=1}^N{{< /katex >}}, {{< katex >}}\bm{f} = [f(\bm{x}_i)]_{i=1}^N{{< /katex >}}, {{< katex >}}\bm{\mu_0} = [\mu_0(\bm{x}_i)]_{i=1}^N{{< /katex >}}, and {{< katex >}}\bm{K}=[k(\bm{x}_i, \bm{x}_j)]_{i,j=1}^N{{< /katex >}}.[^the-ugly-details]

[^the-ugly-details]: (the ugly details) I will skip the ugly details in this blogpost. The computation of a Gaussian Process' predictive posterior can be found in Sec. 2.2. of [Gaussian Processes for Machine Learning, by Rasmussen and Williams](https://gaussianprocess.org/gpml/chapters/RW2.pdf), or on Sec. 4.2.2. of [Probabilistic Numerics, by Hennig et al.](https://www.probabilistic-numerics.org/assets/ProbabilisticNumerics.pdf). Notice that I'm skipping the discussion on diagonal noise modeling.


Consider the distribution around {{< katex >}}x_*=0{{< /katex >}} in our running example, assuming that the prior {{< katex >}}\mu_0{{< /katex >}} equals {{< katex >}}0{{< /katex >}}. The closest values we have in the dataset are at {{< katex >}}x=-0.7{{< /katex >}} and {{< katex >}}x=1.1{{< /katex >}}, which means {{< katex >}}x_*=0{{< /katex >}} is far away from the training set. This is automatically quantified by the variance in the posterior distribution of {{< katex >}}f(x_*){{< /katex >}}:

{{< figure src="/static/assets/bo_blogpost/xsinx_dist_at_0.jpg" alt="A Gaussian distribution with mean and variance predicted by a Gaussian Process, fitted on noisy samples of x * sin(x)" class="midSize" >}}

Let's visualize this distribution for all points in the interval [-10, 10], sweeping through them and highlighting the actual value of the function {{< katex >}}f(x) = x\sin(x){{< /katex >}}.

{{< figure src="/static/assets/bo_blogpost/xsinx_gp_all_distributions.gif" alt="An animation showing several Gaussian distributions predicted by a GP, sweeping through the x axis, of x * sin(x)">}}

Notice how the distribution spikes when close to the training points, and flattens outside the support of the data. This is exactly what we mean by uncertainty quantification.

This process of predicting Gaussian distributions generalizes to collections of previously unseen points {{< katex >}}\bm{X}_*{{< /katex >}}, allowing us consider the mean prediction and the uncertainties around it. The math is essentially the same, and the details can be found in the references.

If we consider a fine enough grid of points and consider their posterior distribution, we can essentially "sample a function" out of our Gaussian Process using the same formulas described above. For example, these are 5 different samples of potential approximations of {{< katex >}}f(x) = x\sin(x){{< /katex >}} according to the posterior of the GP:

{{< figure src="/static/assets/bo_blogpost/xsinx_samples.jpg" alt="Five samples from a GP fitted on the noisy evaluations of x * sin(x)" class="midSize" >}}

Common choices of kernels are the RBF and the Matérn family:

- {{< katex >}}k_{\text{RBF}}(\bm{x},\bm{x}';\; \theta_{\text{out}}, \bm{\theta}_\text{l}) = \theta_{\text{out}}\exp\left(-\frac{1}{2}r(\bm{x}, \bm{x}'; \bm{\theta}_{\text{l}})\right){{< /katex >}}

- {{< katex >}}k_\nu(\bm{x}, \bm{x}';\; \theta_{\text{out}}, \bm{\theta}_\text{l}) = \theta_{\text{out}}\frac{2^{1 - \nu}}{\Gamma(\nu)}(\sqrt{2\nu}r)^\nu K_\nu(\sqrt{2\nu}r(\bm{x}, \bm{x}'; \bm{\theta}_{\text{l}})){{< /katex >}}

Where
- {{< katex >}}r(\bm{x}, \bm{x}'; \bm{\theta}_{\text{l}}) = (\bm{x} - \bm{x}')^\top(\bm{\theta}_{\text{l}})(\bm{x} - \bm{x}'){{< /katex >}} is the distance between {{< katex >}}\bm{x}{{< /katex >}} and {{< katex >}}\bm{x}'{{< /katex >}} mediated by a diagonal matrix of lengthscales {{< katex >}}\bm{\theta}_{\text{l}}{{< /katex >}}, {{< katex >}}\theta_{\text{out}} > 0{{< /katex >}} is a positive hyperparameter called the output scale,
- {{< katex >}}\Gamma{{< /katex >}} is the [Gamma function](https://mathworld.wolfram.com/GammaFunction.html), {{< katex >}}K_\nu{{< /katex >}} is the [modified Bessel function](https://en.wikipedia.org/wiki/Bessel_function) of the second kind, and
- {{< katex >}}\nu{{< /katex >}} is usually 5/2, but it can be any positive real number. For {{< katex >}}\nu = i + 1/2{{< /katex >}} (with {{< katex >}}i{{< /katex >}} a positive integer) the kernel takes a nice closed form.[^details-on-matern]

[^details-on-matern]: (details on Matérn) For more details, see [Probabilistic Numerics, Sec. 5.5.](https://www.probabilistic-numerics.org/assets/ProbabilisticNumerics.pdf).

These kernels have some hyperparameters in them (like the lengthscale matrix {{< katex >}}\bm{\theta}_l{{< /katex >}} or the output scale {{< katex >}}\theta_{\text{out}}{{< /katex >}}). **Fitting** a Gaussian Process to a given dataset consists of estimating these hyperparameters by maximizing the likelihood of the data.

To summarize,
- By using GPs, we assume that finite collections of evaluations of a function are distributed normally around a certain prior {{< katex >}}\mu_0{{< /katex >}}, with covariances dictated by a kernel {{< katex >}}k{{< /katex >}}.
- This assumption allows us to compute distributions over previously unseen points {{< katex >}}\bm{x}_*{{< /katex >}}.
- Kernels define the space of functions we can use to approximate, and come with certain hyperparameters that we need to fit by maximizing the likelihood.

## Gaussian Processes in practice

### scikit-learn

There are several open source implementations of Gaussian Process Regression, as described above. First, there's `scikit-learn`'s interface, which quickly allows for specifying a kernel. This is the code you would need to create the noisy samples described above, and to fit a Gaussian Process to those samples:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

import numpy as np

def fit_gaussian_process(
    inputs: np.ndarray, outputs: np.ndarray
) -> GaussianProcessRegressor:
    """
    Fits a Gaussian Process with an RBF kernel
    to the given inputs and outputs.
    """

    # --------- Defining the kernel ---------
    # the "1 *" in front is understood as the
    # output scale, and will be optimized during
    # training. Besides the RBF, we also add a
    # WhiteKernel, which corresponds to adding a
    # constant to the diagonal of the covariance.
    kernel = 1 * RBF() + WhiteKernel()

    # ------ Defining the Gaussian Process -----
    # Besides the kernel, we could also specify the
    # internal optimizer, the number of iterations,
    # etc.
    model = GaussianProcessRegressor(kernel=kernel)
    model.fit(inputs, outputs)

    return model
```

Internally, `scikit-learn` uses `scipy`'s optimizers to maximize the likelihood w.r.t. the kernel parameters. You can check the kernel's optimized parameters by printing `model.kernel_`. These are the hyperparameters for our running example on {{< katex >}}f(x) = x\sin(x){{< /katex >}}:

```
4.72**2 * RBF(length_scale=1.58) + WhiteKernel(noise_level=0.229)
```

And this is the predicted mean and uncertainty:

{{< figure src="/static/assets/bo_blogpost/gp_in_sklearn.jpg" alt="A GP fitted on the noisy evaluations of x * sin(x) using an RBF kernel with diagonal noise, using scikit-learn as the backend." class="midSize" >}}

A pretty good fit! `scikit-learn`'s implementation is great, but it only deals with the classic **exact** inference, in which we solve for the inverse of the Gram matrix of all the data. This is prohibitively expensive for large datasets, since the complexity of computing this inverse scales cubically with the size of the dataset. More contemporary approaches use approximate versions of this inference.

### GPyTorch

GPyTorch was developed with other forms of Gaussian Processes in mind, plus scalable exact inference. The authors promise exact inference with millions of datapoints, and they achieve this by reducing the computational complexity of inference from {{< katex >}}O(n^3){{< /katex >}} to {{< katex >}}O(n^2){{< /katex >}}. [Check their paper for more details](https://arxiv.org/abs/1809.11165).

Following the tutorials, this is how you would specify the same Gaussian Process we did for scikit-learn:

```python
import torch

from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel

from gpytorch.mlls import ExactMarginalLogLikelihood


class OurGP(ExactGP):
    def __init__(
        self,
        train_inputs: torch.Tensor,
        train_targets: torch.Tensor,
        likelihood: GaussianLikelihood,
    ):
        super().__init__(train_inputs, train_targets, likelihood)

        # Defining the mean
        # The convention is to call it "mean_module",
        # but I call it mean.
        self.mean = ZeroMean()

        # Defining the kernel
        # The convention is to call it "covar_module",
        # but I usually just call it kernel.
        self.kernel = ScaleKernel(RBFKernel())

    def forward(self, inputs: torch.Tensor) -> MultivariateNormal:
        """
        The forward method is used to compute the
        predictive distribution of the GP.
        """
        mean = self.mean(inputs)
        covar = self.kernel(inputs)

        return MultivariateNormal(mean, covar)


def fit_gaussian_process(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> OurGP:
    """
    Fits a Gaussian Process with an RBF kernel to
    the given inputs and outputs
    """

    # ------- Defining the likelihood ----------
    # The likelihood is the distribution of the outputs
    # given the inputs and the GP.
    likelihood = GaussianLikelihood()

    # ------------ Defining the model ------------
    model = OurGP(inputs, outputs, likelihood)

    # ------------ Training the model ------------
    # The marginal log likelihood is the objective
    # function we want to maximize.
    mll = ExactMarginalLogLikelihood(likelihood, model)
    mll.train()

    # We optimize the marginal likelihood just like
    # optimize any other PyTorch model. The model
    # parameters in this case come from the kernel
    # and likelihood.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for iteration in range(3000):
        optimizer.zero_grad()
        dist_ = model(inputs)
        loss = -mll(dist_, outputs).mean()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(f"Iteration {iteration + 1}/{3000}: Loss = {loss.item()}")

    # It is IMPORTANT to call model.eval(). This
    # will tell GPyTorch that the forward calls
    # are now inference calls, computing the 
    # posterior distribution with the given
    # kernel hyperparameters.
    model.eval()

    return model
```

Notice that, to train a model using GPyTorch, we need to write significantly more code: specifying that the likelihood is Gaussian, defining the optimizer explicity, and optimizing the likelihood manually. This might seem verbose, but it's actually giving us plenty of flexibility! We could e.g. include a neural network in our `forward`, use other likelihoods, or interplay with other PyTorch-based libraries.

You can also see the kernel hyperparameters:

```python
print("lengthscale: ", model.kernel.base_kernel.lengthscale.item())
print("output scale: ", model.kernel.outputscale.item())
print("noise: ", model.likelihood.noise_covar.noise.item())
```

Which outputs something pretty similar to what scikit-learn gave us:

```
lengthscale:  1.5464047193527222
output scale:  20.46845245361328
white noise:  0.23084479570388794
```

And the predictive posterior looks like this:

{{< figure src="/static/assets/bo_blogpost/gp_in_gpytorch.jpg" alt="A GP fitted on the noisy evaluations of x * sin(x) using an RBF kernel with diagonal noise and outputscale, using GPyTorch as the backend." class="midSize" >}}

Pretty similar to scikit-learn! Which is to be expected: the equations are exactly the same, and the kernel and likelihood hyperparameters are close.

### Other tools for Gaussian Process Regression

There are three other tools for building GPs that come to mind:
- [GPy, developed by Neil Lawrence and his team when he was still at Sheffield](https://github.com/SheffieldML/GPy). 
- [GPJax, made by Thomas Pinder](https://github.com/JaxGaussianProcesses/GPJax). Instead of using PyTorch for the backend, it uses Jax. I'd love to try it at some point!
- [GPFlow, made by James Hensman and Alexander G. de G. Matthews](https://github.com/GPflow/GPflow). It uses Tensorflow as the backend.

## A visual introduction to Bayesian Optimization

Bayesian Optimization (B.O.) has three key ingredients:

- A black-box **objective function** {{< katex >}}f\colon\mathcal{X}\to\mathbb{R}{{< /katex >}}, where {{< katex >}}\mathcal{X}{{< /katex >}} is usually a subset of some {{< katex >}}\mathbb{R}^D{{< /katex >}}, which we are trying to maximize. {{< katex >}}f{{< /katex >}} is called the objective function.
- A probabilistic surrogate model {{< katex >}}\tilde{f}\colon\mathcal{X}\to\mathbb{R}{{< /katex >}}, whose goal is to approximate the objective function. Since we will use Gaussian Processes, the ingredients we will need are a prior {{< katex >}}\mu_0\colon\mathcal{X}\to\mathbb{R}{{< /katex >}} and kernel {{< katex >}}k{{< /katex >}}.[^other-surrogate-models]
- An **acquisition function** {{< katex >}}\alpha(\bm{x}|\tilde{f}){{< /katex >}} which measures the "potential" of new points using the surrogate model. If a given {{< katex >}}\bm{x}_*\in\mathcal{X}{{< /katex >}} scores high in the acquisition function, then it potentially maximizes the objective function {{< katex >}}f{{< /katex >}}.

[^other-surrogate-models]: (other surrogate models) We are using Gaussian Processes as a surrogate model here, but other ones could be used! The main requirement is for them to allow us to have good uncertainty estimates of the objective function. [Recent research by Yucen Lily Li, Tim Rudner and Andrew Gordon Wilson has explored this direction](https://arxiv.org/abs/2305.20028).

The usual loop in a B.O. goes like this:

1. Fit the surrogate model {{< katex >}}\tilde{f}{{< /katex >}} to the data you have collected so far.
2. Using {{< katex >}}\tilde{f}{{< /katex >}}, optimize the acquisition function {{< katex >}}\alpha{{< /katex >}} to find the best candidate {{< katex >}}\bm{x}{{< /katex >}} (i.e. find {{< katex >}}\bm{x} = \arg\max \alpha(\cdot|\tilde{f}){{< /katex >}}).
3. Query the expensive objective function on {{< katex >}}\bm{x}{{< /katex >}} and repeat from 1 including this new data pair.

We usually stop after finding an optimum that is high enough, or after a fixed number of iterations.[^stopping-criteria]

[^stopping-criteria]: (stopping criteria) It's not so simple. The stopping criteria of BO are a topic that is currently under plenty of research. See the references.

As we discussed in the introduction, examples of black-box objective functions include the output of expensive simulations in physics and biology, or the accuracy of an ML setup. Now we discuss four examples of acquisition functions.

## Thompson Sampling

The simplest of all acquisition functions is to sample one approximation of {{< katex >}}f{{< /katex >}} and to optimize it. This can be done quite easily in most probabilistic regression methods, including Gaussian Processes.

Circling back to our guiding example of {{< katex >}}x\sin(x){{< /katex >}}, let's start by only having {{< katex >}}\mathcal{D}_{\text{init}} = \{(0, f(0))\}{{< /katex >}} in our dataset, and iteratively query the next proposal suggested by Thompson sampling:

{{< figure src="/static/assets/bo_blogpost/bo_w_TS.gif" alt="A GIF showing how Thompson Sampling works: we fit a Gaussian Process, sample from it, and consider this sample's optimum." >}}

At first, the samples are not approximating the objective function at all, and so we end up exploring the domain almost at random. These initial explorations allow us to get a better approximation of the function. TS quickly finds the left optimum, and tends to get stuck there, since further samples are likely to have their maximum there.

### Improvement-based policies

Since we have estimates of the posterior mean {{< katex >}}\mu{{< /katex >}} and the posterior variance {{< katex >}}\sigma^2{{< /katex >}} at each step, we have access to a lot more than just a single sample (as in TS): we can query for probabilistic quantities such as expectations or probabilities.

For each new point {{< katex >}}\bm{x}_*{{< /katex >}}, consider its "improvement":

{{< katex display >}} I(\bm{x}_*; \mathcal{D}) = \max(0, f(\bm{x}_*) - f_{\text{best}}){{< /katex >}}

where {{< katex >}}f_{\text{best}}{{< /katex >}} is the maximum value for {{< katex >}}f(\bm{x}_n){{< /katex >}} in our trace {{< katex >}}\mathcal{D} = \{(\bm{x}_n, f(\bm{x}_n))\}_{n=1}^N{{< /katex >}}. The improvement measures how much we are gaining at point {{< katex >}}f(\bm{x}_*){{< /katex >}}, compared to the current best.

since we have a probabilistic approximation of {{< katex >}}f{{< /katex >}}, we can compute quantities like the **probability of improvement** (PI):

{{< katex display >}}
\alpha_{\text{PI}}(\bm{x}_*; \mathcal{D}) = \text{Prob}[I(\bm{x}_*;\mathcal{D}) > 0] = \text{Prob}[f(\bm{x}_*) > f_{\text{best}}],
{{< /katex >}}

In other words, PI measures the probability that a given point will be better than the current maximum. Another quantity we can measure is the **expected improvement** (EI)

{{< katex display >}} \alpha_{\text{EI}}(\bm{x}_*; \mathcal{D}) = \mathbb{E}[I(\bm{x}_*;\mathcal{D})]{{< /katex >}}

which measures by how much a given {{< katex >}}\bm{x}_*{{< /katex >}} may improve on the current best.[^closed-form]

[^closed-form]: (closed form) These two acquisition functions have closed form when we use vanilla Gaussian Processes. For the technical details, check the references.

Using the same running example and sample we showed for TS, here's how EI fairs:

{{< figure src="/static/assets/bo_blogpost/bo_w_EI.gif" alt="A GIF showing how Expected Improvement-based Bayesian Optimization works: we fit a Gaussian Process and use the uncertainty estimates (mean and covariance) to determine the potential improvement of each point in the domain." >}}

Notice that the right size of this plot now shows the values for {{< katex >}}\alpha_{\text{EI}}{{< /katex >}} in a different scale. At first, all points have almost the same expected improvement (since we haven't explored the function at all, and so we are uncertain). Querying the maximum of {{< katex >}}\alpha_{\text{EI}}{{< /katex >}} iteratively allows us to find not only the left optimum, but also the right one!

### Upper-Confidence Bound 

Another type of acquisition function is the Upper-Confidence Bound (UCB), which optimistically chooses the points are on the upper bounds of the uncertainty. More explicitly, let {{< katex >}}\mu{{< /katex >}} and {{< katex >}}\sigma{{< /katex >}} be the posterior mean and standard deviation on a given point {{< katex >}}\bm{x}{{< /katex >}} after fitting the GP, then

{{< katex display >}} \alpha_{\text{UCB}}(\bm{x}_*; \mathcal{D}, \beta) = \mu(\bm{x}_*) + \beta\sigma(\bm{x}_*), {{< /katex >}}

where {{< katex >}}\beta > 0{{< /katex >}} is a hyperparameter that states how optimistic we are on the upper bound of the variance. High values of {{< katex >}}\beta{{< /katex >}} encourage exploration (i.e. exploring regions with higher uncertainty), while lower values of {{< katex >}}\beta{{< /katex >}} encourage exploitation (staying close to what the posterior mean has learned). To showcase the impact of this hyperparameter, here are two versions of BO on our running example: one with {{< katex >}}\beta=1{{< /katex >}}, and another with {{< katex >}}\beta=5{{< /katex >}}.

{{< figure src="/static/assets/bo_blogpost/bo_w_UCB_100.gif" alt="A GIF showing how Upper Confidence Bound-based Bayesian Optimization works: we fit a Gaussian Process and use the uncertainty estimates to compute the mean plus a multiple of the standard deviation, which we consider an optimistic choice for where to sample next." >}}

Notice that, for {{< katex >}}\beta=1{{< /katex >}}, the optimization gets stuck on the first global optima. This doesn't happen for {{< katex >}}\beta=5{{< /katex >}}:

{{< figure src="/static/assets/bo_blogpost/bo_w_UCB_500.gif" alt="A GIF showing how Upper Confidence Bound-based Bayesian Optimization works: we fit a Gaussian Process and use the uncertainty estimates to compute the mean plus a multiple of the standard deviation, which we consider an optimistic choice for where to sample next. In this case, the beta is chosen to be 5.">}}

{{< figure src="/static/assets/bo_blogpost/bo_w_UCB_500.gif" alt="A GIF showing how Upper Confidence Bound-based Bayesian Optimization works: we fit a Gaussian Process and use the uncertainty estimates to compute the mean plus a multiple of the standard deviation, which we consider an optimistic choice for where to sample next. In this case, the beta is chosen to be 5."  >}}

## Bayesian Optimization in practice: BoTorch

BoTorch is a Python library that interoperates well with GPyTorch for running BO experiments. All the animations I presented above were made by fitting BoTorch's models for exact GP inference, and by using their implementations of the acquisition functions (except for Thompson Sampling).

Following their tutorial (and adapting it to our example), this is a minimal working version of the code for doing a Bayesian Optimization loop:

```python
from typing import Callable, Tuple

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ZeroMean

# Single Task GP is a Gaussian Process on one value,
# using certain defaults for kernel and mean.
# mean: a constant that gets optimized.
# kernel: Matérn 5/2 kernel with output scale.
from botorch.models import SingleTaskGP

# BoTorch provides tools for fitting these
# models easily.
from botorch.fit import fit_gpytorch_model

# Using the UCB acquisition function
from botorch.acquisition import UpperConfidenceBound


def bayesian_optimization(
    objective_f: Callable[[float], float],
    n_iterations: int = 20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Some hyperparameters
    ACQ_BETA = 1.0
    domain = torch.linspace(-10, 10, 200)

    # Querying at 0
    # This could be modified to sample at random
    # from the domain instead.
    inputs_trace = [torch.Tensor([0])]
    outputs_trace = [objective_f(inputs_trace[-1])]

    for _ in range(n_iterations):
        # Consolidating the inputs and outputs
        # for this iteration
        inputs = torch.Tensor(torch.cat(inputs_trace))
        outputs = torch.Tensor(torch.cat(outputs_trace))

        # --- Fitting the GP using BoTorch's tools ---
        # Defining the model.
        # Instead of using the constant mean given by
        # BoTorch as a default, we go for a zero mean.
        gp_model = SingleTaskGP(
            inputs.unsqueeze(1),
            outputs.unsqueeze(1),
            mean_module=ZeroMean(),
        )

        # Defining the marginal log-likelihood
        mll = ExactMarginalLogLikelihood(
            gp_model.likelihood,
            gp_model,
        )

        # Maximizing the log-likelihood
        fit_gpytorch_model(mll)

        gp_model.eval()

        # ---- Maximizing the acquisition function -----
        acq_function = UpperConfidenceBound(model=gp_model, beta=ACQ_BETA)
        # BoTorch expects the input to be
        # of shape (b, 1, 1) in this case.
        acq_values = acq_function(domain.reshape(-1, 1, 1))

        # The next candidate is wherever the
        # acquisition function is maximized
        next_candidate = domain[acq_values.argmax()]

        # ---- Querying the obj. function ---------
        inputs_trace.append(
            torch.Tensor([next_candidate])
        )
        outputs_trace.append(
            torch.Tensor([objective_f(next_candidate)])
        )

    return (
        torch.Tensor(torch.cat(inputs_trace)),
        torch.Tensor(torch.cat(outputs_trace)),
    )
```

## Bayesian Optimization loops for Easom and Cross-in-Tray

Bayesian Optimization works for black-box objective function with general domains {{< katex >}}\mathcal{X}\subseteq\mathbb{R}^D{{< /katex >}}. To showcase this, we can run BO in {{< katex >}}\mathbb{R}^2{{< /katex >}} on the test functions `Easom` and `Cross-in-tray`:

{{< figure src="/static/assets/bo_blogpost/bo_easom.gif" alt="A GIF showing how Bayesian Optimization works in the easom 2D test function."  >}}

{{< figure src="/static/assets/bo_blogpost/bo_cross_in_tray.gif" alt="A GIF showing how Bayesian Optimization works in the cross-in-tray 2D test function."  >}}

`Cross-in-tray` is a particularly difficult function for black-box optimization algorithms based on Evolutionary Strategies.[^David-Ha-blogpost] Let me show you e.g. CMA-ES trying to optimize it:

[^David-Ha-blogpost]: (David Ha's blogpost) If you're not familiar with evolutionary strategies, I can't recommend [David Ha's introduction](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/) enough.

{{< figure src="/static/assets/bo_blogpost/cma_es.gif" alt="A GIF showing how CMA-ES works in the cross-in-tray 2D test function."  >}}

Unless it gets lucky with the initialization, CMA-ES tends to get stuck in one of the many local optima of `Cross-in-tray`. Of course, there are ways to encourage more exploration in Evolutionary Strategies, but this is something that is accounted for automatically in BO. 

Another advantage of BO vs. Evolutionary Strategies is **sample efficiency**: for each generation, an evolutionary strategy must query the objective function as many times as the population size. This is prohibitively expensive if the objective function is e.g. an expensive simulation that cannot be run in parallel.[^sample-efficiency-comparison]

[^sample-efficiency-comparison]: (sample efficiency comparison) In [Sec. 3.4. of my dissertation](https://www.miguelgondu.com/assets/phdthesis.pdf) I run a sample-efficiency comparison between BO and CMA-ES for these two test functions. The results show that BO reliably finds an optima in less queries to the objective function. I plan to run experiments on higher-dimensional examples in the future.

## Bayesian Optimization's drawbacks

Bayesian Optimization is a major achievement of Bayesian statistics. It, however, has major drawbacks and hidden assumptions. Let's list a couple of them:

### How do we optimize the acquisition function?

You might have noticed that, in our running example, we optimized the acquisition function by evaluating it on the entire domain, and then querying its maximum (i.e. we just did a grid search). This obviously doesn't scale to higher dimensions!

There are several alternatives to this. Since we can evaluate the acq. function easily, we could simply use Evolutionary Strategies (like CMA-ES, shown above) to optimize it. Otherwise, we could rely on gradient-based optimization.[^gps-are-differentiable]

[^gps-are-differentiable]: (GPs are differentiable) Depending on our choice of kernel, we can also compute distributions over the gradient of a Gaussian Process. I would love to dive deeper here, but I think it's a topic on its own. Check [Sec. 9.4. of *Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/RW9.pdf) or [this paper on estimating gradients of acquisition functions](https://proceedings.neurips.cc/paper_files/paper/2018/file/498f2c21688f6451d9f5fd09d53edda7-Paper.pdf) for more details.

Other alternatives to high dimensional inputs focus on running the optimization of the acquisition function on only **trust regions**, centered around the best performing points. These trust regions are essentially small cubes, whose size adapts over the optimization.

### (Exact) Gaussian Processes don't scale well

Vanilla versions of kernel methods and Gaussian Processes don't scale well with either the size of the dataset, or the dimension of the input space. Folk knowledge says that (vanilla) Gaussian Processes don't fit well above 20 input dimensions.[^folk]

[^folk]: (folk) Check for example [this Stack Exchange question](https://stats.stackexchange.com/questions/564528/why-does-bayesian-optimization-perform-poorly-in-more-than-20-dimensions).

When it comes to large datasets, exact GP inference may be possible with GPyTorch and their implementation. Approximate inference is usually done through inducing points, which are a set of points that's representative of the entire dataset.[^inducing-points]

[^inducing-points]: (inducing points) The classic reference to cite for inducing points is [a paper by Quiñonero-Candela and Rasmussen](https://proceedings.neurips.cc/paper_files/paper/2018/file/498f2c21688f6451d9f5fd09d53edda7-Paper.pdf). Check it if you're interested!

High dimensional inputs are usually projected to lower-dimensional spaces, using either linear mappings (assuming e.g. that there are several dimensions that are uninformative and can be ignored), or using neural networks. Another alternative is to provide a handcrafted mapping from solution to a so-called *behavior space* (as they do in the Intelligent Trial-and-Error paper, using MAP-Elites).[^intelligent-trial-and-error]

[^intelligent-trial-and-error]: (intelligent trial-and-error) See [*Robots that can adapt like animals*, by Cully et al](https://www.nature.com/articles/nature14422). The authors learn a corpus of different robot controllers using handcrafted dimensions, and then adapt to damage using Bayesian Optimization.

### Parallelizing

As formulated above, BO is sequential in nature: we fit a surrogate model to the current dataset, we use this surrogate model to construct an acquisition function, which we optimize to get the next point. Parallelizing BO isn't trivial, and the literature shows that the principled ways for parallelization are highly dependant on the acquisition function. See the references for more details.

### Stopping criteria

A problem that is transversal to black-box optimization algorithms is knowing when to stop. Most practitioners just have a fixed amount of compute, or decide to stop when they have found a solution that is "good enough" for them (e.g. a hyperparameter search that's above a certain accuracy).

There are theory-driven ways to solve this problem. Gaussian Processes allows for computing notions of "regret", which are being used to define optimization processes that stop automatically.

## Conclusion

Bayesian Optimization lets you optimize black-box functions (i.e. functions that have no closed form, and are expensive to query) by approximating them using a probabilistic surrogate model. In this blogpost, we explored using Gaussian Processes as surrogates, and several acquisition functions that leverage the uncertainties that GPs give. We briefly showed how these different acquisition functions explore and exploit the domain to find a suitable optimum.

When compared to e.g. Evolutionary Strategies, we briefly saw how CMA-ES might get stuck in local optima if the landscape is tough. BO is also more sample efficient, since Evolutionary Strategies need to query the objective function in entire populations at each generation.

We wrapped up by discussing some of the drawbacks of BO: optimizing the acquisition function is not trivial in high dimensions, Gaussian Processes rely on approximations to fit large or high dimensional datasets, and parallelization depends highly on the choice of acquisition function.

## Cite this blogpost

If you found this blogpost useful, feel free to cite it.

```
@online{introToBO:Gonzalez-Duque:2023,
  title = {Bayesian Optimization using Gaussian Processes: an introduction},
  author = {Miguel González-Duque},
  year = {2023},
  url = {https://miguelgondu.com/blogposts/2023-04-08/intro-to-bo}
}
```

## References

### Gaussian Processes

- [*Gaussian Processes for Machine Learning* by Rasmussen and Williams. Chapters 2, 4 and 9](https://gaussianprocess.org/gpml/). This book is "the bible" of Gaussian Processes.
- [*Probabilistic Numerics* by Hennig, Osborne and Kersting. Secs. 4.2 and 5.5](https://www.probabilistic-numerics.org/textbooks/). I really like their presentation of Gaussian algebra and of Gaussian Processes. The computations for the posterior GP can be found either here or in the bible.
- [*A Unifying View of Sparse Approximate Gaussian Process Regression*, by Quiñonero-Candela and Rasmussen](https://proceedings.neurips.cc/paper_files/paper/2018/file/498f2c21688f6451d9f5fd09d53edda7-Paper.pdf). A paper about inducing points/active sets/pseudo-inputs in GPs.
- [Exact Gaussian Processes on a Million Data Points](https://arxiv.org/abs/1903.08114).

### Bayesian Optimization

- [*Taking the human out of the loop: a review of Bayesian Optimization*, by Shahriari et al.](https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf). What a classic.
- [*Maximizing acquisition functions for Bayesian optimization*, by Wilson, Hutter and Deisenroth.](https://proceedings.neurips.cc/paper_files/paper/2018/file/498f2c21688f6451d9f5fd09d53edda7-Paper.pdf) This paper discusses how to approximate the gradients of acquisition functions using MC, and leveraging differentiable kernels.
- [*Scalable Global Optimization via Local Bayesian Optimization*, by Eriksson et al.](https://arxiv.org/abs/1910.01739). This paper introduces `TurBO`, or Bayesian Optimization with trust regions.
- Maximilian Balandat has several papers on parallel versions of Bayesian Optimization. [Check his google scholar for some examples](https://scholar.google.com/citations?user=xvnfpkMAAAAJ&hl=en).
- The stopping criteria is currently under plenty of research. Check for example the work of [Anastasiia Makarova](https://scholar.google.com/citations?user=skAF5s8AAAAJ&hl=en&oi=sra). In particular [*Automatic Termination for Hyperparameter Optimization*](https://proceedings.mlr.press/v188/makarova22a/makarova22a.pdf).
- [*Robots that can adapt like animals*, by Cully et al](https://www.nature.com/articles/nature14422). A great example of Bayesian Optimization on top of handcrafted features.
