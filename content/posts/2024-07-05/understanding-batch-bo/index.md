---
date: "2024-07-06"
title: "Understanding batch Bayesian optimization"
slug: batch-bo
images:
- static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
description: A brief introduction to batch Bayesian optimization, comparing it against CMA-ES
---

By its nature, Bayesian optimization is _sequential_, evaluating the potential of 1 point in input space at a time. This is
rather unfortunate, because several problems out there are well suited for evaluating large batches at once. Think for example
of Reinforcement learning traces: an agent can be trained on several runs at the same time using e.g. [Isaac Gym]() or [Brax]().
Other examples come from biology and chemistry, where practitioners evaluate their assays in plates that can hold up to 96
"experiments" at the same time.[^an-example]

[^an-example]: Check [this paper]() and [this technology]().

Since there's a big practical need for batching BO, several methods have been proposed since the 2000s. In this blogpost
I explain some of them, explore the tools already available in frameworks like `BoTorch`, and compare against the default
choice for evolutionary strategies (another black-box optimization technique which _is meant_ to work on batches, or populations).

[Image of a plate, or image of several robots being trained with caption: sometimes we can parallelize the evaluation of our objective.]

The code used for this blogpost is shared on [the code companion]().

# Bayesian optimization: a recap

But first, let's start with a brief recap of Bayesian optimization to set up the notation. I'll go pretty fast; [one of my previous blogposts gives an introduction to BO if you're interested](/blogposts/2023-07-31/intro-to-bo).

There are three ingredients in BO:
- An objective function {{< katex >}}f\colon\mathbb{R}^D\to\mathbb{R}{{< /katex >}} which is expensive to query, and which you're interested in maximizing.
- A surrogate model (usually Gaussian Process Regression) of the objective function {{< katex >}}\tilde{f}\sim\text{GP}(\mu, k){{< /katex >}}.
- An acquisition function {{< katex >}}\alpha(\bm{x};\tilde{f}){{< /katex >}} which uses the approximation of the objective function to propose new points to evaluate in. Acquisition functions are maximized at points {{< katex >}}\bm{x}\in\mathbb{R}^D{{< /katex >}} that have potential to maximize the objective.

With these three ingredients, we optimize the objective function {{< katex >}}f{{< /katex >}} **sequentially** by querying the best point {{< katex >}}x_{\text{best}}{{< /katex >}} according to the acquisition function {{< katex >}}\alpha{{< /katex >}}, and adding the pair {{< katex >}}(x_{\text{best}}, f(x_{\text{best}})){{< /katex >}} to the dataset.

As we have described it so far, this framework only allows us to query a single point {{< katex >}}x_{\text{best}}{{< /katex >}} at each iteration. The main reason: we choose to focus on the maxima of the acquisition function. To opimize in a batch, it's necessary to (i) either consider a different acquisition function that works on batches instead of single points, or (ii) squeeze more information out of the acquisition functions we have already defined.

I'll now dive deeper into versions of these modifications, starting with how some acq. functions are easily parallelizable, following up with batch versions of acq. functions, and finishing with alternatives that modify the acq. functions we already have.

# Thompson sampling is easily parallelizable
- Parallel and distributed thompson sampling for large-scale accelerated exploration of chemical space. by Hernández-Lobato et al.
- Parallelised bayesian optimisation via thompson sampling by the ADD-GP-UCB crowd.


# Batch versions of Expected Improvement

- Kriging is well-suited ... paper.

## Kriging believer

## Constant liar

# Penalizing locality (González et al.)
- Local penalization work by Javier G.

# Batch UCB
- Parallelizing exploration-exploitation tradeoffs in gaussian process bandit optimization

# `TuRBO` was meant to be parallel
- The turbo paper

# Some simple comparisons

## The silliest baseline: sampling entirely at random.

## A more powerful baseline: CMA-ES

