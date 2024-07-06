---
date: '2024-07-06'
title: Understanding batch Bayesian optimization
slug: batch-bo
images:
  - static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
description: A brief introduction to batch Bayesian optimization
format: hugo-md
jupyter: python3
---


# BO: a recap

# We need different acq. functions

# The silliest baseline: sampling entirely at random

# A more powerful baseline: CMA-ES (or EvoStrats).

# Thompson sampling is easily parallelizable

-   Parallel and distributed thompson sampling for large-scale accelerated exploration of chemical space. by Hernández-Lobato et al.
-   Parallelised bayesian optimisation via thompson sampling by the ADD-GP-UCB crowd.

# Batch versions of Expected Improvement

-   Kriging is well-suited ... paper.

## Kriging believer

## Constant liar

# Penalizing locality (González et al.)

-   Local penalization work by Javier G.

# Batch UCB

-   Parallelizing exploration-exploitation tradeoffs in gaussian process bandit optimization

# `TuRBO` was meant to be parallel

-   The turbo paper
