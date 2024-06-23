---
date: "2024-06-21"
title: "A map of high-dimensional Bayesian optimization: introduction"
slug: a-map-part-1
image: /static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
images:
- static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
description: Starting a map of high-dimensional Bayesian optimization
---

> This blogpost is the first part of a series in which I take [a paper we recently published]() and tutorialize it. It assumes familiarity with Gaussian Processes and Bayesian Optimization. [Check one of my previous blogposts if you need a refresher](/blogposts/2023-07-31/intro-to-bo).

## Introduction to the introduction

[high-dimensional BO is pretty important nowadays: AutoML, self-driving labs, drug discovery, protein engineering]

Bayesian optimization (BO) shows great promise for several tasks: researchers are using it to [automate the development of Machine Learning models](), [build self-driving labs](), and [create entire new families of antibiotics]().

These tasks often involve high-dimensional inputs with a lot of structure. Two examples: to derive a chemical compound you might modulate several variables: the pH and type of the solvent, the amount of different solutions to mix in, the temperature, the pressure... or to quickly optimize a Machine Learning model to the best accuracy possible, you might need to modulate the choice of opimizer, the learning rate, the number of layers and neurons.

We need, then, **high-dimensional Bayesian optimization** over highly structured inputs. And researchers have stepped up: high-dimensional BO (HDBO) is a large research field, and we recently wrote a paper updating surveys of the field. This survey provides a taxonomy of HDBO methods divided into 7 families:
- **Variable Selection**,
- **Linear embeddings**,
- **Trust regions**,
- **Gradient information**,
- **Additive models**,
- **Non-linear embeddings** and
- **Structured Spaces**.

Here's a visual overview (click it for a version with links to all the references and open source implementations):

{{< figure src="/static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg" alt="A timeline of high-dimensional Bayesian optimization." class="largeSize" title="A timeline and taxonomy of high-dimensional Bayesian optimization. Click it for an interactive version with links to papers and open source implementations." link="/assets/hdbo_timeline.pdf" >}}

**In this blogpost** and the ones that follow I will dive into this taxonomy and each of the families. The final goal is to provide a _map_ of HDBO: a comprehensive overview, tutorialized with code. This blogpost in particular presents an introduction to HDBO, and a couple of baselines. The next ones will dive into the families of the taxonomy.

## Our focus: discrete sequence optimization

I plan to focus this overview on **discrete sequence optimization** problems. In these, we explore a space of "words" of length {{< katex >}}L{{< /katex >}}. Of the problems mentioned above, drug discovery and protein engineering can be framed as discrete sequence optimization problems: both small molecules and proteins can be represented as sentences, as we will see later in this post.

Let's formalize what we mean: consider a vocabulary of **tokens** {{< katex >}}v_1, \dots v_V{{< /katex >}}. We are trying to optimize a function {{< katex >}}f\colon\mathcal{X}_L\to\mathbb{R}{{< /katex >}} where {{< katex >}}\mathcal{X}_L = \{\bm{s} = (s_1,\dots, s_L) | s_i \in \{v_1,\dots, v_V\}\}{{< /katex >}} is the space of sentences of length {{< katex >}}L{{< /katex >}}. Our goal is finding
{{< katex display >}}
\argmax_{\bm{s}\in\mathcal{X}_L} f(\bm{s})
{{< /katex >}}

The next section introduces a guiding example we will use for this blogpost and the ones that follow.

## A guiding high-dimensional example: small molecules

One example of a discrete sequence optimization problem that is highly relevant for drug discovery is **small molecule optimization**. We could, for example, optimize small molecules such that they bind well to a certain receptor, or active site in a protein.

We can represent molecules in several ways (e.g. as graphs, as 3D objects...).[^small-molecule-representation] Here, we focus on the [SELFIES representation](). SELFIES are a follow-up on [SMILES](), which were originally deviced as a way to reprenset chemical molecules as text for [TODO: ADD].

[^small-molecule-representation]: [Here is a nice survey on small molecule representations]() in case you're curious.

The problem with SMILES that not all sequences correspond to valid molecules. SELFIES mitigates this issue by [TODO: add].

You can find the SMILES representations of e.g. aspirine by checking on PubChem:

[Screenshot of pubchem, with SMILES highlighted]

And you can transform between SMILES and SELFIES [using the Python package `selfies`, developed by Aspuru-Guzik's lab]().

In these blogposts, we will work [on the Zinc250k dataset](). [TODO: add a small description of the dataset here.]

Zinc250k can easily be downloaded using `TorchDrug` in several of the representations mentioned above (including, of course, as SMILES strings). We transform these SMILES into SELFIES, and store e.g. the maximum sequence length, and the entire alphabet used.[^you-can-easily-replicate-this]

[^you-can-easily-replicate-this]: We have included all these pre-processing scripts [in the repository of our project (including a `conda` env that is guaranteed to build)]().

[Img of the longest molecule, img of the smallest molecule, their SELFIES]

Let me also show you the entire vocabulary that arises from this:

[Img of the vocabulary]

[The entire alphabet, sequence length of 70]

In the formal maths language that we introduced in the previous section, {{< katex >}}v_1, \dots v_V{{< /katex >}} correspond to the {{< katex >}}V=64{{< /katex >}} unique tokens, and {{< katex >}}\mathcal{X}_L{{< /katex >}} corresponds to all the sequences with length {{< katex >}}L=70{{< /katex >}}.

## Optimizing small molecules (or the PMO benchmark)

[The PMO benchmark]

## The "worst" baseline: sampling at random

[Sampling tokens at random]

[plot with a line going up or down]

## A less silly baseline

[Evolving the best level through random sampling]

## Can we optimize SELFIES using Bayesian Optimization?

[One-hot representations of SELFIES Zinc250k]

[Can vanilla BO find anything in one-hot space?]

[[most likely not, check the results of that experiment]]

## What comes next?

[Several strategies are being proposed for dealing with high-dimensional inputs, and in the following blogposts we'll discuss each one.]

[Pointing back to the pie, and adding a brief explanation of each family in the taxonomy]
