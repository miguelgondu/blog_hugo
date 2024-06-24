---
date: "2024-06-21"
title: "A map of high-dimensional Bayesian optimization: introduction"
slug: a-map-part-1
images:
- static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg
description: Starting a map of high-dimensional Bayesian optimization (of discrete sequences)
---

<!-- > This blogpost is the first part of a series in which I take [a survey on high-dimensional Bayesian optimization we recently published]() and tutorialize it. It assumes familiarity with Gaussian Processes and Bayesian Optimization. [Check one of my previous blogposts if you need a refresher on those topics](/blogposts/2023-07-31/intro-to-bo). -->

>  *It is written that animals are divided into*
> 1. *those that belong to the Emperor,*
> 2. *embalmed ones,*
> 3. *those that are trained,*
> 4. *suckling pigs,*
> 5. *mermaids,*
> 6. *fabulous ones,*
> 7. *stray dogs,*
> 8. *those that are included in this classification,*
> 9. *those that tremble as if they were mad,*
> 10. *innumerable ones,*
> 11. *those drawn with a very fine camel's hair brush,*
> 12. *others,*
> 13. *those that have just broken a flower vase,*
> 14. *those that resemble flies from a distance.*
>
> *[...]*
> *I have noted the arbitrariness of Wilkins, of the unknown (or apocryphal) Chinese encyclopedist, and of the Bibliographical Institute of Brussels; obviously there is no classification of the universe that is not arbitrary and conjectural.*
>
> The Analytical Language of John Wilkins, by Jorge Luis Borges.


# Introduction to the introduction

<!-- [high-dimensional BO is pretty important nowadays: AutoML, self-driving labs, drug discovery, protein engineering] -->

Bayesian optimization (BO) shows great promise for several tasks: It is used for [automating the development of Machine Learning models](), [building self-driving labs](), and [creating entire new families of antibiotics]().[^need-a-refresher?]

[^need-a-refresher?]: If you need a refresher on Bayesian optimization, [check the blogpost I wrote a couple of months ago]().

These tasks often involve high-dimensional inputs with a lot of structure. Two examples: to derive a chemical compound you might modulate several variables: the pH and type of the solvent, the amount of different solutions to mix in, the temperature, the pressure... or to quickly optimize a Machine Learning model to the best accuracy possible, you might need to modulate the choice of opimizer, the learning rate, the number of layers and neurons.

We need, then, **high-dimensional Bayesian optimization** over highly structured inputs. High-dimensional BO (HDBO) is a large research field, and we recently wrote a paper with an updated survey and taxonomy composed of 7 families of methods:
- **Variable Selection**: choosing only a subset of variables and optimizing there.
- **Linear embeddings**: Optimizing in a linear subspace,
- **Trust regions**: Limiting the optimization of the acquisition function to a small cube in the high-dimensional space,
- **Gradient information**: Using information from the gradient predicted by the underlying GP,
- **Additive models**: Decomposing the objective function into a sum of functions with less variables,
- **Non-linear embeddings**: Using e.g. neural networks to learn latent spaces and optimizing therein,
- **Structured Spaces**: 

Here's a visual overview (click it for a version with links to all the references and open source implementations):

{{< figure src="/static/assets/hdbo_blogposts/a_map_part_1/hdbo_pie.jpg" alt="A timeline of high-dimensional Bayesian optimization." class="largeSize" title="A timeline and taxonomy of high-dimensional Bayesian optimization. Click it for an interactive version with links to papers and open source implementations." link="/assets/hdbo_timeline.pdf" >}}

**In this blogpost** and the ones that follow I will dive into this taxonomy and each of the families. The final goal is to provide a _map_ of HDBO: a comprehensive overview, tutorialized with code. This blogpost in particular presents an introduction to HDBO, and a couple of baselines. The next ones will dive into the families of the taxonomy.

I want to emphasize that [I'm building on prior work done by Binois & Wycoff](), as well as [Santoni et al.'s comparison of HDBO methods](). The taxonomy shown here is an extention of what's proposed in these two papers.

# Our focus: discrete sequence optimization

I plan to focus this overview on **discrete sequence optimization** problems. In these, we explore a space of "words" of length {{< katex >}}L{{< /katex >}}. Of the problems mentioned above, drug discovery and protein engineering can be framed as discrete sequence optimization problems: both small molecules and proteins can be represented as sentences, as we will see later in this post.

Let's formalize what we mean: consider a vocabulary of **tokens** {{< katex >}}v_1, \dots v_V{{< /katex >}}. We are trying to optimize a function {{< katex >}}f\colon\mathcal{X}_L\to\mathbb{R}{{< /katex >}} where {{< katex >}}\mathcal{X}_L = \{\bm{s} = (s_1,\dots, s_L) | s_i \in \{v_1,\dots, v_V\}\}{{< /katex >}} is the space of sentences of length {{< katex >}}L{{< /katex >}}. Our goal is finding
{{< katex display >}}
\argmax_{\bm{s}\in\mathcal{X}_L} f(\bm{s})
{{< /katex >}}

The next section introduces a guiding example we will use for this blogpost and the ones that follow.

# A guiding high-dimensional example: small molecules

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

<!-- [The entire alphabet, sequence length of 70] -->

In the formal maths language that we introduced in the previous section, {{< katex >}}v_1, \dots v_V{{< /katex >}} correspond to the {{< katex >}}V=64{{< /katex >}} unique tokens, and {{< katex >}}\mathcal{X}_L{{< /katex >}} corresponds to all the sequences with length {{< katex >}}L=70{{< /katex >}}.

The search space {{< katex >}}\mathcal{X}_L{{< /katex >}} is **enormous**. Enumerating it would take all the humans that have ever existed way longer than the age of the universe. Looking at the combinatorics of it, we have 70 possible locations, and for each we have 64 possible values: {{< katex >}}70^{64}{{< /katex >}} possible SELFIES sentences.[^the-combinatorics]

[^the-combinatorics]: So, the estimated life of the universe is in the order of magnitude of {{< katex >}}10^{17}{{< /katex >}} seconds; the estimated number of humans that have ever existed is around {{< katex >}}10^{11}{{< /katex >}}. That barely makes a dent on {{< katex >}}70^{64}{{< /katex >}}.

# Optimizing small molecules (or the PMO benchmark)

We have defined our search space {{< katex >}}\mathcal{X}_L{{< /katex >}}, now we're missing examples of objective functions {{< katex >}}f\colon\mathcal{X}_L\to\mathbb{R}{{< /katex >}}.

Molecular optimization is a hot topic, and in these last couple of years several collections of black box functions have been developed. At first, we used to optimize relatively simple metrics like the *Quantitative Estimate of Druglikeness* (QED) or *log-solubility* (LogP). Attention quickly turned into more complex objectives, with [Nathan et al. proposing GuacaMol](): a benchmark of several black box functions for molecule optimization. Soon after [Gao et al.]() extended this benchmark [by proposing **Practical Molecular Optimization** (PMO)](), focusing on sample efficiency.[^about-tdc]

[^about-tdc]: The black boxes inside PMO can be queried easily using [the Therapeutics Data Commons Python]() package. We recently extended this benchmark a bit in our unified testing framework of [poli]() and [poli-baselines]().

GuacaMol and PMO include several different *families* of black-box objectives (e.g. being similar to a certain well-known medicine, or matching a given molecular formula). It's worth noting that these black-boxes often rely on simple calculations over the molecule, or on data-driven oracles. They are often **instant to query**, which is not ideal for our setting: we might grow lazy and get the wrong impression of the speed of our algorithms, especially when actual applications can be in the timeframe of weeks or months per evaluation of the black box.

We will optimize small molecules on the PMO benchmark.

# The "worst" baseline: sampling at random

<!-- [Sampling tokens at random] -->
One of the silliest ways to optimize a black-box objective {{< katex >}}f\colon\mathcal{X}_L\to\mathbb{R}{{< /katex >}} is to consider randomly sampled sentences. Let me show you some examples of what happens when we sample random SELFIES from our alphabet:

[img of random molecules]

We can implement such a solver easily. Here we will use a framework for benchmarking black boxes I've been developing called `poli-baselines`:

[Implementation of a random sampler]

Let's try to optimize one of the black-boxes inside PMO: `drd2_docking` [TODO:CHECK-A-GOOD-ONE]. [TODO: add description of the black-box].

To keep testing homogeneous, let's give every algorithm we will test in these blogposts the same evaluation budget, as well as the same initialization and max. runtime.

Here's the result of just randomly sampling sentences of SELFIES tokens for 500 iterations (with an "initialization" of 10 evaluations).

[plot with a line going up]

As far as baselines go, this one is the most naïve. It is surprisingly powerful at times, and given enough budget it can surpass whatever other technique. [TODO: read up on the no-free-lunch theorem]. Here, the performance can get pretty high because there's plenty of heavy lifting being done by the SELFIES encoding. If we were to optimize on SMILES space, I would assume several of our samples would correspond to gibberish, invalid SMILES. 

# A less silly baseline: discrete hill-climbing

<!-- [Evolving the best level through random sampling] -->

Instead of randomly sampling each time, why don't we just take the best performing model and *mutate* it at random? Let's implement this simple logic in `poli-baselines` again:

[Implementation of fixed-length random mutations]

This is actually a very rudimentary version of a genetic algorithm in which we're only selecting the best element for mutation. If we were thorough, we would actually implement a proper genetic algorithm here. It's worth noting that [`mol-ga`]() (a relatively simple genetic algorithm designed for SMILES) is allegedly performing better than _all_ the 28 methods originally tested in the PMO benchmark. [Check the paper by Tripp and Hernández-Lobato here]().

# Can we optimize SELFIES using Bayesian Optimization?

<!-- [One-hot representations of SELFIES Zinc250k] -->
A first attempt at optimizing in $\mathcal{X}_L$ using Bayesian optimization would be to transform this discrete problem into a continuous one. We will dive deeper on how to do this in the **non-linear embeddings** and **structured spaces** families, but for now let's do the simplest alternative: running Bayesian optimization in one-hot space.

By one-hot space, we mean encoding a given sequence as a collection of vectors
$\bm{s} = (s_1, \dots, s_L) \mapsto \bm{X} = (\bm{x}_1, \dots, \bm{x}_L)$ where each $\bm{x}_i$ is a vector of length $V$ with $1$ for the index of the corresponding token $s_i$. Here's a visual representation:

[A visual representation like the one I did for SMB]

Notice that $\bm{X}$ is still pretty large: if we flatten it, it will be a vector of length $L * V = 70 * 64 = 4480$. Remember that GPs start struggling around 30 dimensions (depending on the training set-up and the amount of data)[^vanilla-gps-blogpost]. As we discussed in the previous blogpost about vanilla Gaussian Processes, we will need plenty of samples to even _fit_ such a surrogate model.

[^vanilla-gps-blogpost]: I explored this in the previous blogpost, in case you are curious.

<!-- [Can vanilla BO find anything in one-hot space?] -->

Still, let's build a simple Bayesian Optimization loop in this space, and see how it fairs. [TODO: add the implementation and discuss it]

[Implementation]

[[most likely not, check the results of that experiment]]

# What comes next? Aternatives for high-dimensional BO

<!-- [Several strategies are being proposed for dealing with high-dimensional inputs, and in the following blogposts we'll discuss each one.] -->
It's understandable that standard Bayesian optimization is not performant in one-hot space in this setting. A 4480-dimensional continuous search space is awfully large for GPs. In the coming blogposts we'll dive into the families of the taxonomy we recently proposed.

<!-- [Pointing back to the pie, and adding a brief explanation of each family in the taxonomy] -->
Here's the plan: I will start by discussing methods that explore **structured spaces**. Most of these work directly on the discrete representations. Afterwards, I will dive into **non-linear embeddings** and on using latent representations of our data for easing the optimization. These continuous latent spaces will open the doors for all the other families as well.

# Conclusion

In this blogpost I introduced a taxonomy of high-dimensional Bayesian optimization methods, building on the work of [Santoni et al.]() and [Binois & Wycoff]().

We focus our attention on discrete sequence optimization, which is an alternative for drug design. Small molecules can be represented as discrete sequences, and we will use them as a guiding example.

This blogpost also introduces two simple baselines, and compares standard BO on one-hot representations of discrete sequences. Results show that standard BO is not as competitive, which is understandable given the size and nature of the continuous space that is being explored.
