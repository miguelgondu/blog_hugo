---
date: "2025-02-02"
title: "The future of black box optimization benchmarking is procedural"
description: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
summary: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
---

In 2021, NeurIPS opened the first call for papers on a *Datasets & Benchmarks* track.
The organizers of the conference, which is one of the four or five largest Machine Learning (ML)
conferences in the planet, highlighted both datasets and benchmarks as foundational components of
ML research, vital to the future of the field. Other conferences (like ICLR) have followed suit,
opening similar tracks aimed at celebrating research on these two subfields (gathering and curating
datasets, or benchmark construction).

This blogpost discusses 
[a recent position paper by Saxon et al. called *Benchmarks as Microscopes: a Call for Model Metrology*.](https://openreview.net/forum?id=bttKwCZDkm&noteId=Yfwy2d4fiT)
Saxon et al. argue that the benchmarks we construct ought to be constrained, dynamic, and plug-and-play.
I'll dive deeper into what each of these words mean in the context of black box optimization,
and I'll argue that researchers in Procedural Content Generation (PCG) are in an ideal position to
construct such benchmarks. Their paper focuses on Natural Language Processing and language models,
but I think their arguments translate well to other domains.

With this blogpost I try to bridge these two fields, encouraging PCG researchers
to take a look into benchmark construction for other fields (becoming what Saxon et al.
call *model metrologists*). It assumes a passing familiarity with black box optimization.

# A definition and examples of benchmarking

But first, just so we are all in the same page, let's define what a benchmark is. To me,
a benchmark is a collection of problems in which a given system or model is tested. The
performance of a model is gauged by a collection of metrics, for example
- classification accuracy in image recognition,
- number of correctly answered questions in the BAR exam in language modelling,
- [prediction error with respect to the solutions of a differential equation](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf) in scientific ML.
- [prediction error when mutating proteins](https://proteingym.org/) in structural biology.

# Benchmarking in black box optimization

In the case of black box optimization, this collection of problems tends to be
a [family of synthetic functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
Check several papers on black box optimization, and you will always find regret plots
with performance on these.[^some-examples]

[^some-examples]: Check for example [this]() paper, or [this]() paper, or [this other]()
paper.

[Images of some of these].

We, as a community, have devoted a significant efforts on keeping track
of the performance of our optimizers on families of synthetic benchmarks.
There are for example entire platforms dedicated to measuring the performance
of black box optimizers on these synthetic functions. An example of such is
Hansen et al.'s *COmparing Continuous Optimizers* ([COCO](https://coco-platform.org/)).

I would like to argue that **we are doing it wrong**. The goal of these synthetic functions
is, allegedly, to give us proxies for real-world optimization tasks, but we don't know
how the real world is until we face it. These synthetic black box functions (as well
as many other benchmarks in black box optimization for chemistry or biology) fall short
when describing the potential of an optimizer in a given real-world task.

The current state of our benchmarking is such that, when a practitioner arrives with an
interesting problem to optimize, we don't have much to show, and we can't confidently
assess which algorithms would perform best in their problems. Best we can say is which
algorithms perform best on some highly synthetic, general set of examples.

<!-- Admitedly, these synthetic functions are mainly used to detect whether the algorithm
works (i.e. they're used as a sanity check, and not as grounds for decision-making).
Some of these black boxes have specific behaviors (e.g. a single optima in a very
flat region, or several local optima with deceiving gradient information), which
also allow us to detect the strengths and weaknesses of our black box optimizers. -->

# Model metrology (or best practices when it comes to benchmarking)

<!-- [talking about Saxon's work] -->
Saxon et al. gave me the language to formulate exactly what we are doing wrong: these
benchmarks are *too general*, and we should be constructing specialized, **constrained**
problems to test our optimizers, focusing on what a given practitioner needs. Moreover,
our benchmarks are *static* (i.e. they're the same over time), and thus we run the risk
of overfitting to them. We need **dynamic** benchmarks. One aspect that we have nailed,
though, and that we should keep in future iterations is that our benchmarks are easily
deployable, or **plug-and-play**.

These are the three keywords they claim make good benchmarks. Good benchmarks are

- **constrained**, bounded and highly specific to a given application,
- **dynamic** to avoid overfitting to certain behaviors, and
- **plug-and-play**, easy to deploy.

The authors call for a new research field, entirely devoted to evaluating model performance,
called *model metrology*. A model metrologist is someone who has the tools to create such
constrained, dynamic & plug-and-play benchmarks, tailored to the needs of a given practitioner.

# Procedural Content Generation

[A brief presentation of PCG as]

# PCG researchers know about creating constrained & dynamic environments

[...]

# An example: closed-form test functions in biology

[Ehrlich functions]

# Another example: PCG use in Reinforcement Learning

[PCG is already used to create agents that perform certain tasks in Reinforcement Learning]

# Conclusion

