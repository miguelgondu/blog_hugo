---
date: "2025-02-02"
title: "The future of black box optimization benchmarking is procedural"
description: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
summary: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
---
<!-- 
In 2021, NeurIPS opened the first call for papers on a *Datasets & Benchmarks* track.
The organizers of the conference, which is one of the four or five largest Machine Learning (ML)
conferences in the planet, highlighted both datasets and benchmarks as foundational components of
ML research, vital to the future of the field. -->

This blogpost discusses 
[a recent position paper by Saxon et al. called *Benchmarks as Microscopes: a Call for Model Metrology*.](https://openreview.net/forum?id=bttKwCZDkm&noteId=Yfwy2d4fiT)
The authors argue that the benchmarks we construct ought to be constrained, dynamic, and plug-and-play.
I'll dive deeper into what each of these words mean,
and I'll argue that researchers in Procedural Content Generation (PCG) are in an ideal position to
construct such benchmarks. Their paper focuses on Natural Language Processing and language models,
but I think their arguments translate well to my domain: black box optimization.

With this blogpost I try to bridge the fields of black box benchmarking and PCG, encouraging PCG researchers
to take a look into benchmark construction for black box optimization and other fields (becoming what Saxon et al.
call *model metrologists*).

Since the goal is to bring people from different fields together, I won't assume you have knowledge on black box
optimization nor PCG. If you're already familiarized with these ideas, feel free to skip the sections that
introduce said topics.

# The name of the game: black box optimization

<!-- [The definition of a black box] -->
Think of a black box function as any process that transforms a set of inputs {{< katex >}}x_1, x_2, \dots, x_d{{< /katex >}} into a real number {{< katex >}}r = r(x_1, \dots, x_d){{< /katex >}}. This definition is vague on purpose, and allows us to define several things as black box functions. Two examples: when you log into Instagram, the app that you are shown is "parametrized" by several {{< katex >}}x_i{{< /katex >}}s, such as, for example, {{< katex >}}x_1{{< /katex >}} could be whether your reels are shown in a loop or just once, {{< katex >}}x_2{{< /katex >}} could be the person you first see on your timeline, and {{< katex >}}x_3{{< /katex >}} could be how often you're shown an ad.[^this-is-just-a-hypothesis] One example of a metric {{< katex >}}r(x_1, x_2, x_3){{< /katex >}} would be how long you spend on the app. Another less macabre example is making a recipe for cookies, with the different {{< katex >}}x_i{{< /katex >}}s being ingredient amounts, and the result being a combination of how tasty it is compared to the original recipe.[^google-did-this]

[^this-is-just-a-hypothesis]: These are just examples, of course. I have zero clue which variables they use in their A/B testing, or how they define their reward.

[^google-did-this]: This is the canonical example of black box optimization, and [Google actually did this once](https://static.googleusercontent.com/media/research.google.com/es//pubs/archive/46507.pdf).

More formally, black boxes are mappings {{< katex >}}r\colon\mathcal{X}\to\mathbb{R}{{< /katex >}} from a search space {{< katex >}}\mathcal{X}{{< /katex >}} (which may be fully continuous, discrete, or a combination of both) to a real number. What makes it a black box (say, compared to other types of functions) is that **we don't have access to anything beyond evaluations.** In other types of mathematical functions, we usually have them in a closed form.

Another idea that we usually attach to black boxes is that **evaluating them is expensive and cumbersome**. Cooking a batch of cookies might not sound like much but, in other applications, getting the feedback, reward or cost {{< katex >}}r(x_1, x_2, \dots, x_d){{< /katex >}} might take days, or involve using assets that are incredibly expensive. Think for example of a chemical reaction involving scarse and expensive solvents, or training a large Machine Learning model that takes days to train on sizeable compute.

<!-- [Optimizing black boxes] -->
The goal, then, is to **optimize** the black box. We want to find the ideal values for {{< katex >}}x_1, \dots, x_d{{< /katex >}} that maximize (or minimize) the signal {{< katex >}}r(x_1, \dots, x_d){{< /katex >}}. We want to find the recipe for the tastiest cookies. The lack of a closed form for our signal {{< katex >}}r{{< /katex >}} renders unavailable all the usual mathematical optimization techniques that are based on convexity, gradients, or Hessians, which means that we need to come up with alternatives that rely only on evaluating the function.

<!-- [Black boxes are everywhere] -->
Black box optimization is everywhere nowadays. This framework is generic enough to allow us to express several processes as black boxes to be optimized. There's plenty of contemporary work on this, with applications ranging in fields as diverse as self-driving labs, molecule and protein design for therapeutical purposes, hyperparameter tuning and automatic Machine Learning.

# Benchmarking in black box optimization

Just so we are all in the same page, let's define what a benchmark is. To me,
a benchmark is a collection of problems in which a given system or model is tested. The
performance of a model is gauged by a collection of metrics, for example
- classification accuracy in image recognition,
- number of correctly answered questions in a standardized test (e.g. the BAR) in language modelling,
- [prediction error with respect to the solutions of a differential equation](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf) in scientific ML.
- [prediction error on the properties of mutated proteins](https://proteingym.org/) in structural biology.

In the case of black box optimization, this collection of problems tends to be
a [family of synthetic functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization),
and the metric that we use to check whether we are doing well is e.g. how quickly we're finding the maximum/minimum
of the function (or how close we get to finding it), with a metric called **cummulative regret**, defined mathematically
as [TODO:ADD].

Although they have a closed form (which is slightly contradicting our definition of a black box), their form is
designed to mimic certain behaviors that we might expect from real black boxes out there: multiple local minima/maxima,
useless gradient information, needle-in-a-haystack-type problems... Here are some plots of how these functions look
like in the 2D case.

[Images of some of these].

We, as a community, have devoted a significant efforts on keeping track
of the performance of our optimizers on these synthetic benchmarks.
Check any paper on black box optimization, and you will always find regret plots
with performance on these.[^some-examples] There are even entire platforms dedicated
to measuring the performance of black box optimizers on these synthetic functions.
An example of such is Hansen et al.'s *COmparing Continuous Optimizers* ([COCO](https://coco-platform.org/)).

[^some-examples]: Check for example [this]() paper, or [this]() paper, or [this other]()
paper.

I would like to argue that **our efforts are better spent doing something different**.
The goal of these synthetic functions is, allegedly, to give us proxies for real-world
optimization tasks, but we don't know how the real world is until we face it. These
synthetic black box functions (as well as many other benchmarks in black box optimization
for chemistry or biology) fall short when describing the potential of an optimizer in a
given real-world task.

The current state of our benchmarking is such that, when a practitioner arrives with an
interesting problem to optimize, we don't have much to show, and we can't confidently
assess which algorithms would perform best in their problems. Best we can say is which
algorithms perform best on these highly synthetic, general set of examples.[^carolas-work]

[^carolas-work]: This is not entirely true. We can tell practitioners to use tools
for automatic selection of optimizers. There's at least plenty of research on making
dynamic tools for black box optimization, with plenty of progress on packages like
[nevergrad]() or [Ax](). If you're interested in this line of thinking, check
[Carola Doerr's]() work.

<!-- Admitedly, these synthetic functions are mainly used to detect whether the algorithm
works (i.e. they're used as a sanity check, and not as grounds for decision-making).
Some of these black boxes have specific behaviors (e.g. a single optima in a very
flat region, or several local optima with deceiving gradient information), which
also allow us to detect the strengths and weaknesses of our black box optimizers. -->

# Model metrology (or best practices for benchmarking)

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

# A personal example: Super Mario Bros and discrete sequence optimization

# Conclusion

