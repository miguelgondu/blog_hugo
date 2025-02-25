---
date: "2025-02-16"
title: "The future of black box optimization benchmarking is procedural"
description: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
summary: Recent work by Saxon et al. highlights the need for dynamic benchmarks, and I think procedural content generators might provide an answer.
---

This blogpost discusses 
a recent position paper by Saxon et al. called [*Benchmarks as Microscopes: a Call for Model Metrology*.](https://openreview.net/forum?id=bttKwCZDkm&noteId=Yfwy2d4fiT)
The authors argue that the benchmarks we construct ought to be constrained, dynamic, and plug-and-play.
I'll dive deeper into what each of these words mean,
and I'll argue that researchers in Procedural Content Generation (PCG) are in an ideal position to
construct such benchmarks.

Saxon et al.'s paper focuses on Natural Language Processing and language models,
but I think their arguments translate well to my domain: black box optimization.

With this blogpost I try to bridge the fields of black box benchmarking and PCG, encouraging PCG researchers
to take a look into benchmark construction for black box optimization and other fields (becoming what Saxon et al.
call *model metrologists*).

Since the goal is to bring people from different fields together, I won't assume you have knowledge on black box
optimization nor PCG. If you're already familiarized with these ideas, feel free to skip the sections that
introduce said topics.

# The name of the game: black box optimization

<!-- [The definition of a black box] -->
Think of a black box function as any process that transforms a set of inputs {{< katex >}}x_1, x_2, \dots, x_d{{< /katex >}} into a real number {{< katex >}}r = r(x_1, \dots, x_d){{< /katex >}}.

This definition is vague on purpose, and allows us to define several things as black box functions. Two examples: when you log into Instagram, the app that you are shown is "parametrized" by several {{< katex >}}x_i{{< /katex >}} variables such as, for example, {{< katex >}}x_1{{< /katex >}} could be whether your reels are shown in a loop or just once, {{< katex >}}x_2{{< /katex >}} could be the person you first see on your timeline, and {{< katex >}}x_3{{< /katex >}} could be how often you're shown an ad.[^this-is-just-a-hypothesis] One example of a metric {{< katex >}}r(x_1, x_2, x_3){{< /katex >}} would be how long you spend on the app. 
Another less macabre example is making a recipe for cookies, with the different {{< katex >}}x_i{{< /katex >}}s being ingredient amounts, and the result being a combination of how tasty it is compared to the original recipe.[^google-did-this]

[^this-is-just-a-hypothesis]: These are just examples, of course.

[^google-did-this]: This is the canonical example of black box optimization, and [Google actually did this once](https://static.googleusercontent.com/media/research.google.com/es//pubs/archive/46507.pdf).

More formally, black boxes are mappings {{< katex >}}r\colon\mathcal{X}\to\mathbb{R}{{< /katex >}} from a search space {{< katex >}}\mathcal{X}{{< /katex >}} (which may be fully continuous, discrete, or a combination of both) to a real number. What makes it a black box (say, compared to other types of functions) is that **we don't have access to anything beyond evaluations.** In other types of mathematical functions, we usually have them in a closed form.

Another idea that we usually attach to black boxes is that **evaluating them is expensive and cumbersome**. Cooking a batch of cookies might not sound like much but, in other applications, getting the feedback, reward or cost {{< katex >}}r(x_1, x_2, \dots, x_d){{< /katex >}} might take days, or involve using assets that are incredibly expensive. Think for example of a chemical reaction involving scarse and expensive solvents, or training a large Machine Learning model for several days on sizeable compute.

<!-- [Optimizing black boxes] -->
The goal, then, is to **optimize** the black box. We want to find the values of {{< katex >}}x_1, \dots, x_d{{< /katex >}} that maximize (or minimize) the signal {{< katex >}}r(x_1, \dots, x_d){{< /katex >}}. We want to find the recipe for the tastiest cookies, and Meta wants to keep us on Instagram as long as possible. The lack of a closed form for our signal {{< katex >}}r{{< /katex >}} renders unavailable all the usual mathematical optimization techniques that are based on convexity, gradients, or Hessians, which means that we need to come up with alternatives that rely only on evaluating the function.[^direct-optimization-is-another-name]

[^direct-optimization-is-another-name]: In [Algorithms for optimization](https://algorithmsbook.com/optimization/) by Kochenderfer and Wheeler, this type of optimization is called _direct optimization_. It has a pretty long history. Check Chap. 7 in their book for a longer introduction on the classics of the field.

<!-- [Black boxes are everywhere] -->
Black box optimization is everywhere nowadays. This framework is generic enough to allow us to express several processes as black boxes to be optimized. There's plenty of contemporary work on this, with applications ranging in fields as diverse as self-driving labs, molecule and protein design for therapeutical purposes, hyperparameter tuning and automatic Machine Learning.

# Benchmarking in black box optimization

Just so we are all in the same page, let's define what a benchmark is. To me,
a benchmark is a collection of problems in which a given system or model is tested. The
performance of a model is gauged by a collection of metrics, for example
- classification accuracy in image recognition,
- number of correctly answered questions in a standardized test (e.g. the BAR) in language modelling,
- [prediction error with respect to the solutions of a differential equation](https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf) in scientific ML, and
- [prediction error on the properties of mutated proteins](https://proteingym.org/) in structural biology.

In the case of black box optimization, it's common to start benchmarking on
a [family of synthetic functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization),
and the metric that we use to check whether we are doing well is e.g. how quickly we're finding the maximum/minimum
of the function (or how close we get to finding it), with a metric called **regret**, defined mathematically
as the distance to the optimal value.

Although they have a closed form (which is slightly contradicting our definition of a black box), their form is
designed to mimic certain behaviors that we might expect from real black boxes out there: multiple local minima/maxima,
useless gradient information, needle-in-a-haystack-type problems... As an example of this, here are the plots of how Rastrigin and Lévi (two common examples) look
in the 2D case: Notice how Rastrigin has several local minima, meaning that derivative-based methods would easily get stuck on local minima, and also notice how one of the dimensions of Lévi (Y) isn't as important as the other (X). These are properties we may also expect of real-world black boxes.

{{< figure src="/static/assets/pbg_blogpost/optimization_test_functions.png" alt="Two synthetic benchmarks: Rastrigin and Lévi" class="largeSize" title="Two commonly-used synthetic black boxes for continuous optimization." >}}

We, as a community, have devoted a significant efforts on keeping track
of the performance of our optimizers on these synthetic benchmarks.
Check any paper on black box optimization, and you will always find regret plots
with performance on these.[^some-examples] There are even entire platforms dedicated
to measuring the performance of black box optimizers on these synthetic functions.
An example of such is Hansen et al.'s *COmparing Continuous Optimizers* ([COCO](https://coco-platform.org/)).

[^some-examples]: Check for example [Fig 2. in the SAASBO](https://proceedings.mlr.press/v161/eriksson21a/eriksson21a.pdf) paper, or [Fig. 5 in the Vanilla BO](https://arxiv.org/pdf/2402.02229) paper.

I would like to argue that **our efforts are better spent doing something different**.
The goal of these synthetic functions is, allegedly, to give us proxies for real-world
optimization tasks, but we don't know how the real world is until we face it. These
synthetic black box functions (as well as many other benchmarks in black box optimization
for chemistry or biology) fall short when describing the potential of an optimizer in a
given real-world task.

The current state of our benchmarking is such that, when a practitioner arrives with an
interesting problem to optimize, we don't have much to show, and we can't confidently
assess which algorithms would perform best in their problems. Best we can say is which
algorithms perform well on highly synthetic, general sets of examples.[^carolas-work]

[^carolas-work]: This is not entirely true. On one hand, there are several data-driven proxies for tasks in biology, chemistry, material science... On the other, we can tell practitioners to use tools
for automatic selection of optimizers. There's at least plenty of research on making
dynamic tools for black box optimization, with plenty of progress on packages like
[nevergrad](https://facebookresearch.github.io/nevergrad/) or [Ax](https://ax.dev/). If you're interested in this line of thinking, check
[Carola Doerr's](https://scholar.google.com/citations?user=CU-V1sEAAAAJ&hl=es) current and future work.

# Model metrology (or best practices for benchmarking)

Saxon et al. gave me the language to formulate exactly what we are doing wrong: these
benchmarks are *too general*, and we should be constructing specialized, **constrained**
problems to test our optimizers, focusing on what a given practitioner needs. Moreover,
our benchmarks are *static* (i.e. they're the same over time), and thus we run the risk
of overfitting to them.[^silly-optimization] We need **dynamic** benchmarks. One aspect that we have nailed,
though, and that we should keep in future iterations is that our benchmarks are easily
deployable, or **plug-and-play**.

[^silly-optimization]: I've heard from two researchers in the field that one optimizer was performing surprisingly well in several benchmarks, optimizing them instantly... it turned out that the optimizer started by proposing the origin point {{< katex >}}(0, 0, \dots, 0){{< /katex >}}, which is coincidentally the optima location for several of these black boxes. 

These are the three keywords they claim make good benchmarks. Good benchmarks are

- **constrained**, bounded and highly specific to a given application,
- **dynamic** to avoid overfitting to certain behaviors, and
- **plug-and-play**, easy to deploy.

The authors call for a new research field, entirely devoted to evaluating model performance,
called *model metrology*. A model metrologist is someone who has the tools to create such
constrained, dynamic & plug-and-play benchmarks, tailored to the needs of a given practitioner.

# Procedural Content Generation...

...stands for the use of algorithms to generate content. It has plenty of use in video games,
where PCG allows developers to generate assets for their games (from the clothes a character wears,
to the entire game map). Several block-buster games use PCG as a core mechanic. For example,
the whole world one explores inside Minecraft is procedurally generated from a random seed; another example
is No Man's Sky, where an entire universe is created procedurally using an algorithm that depends only
on the player's position.

{{< figure src="/static/assets/pbg_blogpost/no-mans-sky.png" alt="A press-release image of No Man's Sky." class="largeSize" title="An image from No Man's Sky, taken from their press release. A whole universe made procedurally." >}}

PCG is also a research field, in which scholars taxonomize, study and develop novel techniques for
generating content.[^read-more-in-the-pcg-book] Some of them involve searching over a certain representation of game content (e.g. describing a dungeon game as a graph of connected rooms, and searching over all possible graphs of a certain size), other involve exploring formal grammars that define game logic/assets, and most of them use randomness to create content. In Minecraft, the whole world is made using random number generators.

[^read-more-in-the-pcg-book]: If you want to learn more, check the [PCG book](https://www.pcgbook.com/) by several researchers in the field.

**Researchers in PCG are in an ideal position to create constrained, dynamic & plug-and-play benchmarks.**
If we can formulate benchmarks as pieces of content, we could leverage plenty of research in the
PCG community. They already have a language for developing _content_, measuring e.g. the
reliability, diversity and controlability of their developments. We just need to convince them to
work on building synthetic benchmarks for us according to the specific needs of a set of practitioners.

# Towards procedural benchmark generation

We could think of creating a novel subfield of model metrology: *procedural benchmark generation*.
PCG researchers could start by establishing contact with practitioners in black-box optimization and,
together with the model metrologists of said domain, they could establish the requirements that
the developed benchmarks would need to meet.

Afterwards, it's all about procedural generation: cleverly combining algorithms that rely on randomness (or grammars,
or useful representations followed by search, or any other PCG technique) to create a procedurally generated benchmark.
This process of generating benchmarks could then be evaluated using all the language we have for content: is it diverse enough?
Can we control it? Does it express the desired behaviors? Are the resulting benchmarks a believable proxy of the actual task?
Is generation fast?

Let me explain further what I mean by this with a recent example. It shows how procedural generation
can be applied to constructing black boxes that are relevant for specific domains.

# Example: closed-form test functions in structural biology

Let me introduce you to the wonderful world of protein engineering.[^talking-about-ab-initio] The set-up goes like this: one starts
with a **discrete sequence** of amino acids {{< katex >}}(a_1, \dots, a_L){{< /katex >}} called the
**wildtype**, and the goal is to find slight deviations from this wildtype such that a signal {{< katex >}}r{{< /katex >}}
(for example thermal stability, or whether the protein binds well to a certain receptor in the body) is optimized.

[^talking-about-ab-initio]: If you're a biologist, you'll cringe at my description.
I'm talking here about *ab initio* protein engineering, where one starts from a wildtype. There's
also *de novo* protein design, where one creates sequences of amino acids "from nothing". An example
of *de novo* design is [the Chroma paper](https://www.nature.com/articles/s41586-023-06728-8).

{{< figure src="/static/assets/pbg_blogpost/DNJA1.png" alt="DNJA1 visualized by Protein Viewer." class="midSize" title="An example of a protein (a tiny bit of DNJA1), which is a sequence of amino acids: Thr-Thr-Tyr-Tyr-Asp-... Each of these can be mutated with the aim of maximizing thermal stability or any other biophysical or medicinal property." >}}

Nowadays, computing such signals is not straightforward. The state-of-the-art at time of writing is using huge Machine Learning
models to get estimates of binding scores (AlphaFold 2, as an example), or using physical simulators. These are not easy to set-up,
and much less query: one needs to install licensed software, have decent amounts of compute, and have the patience to wait for more than
a minute per black box call.[^that's-why-we-developed-poli] In the language of Saxon et al., **these black boxes are not plug-and-play**.

[^that's-why-we-developed-poli]: And that's why, in my previous job, we tried to develop a Python framework for democratizing
access to some of these black boxes. It's called [poli](https://github.com/MachineLearningLifeScience/poli).

In 2024, Stanton et al. published a paper called [*Closed-Form Test Functions for Biophysical Sequence Optimization Algorithms*](https://arxiv.org/abs/2407.00236).
In it, they identify this lack of plug-and-play black boxes in protein design, and propose a black box that mimics the
relevant behavior of the signals {{< katex >}}r{{< /katex>}} described above, while being trivial to query.

The authors set out to create a black box that is
1. Evaluated on discrete sequences and their mutations, which may render feasible or unfeasible sequences.
2. Difficult enough to optimize, mimicking some second-order interactions (*epistasis*) that are usually present in protein design.
3. Instant to query.

To do so, the authors propose *Ehrlich functions*, a procedurally generated black box based on a Markov Decision Process (MDP)
to generate a "feasibility distribution" over the space of amino acids, and a set of motifs that needs to be satisfied in the
sequence. Both the MDP and motifs are constructed at random.

When I first read this paper, I felt as if I was reading a PCG article: The authors propose a _procedurally generated_
black box, which relies on randomness to create a set of conditions that need to be satisfied to _score_ a discrete sequence.
Almost as if they were creating a ruleset for a _game_. Funnily, some of their parameters (the quantization, for example) can
be understood as _increasing the difficulty of the game_.

This, to me, is a first example of how procedural benchmark generation could look like. We could then investigate
whether Ehrlich functions are indeed representative of the issues of protein sequence design, and whether
Ehrlich functions are diverse, controllable, etc.

# Conclusion

Saxon et al. call attention to the fact that, quote, "Benchmarks can aim for generality—or they can be valid and useful".
I agree. And even though they raise their point in the context of evaluating language models, the lessons translate well
to black box optimization (where we have been devoting significant resources to optimizing synthetic benchmarks).

I argue that benchmarks and black boxes can be thought of as a form of content, and that we could in the future leverage
all the language that has been developed for Procedural Content Generation in the context of video games. In other words,
PCG researchers could be great *model metrologists* and benchmark developers. 
A recent example can be found in biology, where a procedurally generated black box is a useful replacement for expensive
simulators that are difficult to set-up.
