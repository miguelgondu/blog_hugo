---
date: "2020-12-15"
title: "Translating measure theory into probability"
image: "/static/assets/translation_table_longer.svg"
---

# Translating measure theory into probability

This post discusses how measure theory formalizes the operations that we usually do in probability (computing the probability of an event, getting the distributions of random variables...). With it, I hope to establish a translation table between abstract concepts (like measurable spaces and {{< katex >}}\sigma{{< /katex >}}-additive measures) and their respective notions in probability. I hope to make these concepts more tangible by using a running example: counting cars and the Poisson distribution. Feel free to skip to the end for a quick summary.

## A running example: counting cars

Let's say you and your friends want to start a drive-through business. In order to allocate work shifts or to predict your daily revenue, it would be great to estimate how many cars will go through your street in a fixed interval of time (say, Mondays between 2pm and 4pm). Let's call this amount {{< katex >}}X{{< /katex >}}. {{< katex >}}X{{< /katex >}} is what we call a **random variable**: there is a stochastic, non-deterministic experiment going on out there in the world, and we want to be able to say things about it. We want to be able to answer questions about **events** that relate to {{< katex >}}X{{< /katex >}}, questions like **how likely** is it that we will get more than 50 cars in said time interval.

Probability theory sets up a formalism for answering these questions, and it does so by using a branch of mathematics called measure theory. I'll use this running example to make the abstract concepts that we'll face more grounded and real.

## Formalizing events

In our scenario, we have several possible events (seeing between 10 and 20 cars, getting less than 5 customers...), and the theory that we will be establishing will allow us to say how _likely_ each one of these events are.

But first, what do we expect of events? If {{< katex >}}A{{< /katex >}} and {{< katex >}}B{{< /katex >}} are events, we would expect {{< katex >}}A \land B{{< /katex >}} and {{< katex >}}A \lor B{{< /katex >}} to be possible events (seeing 5 cars go by _and_/_or_ having 3 takeout orders), we would also expect {{< katex >}}\text{not}\, A{{< /katex >}} to be an event. Collections of sets that are closed under these three operations (intersection, union and complement) are called {{< katex >}}\sigma{{< /katex >}}-algebras:

**Definition:** A {{< katex >}}\sigma{{< /katex >}}-algebra over a set {{< katex >}}\Omega{{< /katex >}} is a collection of subsets {{< katex >}}\mathcal{F}{{< /katex >}} (called _events_) that satisfies:
1. Both {{< katex >}}\Omega{{< /katex >}} and the empty set {{< katex >}}\varnothing{{< /katex >}} are in {{< katex >}}\mathcal{F}{{< /katex >}}.
2. {{< katex >}}\mathcal{F}{{< /katex >}} is closed under (countable) unions, intersections and complements.

The pair {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} is called a **measurable space**.[^1]

You can think of {{< katex >}}\Omega{{< /katex >}} as the set containing all possible outcomes {{< katex >}}\omega{{< /katex >}} of your experiment. In the context of e.g. coin tossing, {{< katex >}}\Omega = \{\text{heads}, \text{tails}\}{{< /katex >}}, and in our example of cars going through your street, {{< katex >}}\Omega = \{\text{seeing no cars}, \text{seeing 1 car}, \dots\}{{< /katex >}}. We have two special events (that are always included, according to the definition): {{< katex >}}\Omega{{< /katex >}} and {{< katex >}}\varnothing{{< /katex >}}. You can think of them as tokens for absolute certainty and for impossibility, respectively.

There are two measurable spaces that I would like to discuss, both because they are examples of this definition, and because they come up when explaining some of the concepts that come next. The first one is the **discrete measureable space** (given by the naturals {{< katex >}}\mathbb{N}{{< /katex >}} and the {{< katex >}}\sigma{{< /katex >}}-algebra of all possible subsets {{< katex >}}\mathcal{P}(\mathbb{N})){{< /katex >}}, and the second one is the set of real numbers {{< katex >}}\mathbb{R}{{< /katex >}} with **Borel sets** {{< katex >}}\mathcal{B}{{< /katex >}}. You can think of the Borel {{< katex >}}\sigma{{< /katex >}}-algebra as the _smallest_ one that contains all open intervals {{< katex >}}(a,b){{< /katex >}}, their unions and intersections.[^2] The definition of {{< katex >}}\mathcal{B}{{< /katex >}} might seem arbitrary for now, but it will make more sense once we introduce random variables.

Notice that, in our running example, we are using the discrete measurable space (by idenfitying {{< katex >}}\text{seeing no cars} = 0{{< /katex >}}, {{< katex >}}\text{seeing 1 car} = 1{{< /katex >}} and so on).

## Formalizing probability

The probability that an event {{< katex >}}E\in\mathcal{F}{{< /katex >}} (for example {{< katex >}}E = \{\text{seeing between 10 and 20 cars}\}{{< /katex >}}) is formalized using the notion of a measure. Let's start with the definition:

**Definition:** A {{< katex >}}\sigma{{< /katex >}}-**additive measure** (or just measure) on a measureable space {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} is a function {{< katex >}}\mu:\mathcal{F}\to[0, \infty){{< /katex >}} such that, if {{< katex >}}\{A_i\}_{i=1}^\infty{{< /katex >}} is a family of pairwise disjoint sets, then
{{< katex display >}}\mu\left(\cup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mu(A_i).{{< /katex >}}

In summary, we expect measures to be non-negative and to treat disjoint sets additively. Notice how this generalizes the idea of, say, volume: if two solids {{< katex >}}A{{< /katex >}} and {{< katex >}}B{{< /katex >}} in {{< katex >}}\mathbb{R}^3{{< /katex >}} are disjoint, we would expect the volume of {{< katex >}}A\cup B{{< /katex >}} to be the sum of their volumes.

In the previous section we talked about two measurable spaces, let's discuss the usual measures they have:

- For {{< katex >}}(\mathbb{R}, \mathcal{B}){{< /katex >}}, we should be specifying the *length* of open intervals {{< katex >}}I = (a, b){{< /katex >}}. Our intuition says we should be defining
{{< katex display >}}\mu(I) = b - a.{{< /katex >}}
This measure can be extended to arbitrary sets in {{< katex >}}\mathcal{B}{{< /katex >}}, but how this is done is a story for another time.
- For {{< katex >}}(\mathbb{N}, \mathcal{P}(\mathbb{N}))){{< /katex >}}, we need to specify the *size* of any possible subset {{< katex >}}A{{< /katex >}} of {{< katex >}}\mathbb{N}{{< /katex >}}. The standard measure that we define in discrete measurable spaces (including finite ones) is **the counting measure**, literally given by counting how many elements are in {{< katex >}}A{{< /katex >}}:
{{< katex display >}}\mu(A) = \#(A) = \text{amount of elements in }A.{{< /katex >}}

There's a key difference between sizes and probabilities, though: **we assume probabilities to be bounded**[^3]. We expect events with probability 1 to be absolutely certain. Adding this restriction we get the following definition:

**Definition:** a **probability space** is a triplet {{< katex >}}(\Omega, \mathcal{F}, \text{Prob}){{< /katex >}} such that
- {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} is a measurable space.
- {{< katex >}}\text{Prob}{{< /katex >}} is a {{< katex >}}\sigma{{< /katex >}}-additive measure that satisfies {{< katex >}}\text{Prob}(\Omega) = 1{{< /katex >}}.[^4]

The measure {{< katex >}}\text{Prob}{{< /katex >}} in a probability space satisfies all three [Kolmogorov's axioms](https://plato.stanford.edu/entries/probability-interpret/#KolProCal), which attempt to formalize our intuitive notions of probability.

## Formalizing random variables

Now we have a probability measure {{< katex >}}\text{Prob}{{< /katex >}} that allows us to measure the probability of events {{< katex >}}E\in\mathcal{F}{{< /katex >}}. How can we link this to experiments that are running somewhere outside in the world?

The outcomes of experiments are measured using random variables. In our example, {{< katex >}}X{{< /katex >}} takes an outcome {{< katex >}}\omega \in \Omega{{< /katex >}} and associates it with a number {{< katex >}}X(\omega)\in\mathbb{R}{{< /katex >}}. We have other examples, like {{< katex >}}Y(\omega) ={{< /katex >}} the number of customers after seeing {{< katex >}}\omega{{< /katex >}} cars or {{< katex >}}Z(\omega) ={{< /katex >}} the revenue after seeing {{< katex >}}\omega{{< /katex >}} cars.

This association is formalized using measurable functions.

**Definition:** Let {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} and {{< katex >}}(\Theta, \mathcal{G}){{< /katex >}} be two measurable spaces. {{< katex >}}f\colon\Omega\to\Theta{{< /katex >}} is a **measurable function** if for all {{< katex >}}B\in \mathcal{G}{{< /katex >}}, {{< katex >}}f^{-1}(B) \in \mathcal{F}{{< /katex >}}. In other words, if the inverse image of a measurable set in {{< katex >}}(\Theta, \mathcal{G}){{< /katex >}} is a measurable set in {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}}.

This condition ({{< katex >}}f^{-1}(B)\in\mathcal{F}{{< /katex >}}) says that it makes sense to query for events of the type {{< katex >}}\{\omega\in\Omega\colon f(\omega) \in B\}{{< /katex >}}, since they will always be measurable sets (if {{< katex >}}B{{< /katex >}} is a measureable set in {{< katex >}}(\Theta, \mathcal{G}){{< /katex >}}).

As we were discussing, the output of a random variable is a real number. This means that random variables are a particular kind of measurable functions:

**Definition:** Let {{< katex >}}(\Theta, \mathcal{G}){{< /katex >}} be either the real numbers with Borel sets, or the discrete measurable space. A function {{< katex >}}X\colon(\Omega, \mathcal{F})\to(\Theta, \mathcal{G}){{< /katex >}} is a **random variable** if {{< katex >}}X^{-1}(B)\in\mathcal{F}{{< /katex >}} for measurable sets {{< katex >}}B\in\mathcal{G}{{< /katex >}}.

People usually make the distinction between *continuous* and *discrete* random variables (depending on whether {{< katex >}}\Theta{{< /katex >}} is {{< katex >}}\mathbb{R}{{< /katex >}} or {{< katex >}}\mathbb{N}{{< /katex >}}, respectively). Thankfully, measure theory allows us to treat these two cases using the same symbols, as we will see when we discuss integration later on.

I still think this definition isn't completely transparent, because "_Borel sets_" sounds abstract. Remember that Borel sets are just (unions or complements of) open intervals in {{< katex >}}\mathbb{R}{{< /katex >}}. Since the inverse image of a set is very well behaved with respect to unions, intersections and complements, then it suffices to consider {{< katex >}}B{{< /katex >}} to be an interval. An alternative (and maybe more transparent) definition of a random variable is usually given as

**Definition:** Let {{< katex >}}(\Theta, \mathcal{G}){{< /katex >}} be either the real numbers with Borel sets, or the discrete measurable space. A function {{< katex >}}X\colon(\Omega, \mathcal{F})\to(\Theta, \mathcal{G}){{< /katex >}} is a **random variable** if {{< katex >}}X^{-1}\left([-\infty, a)\right)\in\mathcal{F}{{< /katex >}} for all {{< katex >}}a\in\mathbb{R}{{< /katex >}}. That is, if the sets {{< katex >}}\{\omega\in\Omega\colon X(\omega) < a\}{{< /katex >}} are events for all {{< katex >}}a\in \mathbb{R}{{< /katex >}}.

Now we are talking! In summary, random variables are formalized as measurable functions: Functions because they associate outcomes {{< katex >}}\omega{{< /katex >}} with real numbers (or only natural numbers), and measurable because we want the sets {{< katex >}}X(\omega) < a{{< /katex >}} (e.g. seeing less than 10 cars) to have meaningful probability.

## The distribution of a random variable

Now let's talk about the distribution of a random variable. You might have heard things like "this variable is normally distributed" or "that variable follows the binomial distribution", and you might have run computations using this fact, and the densities of these distributions. This is all formalized like this:

**Definition:** Let {{< katex >}}(\Theta, \mathcal{G}, \mu){{< /katex >}} be either the real numbers with Borel sets, or the discrete measurable space. Any random variable {{< katex >}}X\colon(\Omega, \mathcal{F}, \text{Prob})\to(\Theta, \mathcal{G}, \mu){{< /katex >}} induces a probability measure {{< katex >}}P_X\colon\mathcal{G}\to[0,\infty){{< /katex >}} on {{< katex >}}\Theta{{< /katex >}} given by
{{< katex display >}}P_X(A) = \text{Prob}(X\in A) = \text{Prob}(X^{-1}(A)).{{< /katex >}}
{{< katex >}}P_X{{< /katex >}} is called the **distribution** of {{< katex >}}X{{< /katex >}}.

In other words, **the distribution of a random variable is a way of computing probabilities of events**. If we have an event like {{< katex >}}E = \{\text{seeing between 10 and 20 cars}\}{{< /katex >}}, we can compute its probability using our random variable:
{{< katex display >}}
\text{Prob}(E) = P_X(\{10, 11, \dots, 20\}),
{{< /katex >}}
and we already have plenty of probability distributions {{< katex >}}P_X{{< /katex >}} that model certain phenomena in the world. In the case of couting cars, people use the Poisson distribution. But how can we go from {{< katex >}}P_X{{< /katex >}} to an actual number?, for that we must rely on **integration** with respect to measures and **densities** of distributions.

But before going through with these two topics, I want to define a concept that we see frequently in probability. Using the distribution {{< katex >}}P_X\colon\mathcal{G}\to[0, \infty){{< /katex >}} we can define the **cumulative density function** as
{{< katex >}}
P_X(x) = P_X(\{w\colon X(w) \leq x\}) = \text{Prob}(X \leq x).
{{< /katex >}}

As you can see, we use the same symbol ({{< katex >}}P_X{{< /katex >}}) to refer to a completely different function (in this case from {{< katex >}}\Theta{{< /katex >}} to {{< katex >}}\mathbb{R}{{< /katex >}}). In many ways, this function _defines_ the distribution of {{< katex >}}X{{< /katex >}} and we will see why after discussing integration and densities.

## An interlude: integration

Remember that the Riemann integral measures the area below a curve given by an integrable function {{< katex >}}f\colon[a,b]\to\mathbb{R}{{< /katex >}}. This area is given by
{{< katex display >}}\int_a^bf(x)\mathrm{d}x = \lim_{n\to\infty}\sum_{i=0}^{n} f(c_i)(x_{i+1} - x_i),{{< /katex >}}
where we are partitioning the {{< katex >}}[a,b]{{< /katex >}} interval into {{< katex >}}n{{< /katex >}} segments of length {{< katex >}}x_{i+1} - x_i{{< /katex >}}, and selecting {{< katex >}}c_i\in[x_i, x_{i+1}]{{< /katex >}}.

Being handwavy[^5], there is a way of defining an integral with respect to arbitrary measures {{< katex >}}\mu{{< /katex >}}, and it looks like this

{{< katex display >}}\int_A f\mathrm{d}\mu \,\,``=" \sum_{E_i}f(c_i)\mu(E_i){{< /katex >}}

where {{< katex >}}c_i\in E_i{{< /katex >}} and {{< katex >}}A{{< /katex >}} is the disjoint union of all sets {{< katex >}}E_i{{< /katex >}}. Notice how this relates to the Riemann sums: we are measuring _abstract lengths_ by replacing {{< katex >}}x_{i+1} - x_i{{< /katex >}} with {{< katex >}}\mu(E_i){{< /katex >}}. This is called the **Lebesgue integration** of {{< katex >}}f{{< /katex >}}, and it extends Riemann integration beyond intervals and into more arbitrary measurable spaces, including probability spaces.

Let's discuss how this integral looks like when we integrate with respect to the measures that we had defined for {{< katex >}}(\mathbb{R}, \mathcal{B}){{< /katex >}} and {{< katex >}}(\mathbb{N}, \mathcal{P}(\mathbb{N})){{< /katex >}}:
- Notice that when using the _elementary measure_ {{< katex >}}\mu((a, b)) = b-a{{< /katex >}}, we end up computing a number that coincides with the Riemann integral. Properly defining the Lebesgue integral (and checking that it matches when a function {{< katex >}}f{{< /katex >}} is both Riemann and Lebesgue integrable) would require some work, so I leave it for future posts[^5].
- If we are in the discrete measurable space with the counting metric {{< katex >}}\mu = \#\colon\mathcal{P}(\mathbb{N})\to[0,\infty){{< /katex >}}, this integral takes a particular form: if {{< katex >}}A = \{a_1, \dots, a_n\}{{< /katex >}} be a subset of {{< katex >}}\mathbb{N}{{< /katex >}}, we can easily decompose it into the following pairwise-disjoint sets: {{< katex >}}E_1 = \{a_1\}{{< /katex >}}, {{< katex >}}E_2 = \{a_2\}{{< /katex >}} and so on... The resulting integral would look like
{{< katex display >}}\int_A f\mathrm{d}\mu = \sum_{i=1}^n f(a_i).{{< /katex >}}
This means that integrating with respect to the counting metric is just our everyday addition!

Circling back to random variables: If we have a random variable {{< katex >}}X{{< /katex >}} and its distribution {{< katex >}}P_X{{< /katex >}}, we can consider the integral of any[^6] real function {{< katex >}}f{{< /katex >}} over an event {{< katex >}}A\in\mathcal{B}{{< /katex >}} with respect {{< katex >}}P_X{{< /katex >}}:
{{< katex display >}}\int_A f\mathrm{d}P_X\,\,``=" \sum_{A_i}f(c_i)P_X(A_i),{{< /katex >}}
and if {{< katex >}}f \equiv 1{{< /katex >}}:
{{< katex display >}}\int_A \mathrm{d}P_X = P_X(A) = \text{Prob}(X\in A).{{< /katex >}}

## The density of a random variable

Let's summarize what we have discussed so far: we have defined events as sets {{< katex >}}E\subseteq \Omega{{< /katex >}} in a {{< katex >}}\sigma{{< /katex >}}-algebra {{< katex >}}\mathcal{F}{{< /katex >}}, we defined the probability of an event {{< katex >}}\text{Prob}{{< /katex >}} as a measure on {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} that satisfies being non-negative, normalized and {{< katex >}}\sigma{{< /katex >}}-additive.

We also considered random variables as measurable functions {{< katex >}}X\colon (\Omega, \mathcal{F}, \text{Prob})\to(\Theta, \mathcal{G}, \mu){{< /katex >}}, where {{< katex >}}\Theta{{< /katex >}} is either the real numbers with Borel sets and Lebesgue measure (for _continuous_ random variables) or the natural numbers with all subsets and the counting measure (for _discrete_ random variables). {{< katex >}}X{{< /katex >}} induces a measure on {{< katex >}}(\Theta, \mathcal{G}, \mu){{< /katex >}} given by {{< katex >}}P_X(A) = \text{Prob}(X\in A){{< /katex >}}. This measure is the distribution of {{< katex >}}X{{< /katex >}}, and it also defines the cumulative density function {{< katex >}}P_X(x) = \text{Prob}(X \leq x){{< /katex >}}.

We are still wondering how to compute {{< katex >}}P_X(A) = \text{Prob}(X\in A){{< /katex >}}, but we noticed that {{< katex >}}\int_A\mathrm{d}P_X = \text{Prob}(A){{< /katex >}}. The density of {{< katex >}}X{{< /katex >}} allows us to compute this integral:

**Definition:** Let {{< katex >}}(\Theta, \mathcal{G}, \mu){{< /katex >}} be either the real numbers with Borel sets, or the discrete measurable space. A function {{< katex >}}p_X:\Theta\to\mathbb{R}{{< /katex >}} that satisfies {{< katex >}}P_X(A) = \int_A \mathrm{d}P_X = \int_A p_X\mathrm{d}\mu{{< /katex >}} is called the **density** of {{< katex >}}X{{< /katex >}} with respect to {{< katex >}}\mu{{< /katex >}}. If {{< katex >}}\Omega{{< /katex >}} is the discrete measurable space, {{< katex >}}p_X{{< /katex >}} is usually called the **mass** of {{< katex >}}X{{< /katex >}}.

Let's see how this definition plays out in our particular example. We know that {{< katex >}}\Theta = \mathbb{N}{{< /katex >}} and that {{< katex >}}\mu{{< /katex >}} is the counting measure, and it is well known[^7] that our variable {{< katex >}}X{{< /katex >}} (cars that go by in an interval of time) follows the Poisson distribution. This means that[^8]
{{< katex display >}}p_X(x;\, \lambda) = \frac{e^{-\lambda}\lambda^x}{x!}{{< /katex >}}

{{< katex display >}}\text{Prob}(10 \leq X \leq 20) = \int_{10}^{20}\mathrm{d}P_X = \int_{10}^{20}p_X\mathrm{d}\mu = \sum_{x=10}^{20}p_X(x; \lambda) = \sum_{x=10}^{20}\frac{e^{-\lambda}\lambda^x}{x!},{{< /katex >}}
and this is a number that we can actually compute after specifying {{< katex >}}\lambda{{< /katex >}}. A good question is: how do we know the _actual_ {{< katex >}}\lambda{{< /katex >}} that makes this Poisson distribution describe the random process of cars going through our street? Estimating {{< katex >}}\lambda{{< /katex >}} from data is called **inference**, but that is a topic for another time.

One final note regarding densities: if the cumulative density function {{< katex >}}P_X(x) = \text{Prob}(X\leq x){{< /katex >}} is differentiable, you can re-construct the density by taking the derivative: {{< katex >}}p_X(x) = P_X'(x){{< /katex >}}.  Using the fundamental theorem of calculus, we realize that we can easily compute probabilities in intervals:
{{< katex display >}}\text{Prob}(a\leq X \leq b) = \int_a^b\mathrm{d}P_X = P_X(b) - P_X(a){{< /katex >}}

## Conclusion: A translation table

In this post we discussed how some concepts from probability theory are formalized using measure theory, ending up with this translation table:

| Probability  | Measure |
|-------------|---------|
|Event|{{< katex >}}E\in\mathcal{F}{{< /katex >}}, where {{< katex >}}(\Omega, \mathcal{F}){{< /katex >}} is a measurable space.|
|Probability|A measure {{< katex >}}\text{Prob}\colon \mathcal{F}\to[0,\infty){{< /katex >}} that satisfies {{< katex >}}\text{Prob}(\Omega) = 1{{< /katex >}}. |
| Random variable | A measurable function {{< katex >}}X\colon(\Omega, \mathcal{F})\to (\Theta, \mathcal{G}){{< /katex >}}, where {{< katex >}}\Theta{{< /katex >}} is either {{< katex >}}\mathbb{R}{{< /katex >}} or {{< katex >}}\mathbb{N}{{< /katex >}}. |
|Distribution| The measure {{< katex >}}P_X(A) = \text{Prob}(X\in A) = \text{Prob}(X^{-1}(A)){{< /katex >}}|
|Cumulative density| The induced function {{< katex >}}P_X\colon\Theta\to\mathbb{R}{{< /katex >}} given by {{< katex >}}P_X(x) = \text{Prob}(X\leq x){{< /katex >}}.|
| Density (or mass)| A function {{< katex >}}p_X\colon\Theta\to\mathbb{R}{{< /katex >}} that satisfies {{< katex >}}P_X(A) = \int_Ap_X\mathrm{d}x{{< /katex >}}.|

## References
- [Terry Tao's notes on Measure Theory.](https://terrytao.files.wordpress.com/2011/01/measure-book1.pdf)
- [These notes on Measure Theory and Probability from Alexander Grigoryan.](https://www.math.uni-bielefeld.de/~grigor/mwlect.pdf)
- [Stanford Encyclopedia of Philosophy's entry on the philosophy of probability.](https://plato.stanford.edu/entries/probability-interpret/)

[^1]:This definition is not standard. It is enough to say that {{< katex >}}\mathcal{F}{{< /katex >}} contains {{< katex >}}\Omega{{< /katex >}}, that it is closed under countable unions and complements.
[^4]:Notice that the {{< katex >}}\sigma{{< /katex >}}-additivity implies that {{< katex >}}\text{Prob}(\Omega) = \text{Prob}(\Omega \cup \varnothing) = \text{Prob}(\Omega) + \text{Prob}(\varnothing){{< /katex >}}, which means that {{< katex >}}\text{Prob}(\varnothing) = 0{{< /katex >}}.
[^5]: If you want a formal treatment of integration in Measure Theory, check the references.
[^3]: Not really, there are formalizations of probability that ditch the _normalization_ axiom. There are also formalizations that skip {{< katex >}}\sigma{{< /katex >}}-additivity or positivity. These, however, belong more to the realm of philosophy of probability than what mathematicians and statisticians use in their daily practice. [The SEP has a great entry on it, in case you want to read more](https://plato.stanford.edu/entries/probability-interpret/).
[^2]: The technical definition is that the Borel sets are the {{< katex >}}\sigma{{< /katex >}}-algebra generated by the standard topology in {{< katex >}}\mathbb{R}{{< /katex >}}. You don't have to worry what any of those words mean for now, but you can think of a topology on a set as defining what the "open intervals" should look like. The fact that Borel sets are defined this way allows for all continuous functions on {{< katex >}}\mathbb{R}{{< /katex >}} to be measurable. See more in the references.
[^6]: This _any_ is a stretch. The functions have to be integrable with respect to {{< katex >}}P_X{{< /katex >}} (i.e. {{< katex >}}\int_Ef\mathrm{d}P_X{{< /katex >}} should be a finite number).
[^7]: The Poisson distribution can be thought of as a limit of the Binomal. But there's also a different way to derive the fact that said {{< katex >}}p_X{{< /katex >}} relates to counting things in fixed intervals of time. [Check this](https://courses.washington.edu/phys433/muon_counting/counting_stats_tutorial_b.pdf).
[^8]: The integral {{< katex >}}\int_A\mathrm{d}P_X{{< /katex >}} is transformed into a sum because {{< katex >}}\mu{{< /katex >}} is the counting measure in {{< katex >}}\mathbb{N}{{< /katex >}}. Remember that measure theory allows us to treat probabilities of events in the continuous and discrete setting using the same symbol.

