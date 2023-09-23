---
date: "2019-10-06"
title: "A Link to my MSc Thesis"
image: "/static/assets/thesis_img_bril.png"
---

The title of this post is the title of my M.Sc. thesis, which I defended on Friday 27th, 2019. In this thesis I explore a method deviced by [Niels Justesen](https://njustesen.com/), [Sebastian Risi](http://sebastianrisi.com/) and I, called **Behavioral Repertoires Imitation Learning**. This method extends Imitation Learning by adding *behavioral features* to state-action pairs. The policies that are trained with this method are able to express more than one *behavior* (e.g. bio-oriented, mech-oriented) on command.

In summary, the way this is done is by designing a *behavior space*, a subset of {{< katex >}}\mathbb{R}^M{{< /katex >}} that encodes the player's behavior in some way. After that, the dimensions are reduced in order to clusterize and understand the different behaviors present in the demonstrations. Finally, we expanded the state-action pairs gathered from the demonstrations with the coordinates in this low-dimensional space. This process was originally described in [this preprint on arXiv](https://arxiv.org/abs/1907.03046). The behavior space I tackled in my thesis was a little bit different.

![](/assets/another_thesis_img.png)

Last week, I pushed the final document of my thesis to the public repositories of the National University of Colombia. [Here's a link to it](https://repositorio.unal.edu.co/handle/unal/77095).