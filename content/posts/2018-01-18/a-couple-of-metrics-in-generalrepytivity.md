---
author: "Miguel González Duque"
date: 2018-01-18
title: "A couple of metrics in generalrepytivity"
---

# A couple of metrics in generalrepytivity

As a small tutorial on [generalrepytivity](https://github.com/miguelgondu/generalrepytivity)[^1], this blogpost explains
how to create a metric tensor in generalrepytivity and, with it,
how to get all the usual geometric invariants one is interested in
(such as the Riemann tensor, the Ricci tensor and the scalar 
curvature). You can find all the code in this article in this
[jupyter notebook](https://gist.github.com/miguelgondu/2fb6d97291a1478d8522ad7b9410748c).

## Installing the library

generalrepytivity is now on [pypi](https://pypi.python.org/pypi?name=generalrepytivity&version=0.1.0&:action=display), so installing it is easy using `pip`:

```bash
pip install generalrepytivity
```

## Importing the library

We heavily recommend importing it along with [sympy](http://www.sympy.org/en/index.html)
(the symbolic computation library of python):

```python
import generalrepytivity as gr
import sympy
```

Tensors come with a fancy LaTeX printing function, so if 
you're working on a [jupyter notebook](http://jupyter.org/) I recommend setting
the printing function of sympy to use latex:

```python
sympy.init_printing(use_latex=True)
```

## Creating a tensor in *generalrepytivity*

To create a Tensor object, one must specify three things:

- **coordinates**, which are a list of sympy symbols.
- **_type**, a tuple of integers `(p,q)`.
- **values**, a dictionary of the non-zero values (whose keys
  are tuples of multiindices).

For example, say you want to create the following tensor in {{< katex >}}\mathbb{R}^4{{< /katex >}} with coordinates {{< katex >}}t, x, y, z{{< /katex >}}:

{{< katex display >}} \Gamma = (x^2 + y^2)\frac{\partial}{\partial t}\otimes \frac{\partial}{\partial x}\otimes dx\otimes dz + \sin(t)\frac{\partial}{\partial x}\otimes \frac{\partial}{\partial y}\otimes dt \otimes dt {{< /katex >}}

in this case, we would need the sympy symbols `t, x, y, z` for the **coordinates**,
the **_type** would be (2,2) (because {{< katex >}}\Gamma{{< /katex >}} is a (2,2)-tensor) and for the **values**
note that the **components** of {{< katex >}}\Gamma{{< /katex >}} in these coordinates are the following:

{{< katex display >}}\Gamma^{0, 1}_{1, 3} = x^2 + y^2{{< /katex >}}

{{< katex display >}}\Gamma^{1, 2}_{0, 0} = \sin(t){{< /katex >}}

so the dictionary of **values** should look like this:

```python
values = {
    ((0, 1), (1, 3)): x**2 + y**2,
    ((1, 2), (0, 0)): sympy.sin(t)
}
```

In summary:

```python
t, x, y, z = sympy.symbols('t x y z')
values = {
    ((0,1), (1, 3)): x**2 + y**2,
    ((1,2), (0, 0)): sympy.sin(t)
}

Gamma = gr.Tensor([t, x, y, z], (2,2), values)
```

and you can access the components of the tensor by indexing:
```python
Gamma[(0,1), (1,3)] == x**2 + y**2
True 
```

## Metrics

A metric {{< katex >}}g{{< /katex >}} is just a (0,2)-tensor, so we could create it using a values dict
just like we did above but, because (0,2)-tensors are so important (and because
they can be represented by matrices), *generalrepytivity* has a simpler way of
creating them. The function to use is `gr.get_tensor_from_matrix`, which has the
following syntax:

```python
gr.get_tensor_from_matrix(matrix, coordinates)
```

where `matrix` is a square sympy matrix and the `coordinates` are just like above
(i.e. a list of sympy symbols).

### Examples of metrics

In this blogpost we will deal with three different solutions to Einstein's
equations: Gödel's metric, Schwarzschild's metric and FRLW.

#### Gödel's metric

In 1949, Gödel proposed a solution to Einstein's Field Equations which described
a rotating universe in which closed world-lines are possible (that is, you could
in some way influence the past). Because of this violation of causality, Gödel's metric
is just interesting from a theoretical perspective, not from a modeling one.

Gödel's metric is

{{< katex display >}}g = a^2(dx_0\otimes dx_0 - dx_1\otimes dx_1 + (e^{2x_1}/2)dx_2\otimes dx_2 - dx_3\otimes dx_3 + e^{x_1}dx_0\otimes dx_2 + e^{x_1}dx_2 \otimes dx_0) {{< /katex >}}

To enter it into generalrepytivity, we could use the matrix representation:

```python
x0, x1, x2, x3 = sympy.symbols('x_0 x_1 x_2 x_3')
a = sympy.Symbol('a')
A = a**2 * sympy.Matrix([
    [1, 0, sympy.exp(x1), 0],
    [0, -1, 0, 0],
    [sympy.exp(x1), 0, sympy.exp(2*x1)/2, 0],
    [0,0,0,-1]])

g_godel = gr.get_tensor_from_matrix(A, [x0, x1, x2, x3])
```

#### Schwarzschild metric

Schwarzschild metric is generally used when it comes to modeling spherical objects
and their influence in the geometry of spacetime. It comes after assuming a static
and spherically symmetrical spacetime.

Schwarzschild metric is (in units such that {{< katex >}}c=1{{< /katex >}} and {{< katex >}}G_N = 1{{< /katex >}})

{{< katex display >}}g =  -\left(1 - \frac{2 m}{r}\right)dt \otimes dt + \left(1 - \frac{2 m}{r}\right)^{-1}dr \otimes dr + (r^{2} \sin^{2}{\left (\theta \right )})d\phi \otimes d\phi + r^{2}d\theta \otimes d\theta {{< /katex >}}

where {{< katex >}}m{{< /katex >}} is the mass of the spherical object. To review the canon way of creating 
tensors, lets use a values dictionary:

```python
t, r, theta, phi = sympy.symbols('t r \\theta \\phi')
m = sympy.Symbol('m')
values = {
    ((), (0,0)): -(1-2*m/r),
    ((), (1,1)): 1/(1-2*m/r),
    ((), (2,2)): r**2,
    ((), (3,3)): r**2*sympy.sin(theta)**2
}
g_sch = gr.Tensor([t, r, theta, phi], (0, 2), values)
```

#### FRLW

The FRLW metric (which stands for Friedmann-Lemaître-Robertson-Walker) models
a homogenous, isotropic and either expanding or contracting (depending on a 
constant) universe.

Its tensor representation in coordinates is

{{< katex display >}}g = -dx_0\otimes dx_0 + A(x_0)^2(dx_1\otimes dx_1 + dx_2\otimes dx_2 + dx_3\otimes dx_3){{< /katex >}}

where {{< katex >}}A(x_0) = \epsilon x_0^q{{< /katex >}} with {{< katex >}}0<q <1{{< /katex >}} and {{< katex >}}\epsilon\in\mathbb{R}{{< /katex >}}.
Using matrices:

```python
x0, x1, x2, x3 = sympy.symbols('x_0 x_1 x_2 x_3')
epsilon, q = sympy.symbols('\\epsilon q')
A = epsilon * (x0 ** q)
g_matrix = sympy.diag(-1, A**2, A**2, A**2)
g_FRLW = gr.get_tensor_from_matrix(g_matrix, [x0, x1, x2, x3])
```

## The Spacetime object

A good way of making a summary of the usual geometric invariants one computes using
a metric is by using the `Spacetime` object in *generalrepytivity*. It takes just
one argument: the metric.

```python
Godel_spacetime = gr.Spacetime(g_godel)
Sch_spacetime = gr.Spacetime(g_sch)
FRLW_spacetime = gr.Spacetime(g_FRLW)
```

Each object now holds:
* The Christoffel symbols `christoffel_symbols`.
* The Riemann tensor `Riem`.
* The Ricci tensor `Ric`.
* The scalar curvature `R`.

For example, Schwarzschild's metric was created in order to be *Ricci flat* (i.e.
{{< katex >}}\text{Ric} = 0{{< /katex >}}). Let's verify this:

```python
Sch_spacetime.Ric == 0
True
```

### Printing a summary in LaTeX

Lastly, the `Spacetime` object comes with a print-to-file function which accepts
two format options: tex or txt. Running the command

```python
Godel_spacetime.print_summary('godel.tex', _format='tex')
```

generates a .tex file with a summary of all the values of the `christoffel_symbols`,
`Riem`, `Ric` and `R`.

---

[^1]: This is the first of a series of posts about [generalrepytivity](https://github.com/miguelgondu/generalrepytivity), a python toolbox I made for computations related to tensors and general relativity.
