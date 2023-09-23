---
title:  "Latin Squares and Finite Groups"
date:   "2018-01-24"
---

Last semester, for an algebra homework, I was trying to prove that there exist only 2 groups of order 6 (namely {{< katex >}}\mathbb{Z}_6{{< /katex >}} and {{< katex >}}S_3{{< /katex >}}). The usual argument uses the classification of groups with order {{< katex >}}pq{{< /katex >}} (with {{< katex >}}p{{< /katex >}} and {{< katex >}}q{{< /katex >}} prime), which itself uses Sylow theorems, but I wondered if I could prove it computationally. Here's my attempt.

**Definition:** a **magma** is a pair {{< katex >}} (G,\ast) {{< /katex >}} where {{< katex >}}G{{< /katex >}} is a non-empty set and {{< katex >}}\ast{{< /katex >}} is a binary operation on {{< katex >}}G{{< /katex >}}. A **quasigroup** is a magma in which the equations {{< katex >}}gx=h{{< /katex >}} and {{< katex >}}yg=h{{< /katex >}}, with {{< katex >}}g,h\in G{{< /katex >}}, always have a unique solution for {{< katex >}}x{{< /katex >}} and {{< katex >}}y{{< /katex >}} in {{< katex >}}G{{< /katex >}}. A **group** is a magma in which {{< katex >}}\ast{{< /katex >}} is associative, modulative and invertive.

Another way of defining a quasigroup is as a magma in which the left and right cancellation laws hold. Note that we're not asking {{< katex >}}\ast{{< /katex >}} to be associative in a magma (let alone in a quasigroup).

All there is to know about a finite magma {{< katex >}}(G,\ast){{< /katex >}} is encoded in its **Cayley table**: a matrix that shows how the elements in {{< katex >}}M{{< /katex >}} operate among themselves. In general, if {{< katex >}}G = \{a_1, \dots, a_n\}{{< /katex >}}, the Cayley table looks like this:

$\begin{matrix}
\ast&a_1&\dots &a_n\\
\hline
a_1&a_1a_1&\dots &a_1a_n\\
\vdots&\vdots&\dots&\vdots\\
a_n&a_na_1&\dots &a_na_n\\
\end{matrix}$ 

That is, the element in the {{< katex >}}(i,j){{< /katex >}}-th position is the result of multiplying {{< katex >}}a_i{{< /katex >}} with {{< katex >}}a_j{{< /katex >}} in that order.

You can tell a lot about a magma and its operation just by looking at its Cayley table. For example, consider **Klein's 4 group** {{< katex >}}V{{< /katex >}}:

$\begin{array}{c|cccc}
*&e&a&b&c\\
\hline
e&e&a&b&c\\
a&a&e&c&b\\
b&b&c&e&a\\
c&c&b&a&e\\
\end{array}$

You can immediately tell its commutative (because the matrix is symmetric), you can tell there is an identity (namely {{< katex >}}e{{< /katex >}}) and that each element is its own inverse. You could also tell if the binary operation is associative from its Cayley table using [Light's test](https://en.wikipedia.org/wiki/Light%27s_associativity_test), although it isn't any better, computationally speaking, that just verifying every case by hand.

We will use Cayley tables as the bridge between algebra and combinatorics.

## Latin squares and quasigroups

**Definition:** a {{< katex >}}n\times n{{< /katex >}} matrix of {{< katex >}}n{{< /katex >}} different entries is called a **latin square** if no element appears more than once in any row or column. This property is called the **latin square property**.

We will deal with latin squares of size {{< katex >}}n{{< /katex >}} whose entries are the integers from {{< katex >}}0{{< /katex >}} to {{< katex >}}n-1{{< /katex >}}. For example

$\begin{bmatrix}
0&1&2\\
1&2&0\\
2&0&1\\
\end{bmatrix}$

is a latin square of size {{< katex >}}3{{< /katex >}}. We will also start indexing by 0.

**Theorem 1:** if {{< katex >}}(G, \ast){{< /katex >}} is a quasigroup, then its Cayley table is a latin square.

**Proof:** Suppose {{< katex >}}G = \{a_1, \dots, a_n\}{{< /katex >}} and that an element {{< katex >}}b\in G{{< /katex >}} appears twice in row {{< katex >}}l{{< /katex >}} (say, in columns {{< katex >}}j{{< /katex >}} and {{< katex >}}k{{< /katex >}}), by the definition of the Cayley table, this means that

{{< katex display >}} a_la_j = b = a_la_k {{< /katex >}}

and because {{< katex >}}G{{< /katex >}} is a quasigroup, the left cancellation law implies that {{< katex >}}a_j = a_k{{< /katex >}}, which is absurd because we assumed that {{< katex >}}j{{< /katex >}} and {{< katex >}}k{{< /katex >}} were different. Analogously, the right cancellation law implies that no element appears twice in any column. **Q.E.D.** 

This theorem has a reciprocal of some sort:

**Theorem 2:** Given a latin square {{< katex >}}L = (l_{ij}){{< /katex >}}, one can construct a quasigroup whose Cayley table is {{< katex >}}L{{< /katex >}}.

**Proof:** Let {{< katex >}}G = \{l_{11}, \dots, l_{1n}\}{{< /katex >}} and denote {{< katex >}}g_i := l_{1i}{{< /katex >}}. Define {{< katex >}}\ast{{< /katex >}} by

{{< katex display >}}g_i * g_j = g_{l_{ij}}{{< /katex >}}

by definition, {{< katex >}}\ast{{< /katex >}} is well defined (that is, it is closed in the set). We need to check that the equations {{< katex >}}gx =h{{< /katex >}} and {{< katex >}}yg = h{{< /katex >}} have unique solutions. Consider {{< katex >}}g_lx = g_k{{< /katex >}}, because {{< katex >}}L{{< /katex >}} is a latin square, {{< katex >}}g_k = l_{1k}{{< /katex >}} appears somewhere in row {{< katex >}}l{{< /katex >}}, call the column it appears in {{< katex >}}m{{< /katex >}}, then {{< katex >}}x = g_m{{< /katex >}} is a solution to {{< katex >}}g_lx = g_k{{< /katex >}}. It is unique, because if there existed {{< katex >}}g_{\widetilde{m}}{{< /katex >}} such that {{< katex >}}g_lg_{\widetilde{m}} = g_k = g_lg_m{{< /katex >}}, then {{< katex >}}g_k{{< /katex >}} would appear twice in row {{< katex >}}l{{< /katex >}}, which contradicts the fact that {{< katex >}}L{{< /katex >}} is a latin square. Analogously, now arguing with columns, {{< katex >}}yg = h{{< /katex >}} has a unique solution in {{< katex >}}G{{< /katex >}}. **Q.E.D**.

So now we're set!, we only need to find all latin squares of size {{< katex >}}n{{< /katex >}} and to verify if they represent a valid binary operation for a group. Moreover, we could force the existence of an identity by focusing on finding **normalized** (or **reduced**) latin squares (that is, latin squares where the first row and column are {{< katex >}}0, 1, \dots, n-1{{< /katex >}}).

### The algorithm for finding normalized latin squares of size n.

I use a [depth-first-search](https://en.wikipedia.org/wiki/Depth-first_search) style algorithm, starting with a normalized {{< katex >}}n\times n{{< /katex >}} matrix 

$A = \begin{bmatrix}
0&1&\cdots&n-1\\
1&-1&\cdots&-1\\
\vdots&\vdots&\ddots&\vdots\\
n-1&-1&\cdots&-1\\
\end{bmatrix}$

where the unvisited locations are labeled with a {{< katex >}}-1{{< /katex >}}. We also start with an empty [stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)) {{< katex >}}S{{< /katex >}}. The algorithm goes as follows:

1. Put matrix {{< katex >}}A{{< /katex >}} in the stack {{< katex >}}S{{< /katex >}}.
2. If the stack {{< katex >}}S{{< /katex >}} is empty, stop; if it isn't, pop a matrix {{< katex >}}B{{< /katex >}} from it.
3. Find the first unvisited position {{< katex >}}(i,j){{< /katex >}} in {{< katex >}}B{{< /katex >}} (i.e. the first {{< katex >}}-1{{< /katex >}}), if there isn't any, it is finished, put it in a special list of finished latin squares and go to step 2.
4. Push into the stack the result of replacing this {{< katex >}}-1{{< /katex >}} with every number from {{< katex >}}0{{< /katex >}} to {{< katex >}}n-1{{< /katex >}} that isn't already on its row or column.
5. Go to step 2.

Here's the algorithm implemented in python:

```python
def dfs_in_matrix(A):
    # First we create an empty stack and we put the initial matrix
    # in it.
    list_of_solutions = []
    stack = LifoQueue()
    stack.put(A)
    
    while not stack.empty():
        # We pop a matrix from the stack
        B = stack.get()
        
        # We check if it's finished.
        if is_finished(B):
            list_of_solutions.append(B)
            continue
        
        # We find an unvisited position
        position = find_first_unvisited_position(B)
        if position == None:
            continue
        
        span = span_of_position(position, B)
        for k in range(len(B)):
            if k not in span:
                C = B.copy()
                C[position] = k
                stack.put(C)

    return list_of_solutions

def find_normalized_latin_squares(number):
    A = np.zeros((number, number))
    for k in range(number):
        A[0, k] = k
        A[k, 0] = k
    for i in range(1, number):
        for j in range(1, number):
            A[i, j] = -1
    list_of_solutions = dfs_in_matrix(A)
    return list_of_solutions
```
(the functions `is_finished`, `find_first_unvisited_position` and `span_of_position` are auxiliary, check [this jupyter notebook](https://gist.github.com/miguelgondu/404619477e50ccec62db8c53b4901091) for all the code discussed in this post). It checks out with the literature on the topic[^1], saying that there are 9408 normalized latin squares of size 6.

### The Magma class

Once we have all the normalized latin squares, we can build up a `Magma` class in python and we can write a verification function to find which of these correspond to associative operations (and thus to groups).

```python
class Magma:
    def __init__(self, _matrix):
        self.cayley_table = _matrix
        self.set = set(range(0, len(_matrix[0,:])))
    
    def mult(self, a, b):
        return int(self.cayley_table[a, b])

def is_magma_associative(mag):
    '''
    This function verifies if magma `mag` is associative by brute force.
    '''
    n = len(mag.cayley_table)
    _flag = True
    for a in range(n):
        for b in range(n):
            for c in range(n):
                _flag = _flag and (mag.mult(a, mag.mult(b,c)) == mag.mult(mag.mult(a,b),c))
    return _flag

def find_groups(number):
    latin_squares = find_normalized_latin_squares(number)
    associative_magmas = [sol for sol in latin_squares if is_magma_associative(Magma(sol))]
    return associative_magmas
```

After running this `is_magma_associative` in all 9408 reduced latin squares of order 6 we're left with 80 reduced latin squares such that, when interpreted as quasigroups, are associative. That is, only 80 of the original 9408 reduced latin squares of size 6 can be interpreted as Cayley tables for groups.

## The main result

We're trying to prove the following theorem:

**Theorem 3:** There are only 2 groups of order 6, namely {{< katex >}}S_3{{< /katex >}} and {{< katex >}}\mathbb{Z}_6{{< /katex >}}.

It is useful, then, to cleary state what we interpret as {{< katex >}}S_3{{< /katex >}} and {{< katex >}}\mathbb{Z}_6{{< /katex >}}. {{< katex >}}\mathbb{Z}_6{{< /katex >}} are the usual integers modulo 6 with sum modulo 6, but note that {{< katex >}}\mathbb{Z}_6{{< /katex >}} can also be interpreted in the following way: its a group of six elements {{< katex >}}\{a_1, a_2, a_3, a_4, a_5, 0\}{{< /katex >}} such that

- {{< katex >}}a_1{{< /katex >}} and {{< katex >}}a_5{{< /katex >}} have order 6.
- {{< katex >}}a_2{{< /katex >}} and {{< katex >}}a_4{{< /katex >}} have order 3.
- {{< katex >}}a_3{{< /katex >}} has order 2.
- {{< katex >}}a_2^2 = a_4{{< /katex >}}
- {{< katex >}}a_1a_2 = a_3{{< /katex >}}

(note that we just changed {{< katex >}}i{{< /katex >}} for {{< katex >}}a_i{{< /katex >}}). We can use this information to find an isomorphism between a latin-square-generated group and {{< katex >}}\mathbb{Z}_6{{< /katex >}}.

{{< katex >}}S_3 = \{\sigma_1, \sigma_2, \sigma_3, \rho_1, \rho_2, \text{id}\}{{< /katex >}} is usually interpreted as the group of symmetries of a triangle (where {{< katex >}}\sigma_i{{< /katex >}} is the reflection that fixes vertex {{< katex >}}i{{< /katex >}} and {{< katex >}}\rho_j{{< /katex >}} is a rotation of {{< katex >}}120*j{{< /katex >}} degrees, but we prefer the following presentation:

{{< katex display >}}S_3 = \langle \sigma, \rho\,\vert\,\sigma^2 = \rho^3 = \text{id},\, \sigma\rho = \rho^2\sigma \rangle{{< /katex >}}

In this presentation, the 6 different elements are {{< katex >}}\text{id}, \sigma, \rho, \rho\sigma, \rho^2\sigma{{< /katex >}} and {{< katex >}}\rho^2{{< /katex >}}.

So, to prove theorem 3, we will follow this strategy: we will give an isomorphism from either {{< katex >}}S_3{{< /katex >}} or {{< katex >}}\mathbb{Z}_6{{< /katex >}} to each of the 80 groups found using latin squares:

**Proof (of Theorem 3):** Theorem 1 and 2 show that all possible groups of a certain order are restricted by the amount of normalized latin squares of said order. After filtering the normalized latin squares of size 6 by verifying which represent an associative binary operation, we're left with 80 Cayley tables for groups. In [this jupyter notebook](https://gist.github.com/miguelgondu/404619477e50ccec62db8c53b4901091) we show an explicit isomorphism between each of these 80 latin square generated groups and either {{< katex >}}\mathbb{Z}_6{{< /katex >}} or {{< katex >}}S_3{{< /katex >}}, but for the sake of completeness we show how these isomorphisms were constructed with explicit examples for {{< katex >}}\mathbb{Z}_6{{< /katex >}} and {{< katex >}}S_3{{< /katex >}}. Consider the following normalized latin square:

$\begin{bmatrix}
0 & 1 & 2 & 3 & 4 & 5\\
1 & 5 & 4 & 2 & 3 & 0\\
2 & 3 & 0 & 1 & 5 & 4\\
3 & 4 & 5 & 0 & 1 & 2\\
4 & 2 & 1 & 5 & 0 & 3\\
5 & 0 & 3 & 4 & 2 & 1\end{bmatrix}$

and call the group it induces {{< katex >}}G{{< /katex >}}. After inspecting it, we can tell that the orders of their elements are either {{< katex >}}2{{< /katex >}} or {{< katex >}}3{{< /katex >}}, so it is a candidate for being isomorphic to {{< katex >}}S_3{{< /katex >}}. Choose {{< katex >}}4\mapsto \sigma{{< /katex >}} and {{< katex >}}5\mapsto \rho{{< /katex >}}, and note that
{{< katex display >}}(5*5)*4 = 1*4 = 3 = 4*5,{{< /katex >}}
that is, this group obeys the presentation given for {{< katex >}}S_3{{< /katex >}}. The isomorphism would then be given by

$\begin{matrix}
S_3 & & G\\
\hline
\text{id}&\mapsto&0\\
\sigma_1&\mapsto&4\\
\sigma_2&\mapsto&3\\
\sigma_3&\mapsto&2\\
\rho&\mapsto&5\\
\rho^2&\mapsto&1
\end{matrix}$

Now consider the group {{< katex >}}H{{< /katex >}} given by the following reduced latin square:

$\begin{bmatrix}
0 & 1 & 2 & 3 & 4 & 5\\
1 & 5 & 4 & 2 & 3 & 0\\
2 & 4 & 5 & 1 & 0 & 3\\
3 & 2 & 1 & 0 & 5 & 4\\
4 & 3 & 0 & 5 & 1 & 2\\
5 & 0 & 3 & 4 & 2 & 1\end{bmatrix}$

because the elements of {{< katex >}}H{{< /katex >}} have order either 2, 3 or 6, we will construct an isomorphism between {{< katex >}}H{{< /katex >}} and {{< katex >}}\mathbb{Z}_6{{< /katex >}} using the identification {{< katex >}}\mathbb{Z}_6 = \{a_1, a_2, a_3, a_4, a_5, 0\} {{< /katex >}} stated before. First note that {{< katex >}}3\in H{{< /katex >}} is an element of order 2, {{< katex >}}2, 4\in H{{< /katex >}} have order 6 and {{< katex >}}1, 5\in H{{< /katex >}} have order 3. Because {{< katex >}}1\ast1 = 5{{< /katex >}} and {{< katex >}}4\ast1 = 3{{< /katex >}}, we construct the following isomorphism

$\begin{matrix}
\mathbb{Z}_6 & & H\\
\hline
0&\mapsto&0\\
1&\mapsto&4\\
2&\mapsto&1\\
3&\mapsto&3\\
4&\mapsto&5\\
5&\mapsto&2\\
\end{matrix}$

 **Q.E.D**

---

[^1]: The results the algorithm gave were in par with what's said in *[Small Latin Squares, Quasigroups and Loops](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.71.1011&rep=rep1&type=pdf)*, an article by Brendan D. Mackay, Alison Meynert and Wendy Myrvold. Check *Table 1* of their article for more details. 