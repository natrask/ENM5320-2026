## Stability Analysis for Norliners Probleme

- Consider a nonsingular $n \times n$ matrix $A$
- Then $A_{x}=b$ has a unifue soln. Now let $f(x)$ be a umooth vector values function, $S: \mathbb{R}^{n} \rightarrow \mathbb{R}^{n}$
- Want to solve
(*) $A x+\varepsilon f(x)=b$
- A simple iterative scheme Picad/

$$
\begin{aligned}
& A x^{(n+1)}+\varepsilon f\left(x^{(n)}\right)=b \quad n=0,4 \ldots \\
& x^{0}=A^{-1} b
\end{aligned}
$$

Then Assume $f(x)$ is unifamly Lipschitz i.e $\exists L$ such that for all $x, y$

$$
|f(x)-f(y)| \leq L|x-y|
$$

If $\psi=\varepsilon\left|A^{-1}\right| L<1$ then
has a unique soln, $\tilde{x}$ and the terative scheme converges $\lim _{n \rightarrow \infty} x_{n}=\tilde{x}$

Pf To khow nuiquenesh, well assume two solutions $x, y$ and show they muit be equal
multant

$$
\begin{aligned}
& A x+\varepsilon f(x)=b \\
& A y+\varepsilon f(y)=b
\end{aligned}
$$

$$
A(x-y)+\varepsilon[f(x)-f(y)]=0
$$

abs looth sides

Put

$$
\begin{aligned}
x-y & =-\varepsilon A^{-1}(f(x)-f(y)) \\
|x-y| & =\varepsilon\left|A^{-1}[f(x)-f(y)]\right| \\
& \leq \varepsilon\left|A^{-1}\right||f(x)-f(y)| \\
& \leq \varepsilon\left|A^{-1}\right| L|x-y| \\
|x-y| & \leq \tau|x-y| \\
(1-\tau)|x-y| \leq 0 & |x-y| \leq 0 \\
& \Rightarrow x=y
\end{aligned}
$$

We can extend the same aryment to analyze converyence of the scheme

$$
A x^{n}+\varepsilon f\left(x^{n-1}\right)=b
$$

by taking $(x, y)=\left(x^{n+1}, x^{n}\right)$

$$
\begin{align*}
x^{n+1}-x^{n} & =-\varepsilon A^{-1}\left(f\left(x^{n-1}\right)-f\left(x^{n-1}\right)\right) \\
\left|x^{n+1}-x^{n}\right| & \leq \tau\left|x^{n}-x^{n-1}\right| \\
& \leq \tau^{2}\left|x^{n-1}-x^{n-2}\right| \\
& \vdots \tau^{n}\left|x^{1}-x^{0}\right| \tag{**}
\end{align*}
$$

For $m>n$

$$
\left|x^{m}-x^{n}\right|=\left|\sum_{i=n}^{m-1}\left(x^{i+1}-x^{i}\right)\right|
$$

$$
\begin{aligned}
& \leq\left(\sum_{i=n}^{m-1} \tau^{i}\right)\left|x^{\prime}-x^{0}\right| \\
& \leq \frac{\tau^{n}}{1-\tau}\left|x^{\prime}-x^{0}\right|
\end{aligned}
$$

Geametric Series formula
$\sum_{k=0}^{n} a c^{k}=a\left(\frac{1-s^{n+k}}{1-r}\right)$

This is an ayment refered to as a Cavely sequence
ie for each ばれでつ between $n, m-1$ ． we get a $\tau^{7}$ from（194）

