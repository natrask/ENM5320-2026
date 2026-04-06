# Homework 5 - Due Mon. 4/13 at 11:59pm

In class we built a data-driven Whitney form tutorial that learns a constitutive flux for the 1D steady advection-diffusion equation. The model decomposes the flux into a trainable linear diffusivity $\mu$ and a nonlinear neural network correction $N(\hat{u})$, and uses Newton's method to solve the resulting nonlinear system on a coarsened Whitney form discretization.

For this assignment, you will swap out the linear advection-diffusion data for data from the **steady viscous Burgers equation**, which has a genuinely nonlinear flux. This will test whether the same Whitney form architecture can discover nonlinear constitutive behavior from data.

**Submission.** All assignments will be submitted via [Canvas](https://canvas.upenn.edu/courses/1912564/assignments). You are free to choose how to submit (scanned handwritten notes/Latex/markdown/whatever). Be sure to include Jupyter notebooks of all code written, and you may want to also submit screen shots of your code output in case we have trouble rerunning your code.

**AI use policy:** You are encouraged to use LLMs to help you explain code and to debug your own code. In order to succeed as the class gets progressively more difficult, you will however need command over how a code like this is structured. **Do not** just ask an LLM to complete these assignments or you will lose out on building the muscle memory needed to complete future assignments!

---

## Background: Steady Viscous Burgers Equation

Consider the steady viscous Burgers equation on $[0,1]$:

$$-\varepsilon\, u'' + u\, u' = 0$$

with Dirichlet boundary conditions

$$u(0) = 1, \qquad u(1) = 0.$$

Unlike the advection-diffusion equation (where the advection velocity is a constant), the advection here is **nonlinear** — the velocity is the solution itself. This means the neural network $N(\hat{u})$ must learn a genuinely nonlinear constitutive law rather than a linear one.

### Deriving the exact solution

**Step 1.** Observe that $u\,u' = \frac{d}{dx}\!\left(\frac{u^2}{2}\right)$. Integrate the PDE once in $x$:

$$-\varepsilon\, u' + \frac{u^2}{2} = C$$

where $C$ is a constant of integration.

**Step 2.** Define $a = \sqrt{2C}$ so that $u' = \frac{u^2 - a^2}{2\varepsilon}$. This is a separable ODE. Using partial fractions:

$$\frac{du}{u^2 - a^2} = \frac{dx}{2\varepsilon} \qquad \Longrightarrow \qquad \frac{1}{2a}\ln\left|\frac{u - a}{u + a}\right| = \frac{x}{2\varepsilon} + K$$

**Step 3.** Exponentiate and solve for $u$:

$$\frac{u - a}{u + a} = B\,e^{ax/\varepsilon}$$

for some constant $B$. Solving for $u$:

$$u(x) = a\,\frac{1 + B\,e^{ax/\varepsilon}}{1 - B\,e^{ax/\varepsilon}}$$

**Step 4.** Apply $u(1) = 0$: this gives $1 + B\,e^{a/\varepsilon} = 0$, so $B = -e^{-a/\varepsilon}$. Substituting:

$$u(x) = a\,\frac{1 - e^{a(x-1)/\varepsilon}}{1 + e^{a(x-1)/\varepsilon}} = a\,\tanh\!\left(\frac{a(1-x)}{2\varepsilon}\right)$$

**Step 5.** Apply $u(0) = 1$: this requires

$$a\,\tanh\!\left(\frac{a}{2\varepsilon}\right) = 1$$

This is a transcendental equation for $a$ that must be solved numerically (e.g., via bisection or `scipy.optimize.brentq`).

### Summary

The exact solution to the steady viscous Burgers equation with $u(0)=1$, $u(1)=0$ is:

$$\boxed{u(x) = a\,\tanh\!\left(\frac{a\,(1-x)}{2\varepsilon}\right)}$$

where $a$ solves $a\,\tanh\!\bigl(\frac{a}{2\varepsilon}\bigr) = 1$.

For small $\varepsilon$, $\tanh(\cdot) \approx 1$ so $a \approx 1$ and $u \approx 1$ everywhere except in a thin viscous layer near $x = 1$.

---

## Question 1 — Implement the Burgers exact solution

Write a function that computes the exact solution $u(x)$ for a given $\varepsilon$. You will need to:

1. Solve the transcendental equation $a\,\tanh(a/(2\varepsilon)) = 1$ for $a$.
2. Evaluate $u(x) = a\,\tanh\bigl(a(1-x)/(2\varepsilon)\bigr)$.

Plot the exact solution for $\varepsilon = 0.05$ and $\varepsilon = 0.5$. Comment on how the solution profile changes with $\varepsilon$ and how this compares to the advection-diffusion boundary layer.

## Question 2 — Swap the data in the Whitney form tutorial

Starting from the `DataDriven_Whitney_Forms.ipynb` notebook, replace the advection-diffusion data with Burgers data:

- Replace the exact solution function.
- Update the boundary conditions to $u(0) = 1$, $u(1) = 0$.
- Keep all other architecture choices the same (mesh, Whitney forms, Newton solver, FluxNN, training loop).

Run the training loop and report:

1. The loss curve over training iterations.
2. The learned value of $\mu$.
3. A plot comparing the learned solution to the exact Burgers solution.

