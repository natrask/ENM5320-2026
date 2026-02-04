# Homework 2 - Due Wed. 2/11 at 11:59pm

In class 2/4 we stepped through a code that showed you how to do two things:
1. Use Newton's method with backtracking to solve systems of nonlinear equations, using PyTorch's `jacrev` to compute Jacobians.
2. Use a PyTorch optimizer to infer a good finite difference stencil for the heat equation, backpropping through the solution of the finite difference solver.

For this homework, we will be doing some light modifications of the in-class exercise to make sure that you were able to follow along. This should not take long if you kept up in class, but will give folks new to PyTorch a chance to get some extra help outside of class.

**Submission.** All assignments will be submitted via [Canvas](https://canvas.upenn.edu/courses/1912564/assignments/14405090). You are free to choose how to submit (scanned handwritten notes/Latex/markdown/whatever). Be sure to include Jupyter notebooks of all code written, and you may want to also submit screen shots of your code output in case we have trouble rerunning your code. 

**AI use policy:** You are encouraged to use LLMs to help you explain code and to debug your own code. In order to succeed as the class gets progressively more difficult, you will however need command over how a code like this is structured. **Do not** just ask an LLM to complete these assignments or you will lose out on building the muscle memory needed to complete future assignments!

# Question 1 - Solve a system of nonlinear equations

Using the Newton-Armijo code from class as a template, solve the following system of 3 nonlinear equations:

$$\begin{align}
f_1(x, y, z) &= x^2 + y^2 + z^2 - 6 = 0 \\
f_2(x, y, z) &= x^2 - y + z = 0 \\
f_3(x, y, z) &= x + y - z^2 = 0
\end{align}$$

**Interpretation:** This system represents:
- A sphere of radius $\sqrt{6}$ centered at the origin
- A paraboloid: $y = x^2 + z$
- Another paraboloid: $z^2 = x + y$

**Tasks:**
1. Compute the Jacobian matrix analytically and verify that PyTorch's `jacrev` gives the same result at a test point.
2. Implement Newton's method with Armijo backtracking to find a solution.
3. Find *one* solution $(x_s,y_s,z_s)$ and plug back into the equation to confirm $f_1=f_2=f_3=0$.

# Question 2 - Data-driven finite difference stencil

In class we reverse engineered a discrete stencil from the following analytic solution to the heat equation.
$$u^{ex}(x,t) = e^{-4\pi^2 t} \sin(2\pi x)$$

Repeat this exercise to learn a discrete stencil for the transport equation
$$\partial_t u + \partial_x u = 0$$
Using the exact solution as data
$$u^{ex}(x,t) = \sin(2\pi (x-t))$$

**Tasks:**
1. How low can you get the error?
2. Look at the numerical value for the stencil that you solved for. Compare this to the $\mathcal{D}_+$ and $\mathcal{D}_-$ stencils that we've discussed in class - how and why is it different?
3. Run the same learned stencil for a different initial condition and analytic solution (e.g. $\sin(2π K (x-t))$ for $K=2,3,...$). What is the effect on the performance and why?