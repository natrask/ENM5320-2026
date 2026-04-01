# Homework 1 - Due Fri. 1/30 at 11:59pm

In class we learned the basics of the finite difference method (FDM), and I gave you two pieces of code that solve an equation using finite differences and with a physics-informed neural network. Eventually, we will use FDM as a platform for learning simulators of physics from data, using analysis to tell us how to set things up so that we get guaranteed performance.

For now, I want to make sure everyone is comfortable making this code go. I've already written it for you, but I want you to use it to run some experiments. 

**Submission.** All assignments will be submitted via [Canvas](https://canvas.upenn.edu/courses/1912564/assignments/14405090). You are free to choose how to submit (scanned handwritten notes/Latex/markdown/whatever). Be sure to include Jupyter notebooks of all code written, and you may want to also submit screen shots of your code output in case we have trouble rerunning your code. 

**AI use policy:** You are encouraged to use LLMs to help you explain code and to debug your own code. In order to succeed as the class gets progressively more difficult, you will however need command over how a code like this is structured. **Do not** just ask an LLM to complete these assignments or you will lose out on building the muscle memory needed to complete future assignments!

# Question 1 - Consistency of FDM

In class we introduced the finite difference stencil
$$\frac{u^{n+1}_i - u^n_i}{k} = \alpha \frac{u^{n+1}_{i-1} - 2u^{n+1}_i + u^{n+1}_{i+1}}{h^2}$$

where $u^n_i$ approximates $u(x_i, t_n)$, $k = \Delta t$ is the time step, and $h = \Delta x$ is the spatial step.

Prove that this finite difference scheme gives a *consistent* approximation to the PDE by showing that the local truncation error goes to zero:

$$\lim_{k,h \rightarrow 0} \left| \frac{u(x_i,t_{n+1}) - u(x_i,t_n)}{k} - \frac{\alpha }{h^2} (u(x_{i-1},t_{n+1}) - 2u(x_i,t_{n+1}) + u(x_{i+1},t_{n+1})) - \partial_t u(x_i,t_n) + \alpha \partial_{xx} u(x_i,t_n) \right|^2 = 0$$

*Hint:* Use the triangle inequality to split the error between the time piece and the spatial piece. Expand in a Taylor series for each piece to get a residual that scales with either $k$ or $h$, respectively.

# Question 2 - Convergence study

Show that for $\alpha = 1$, the following is an exact solution to the heat equation
$$u^{ex}(x,t) = e^{-4\pi^2 t} \sin(2\pi x)$$

We are going to confirm that the code recovers this solution as we refine the mesh.

- Fix $k = \frac{h}{2}$, and $h = \frac{1}{N}$ and run over $t \in [0,1]$
- Run the solver for $N = 8$ and compute the root-mean-square norm of the error at the final step $$E_{RMS} = \sqrt{ \sum_i \left( u^{ex}(x_i,t=1) - u(x_i,t=1) \right)^2 }$$
- Repeat for $N = 8,16,32,64,128$ and generate a log-log plot of $N$ on the x-axis and the RMS norm on the y-axis. Demonstrate algebraic scaling showing that your solution is converging to the true solution as $N \rightarrow \infty$.
