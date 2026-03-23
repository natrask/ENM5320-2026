# Homework 4 - FEM vs PINNs (Due 4/1 by 11:59 PM)

In this homework we will compare a solution to the Poisson equation using FEM and a physics-informed neural network. You are welcome to work with friends or to use external resources/tools. If you do, you **must** attribute them in your submission to comply with the UPenn academic code.

Consider the Poisson equation
$$-\nabla^2 u = f$$
with Dirichlet boundary conditions
$$u|_{\partial \Omega} = g.$$

We will discretize this PDE on the unit square with a circle of radius $0.15$ removed from its center ($\Omega = [0,1]^2 \setminus B_{0.15}(0.5,0.5)$).

To assess the convergence of FEM and PINNs, we will use the *method of manufactured solutions* to generate a smooth solution.

1. "Manufacture" a choice of $f$ and $g$ such that $u = \sin(2\pi x)\sin(2\pi y)$ solves the PDE.

2. Use the scikit-FEM code provided in class to solve this problem. Solve on a sequence of meshes to analyze convergence. Generate a table illustrating the L2 error as a function of the maximum element size, $h$. Comment on how this compares with the FEM convergence theory we discussed in class.

3. Generate a PINN that solves the same equation. It will be challenging to perform an apples-to-apples comparison of performance. Articulate your choices for:
    - The number of collocation points
    - The architecture for the neural network solution, and the number of parameters
    - The optimizer parameters (step_size and how many steps to train)
    - The weight between the PDE residual and the BC residual

4. Discuss/compare the two methods in terms of speed of solution, accuracy, and any other takeaways from the exercise that you find noteworthy.