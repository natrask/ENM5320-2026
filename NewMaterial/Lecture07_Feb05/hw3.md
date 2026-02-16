# Homework 3 - Due Wed. 2/25 at 11:59pm

In the previous assignments we have generated *synthetic data* which comes from PDEs. In class we generated a more realistic set of data that comes from post-processing a simulation of the Kelvin-Helmholtz instability. Unlike our previous exercises, we do not have any idea what sort of PDE should be satisfied for this.

**Submission.** All assignments will be submitted via [Canvas](https://canvas.upenn.edu/courses/1912564/assignments/14471756). You are free to choose how to submit (scanned handwritten notes/Latex/markdown/whatever). Be sure to include Jupyter notebooks of all code written, and you may want to also submit screen shots of your code output in case we have trouble rerunning your code. 

**AI use policy:** You are encouraged to use LLMs to help you explain code and to debug your own code. In order to succeed as the class gets progressively more difficult, you will however need command over how a code like this is structured. **Do not** just ask an LLM to complete these assignments or you will lose out on building the muscle memory needed to complete future assignments!

# Question 1 - Load data

Make a Jupyter notebook that loads the **mean of the concentration field only**. Provide evidence in your notebook that you have successfully loaded the dataset (e.g., make some plots that show the correct trends). Note that the dataset has much more information (both mean and standard deviation for concentration and the vector-valued velocity field).

# Question 2 - Fit a linear stencil to the data

Adapt your code from the previous assignment to fit a stencil to this data. We are keeping the direction for this intentionally vague. In your Jupyter notebook, include in a markdown block an explanation of your training strategy.

Note that using a linear finite difference stencil to predicts the mean of the concentration field is missing *a lot* of information. The process is nonlinear, and depends on coupling between the velocity field (where the fluid instability happens that drives mixing) and unresolved processes that are captured in the standard deviations of the fields. We will use this as a placeholder before finally moving toward a completely nonlinear description of the process in the next assignment.