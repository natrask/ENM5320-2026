import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange, BarycentricInterpolator

# Define the Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Bernstein polynomial
def bernstein_polynomial(f, n, x):
    """Compute Bernstein polynomial of degree n for function f at points x."""
    from scipy.special import comb
    x = np.atleast_1d(x)
    result = np.zeros_like(x)
    
    # Sample f at uniform points
    k_vals = np.arange(n + 1)
    f_vals = f(k_vals / n * 2 - 1)  # Map [0, n] to [-1, 1]
    
    for k in range(n + 1):
        # Map x from [-1, 1] to [0, 1]
        t = (x + 1) / 2
        # Bernstein basis
        B_nk = comb(n, k, exact=True) * (t**k) * ((1 - t)**(n - k))
        result += f_vals[k] * B_nk
    
    return result

# Set up the domain
x_fine = np.linspace(-1, 1, 1000)
f_true = runge(x_fine)

# Number of interpolation points
n = 11

# Uniform nodes
x_uniform = np.linspace(-1, 1, n)
f_uniform = runge(x_uniform)

# Chebyshev nodes
k_cheb = np.arange(n)
x_chebyshev = np.cos((2*k_cheb + 1) * np.pi / (2*n))
x_chebyshev = np.sort(x_chebyshev)
f_chebyshev = runge(x_chebyshev)

# Create Lagrange interpolants
poly_uniform = BarycentricInterpolator(x_uniform, f_uniform)
poly_chebyshev = BarycentricInterpolator(x_chebyshev, f_chebyshev)
f_uniform_interp = poly_uniform(x_fine)
f_chebyshev_interp = poly_chebyshev(x_fine)

# Bernstein polynomial (degree n-1)
f_bernstein = bernstein_polynomial(runge, n * 3, x_fine)  # Use higher degree for better approximation

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Uniform nodes (showing Runge phenomenon)
ax = axes[0, 0]
ax.plot(x_fine, f_true, 'k-', linewidth=2, label='True function')
ax.plot(x_fine, f_uniform_interp, 'r--', linewidth=1.5, label=f'Lagrange (n={n})')
ax.plot(x_uniform, f_uniform, 'ro', markersize=6, label='Uniform nodes')
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('(a) Uniform Nodes - Runge Phenomenon', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)

# Plot 2: Chebyshev nodes (convergent)
ax = axes[0, 1]
ax.plot(x_fine, f_true, 'k-', linewidth=2, label='True function')
ax.plot(x_fine, f_chebyshev_interp, 'b--', linewidth=1.5, label=f'Lagrange (n={n})')
ax.plot(x_chebyshev, f_chebyshev, 'bo', markersize=6, label='Chebyshev nodes')
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('(b) Chebyshev Nodes - Convergent', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)

# Plot 3: Bernstein polynomial (convergent)
ax = axes[1, 0]
ax.plot(x_fine, f_true, 'k-', linewidth=2, label='True function')
ax.plot(x_fine, f_bernstein, 'g--', linewidth=1.5, label=f'Bernstein (n={n*3})')
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('(c) Bernstein Polynomial - Convergent', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)

# Plot 4: Error comparison
ax = axes[1, 1]
error_uniform = np.abs(f_uniform_interp - f_true)
error_chebyshev = np.abs(f_chebyshev_interp - f_true)
error_bernstein = np.abs(f_bernstein - f_true)

ax.semilogy(x_fine, error_uniform, 'r-', linewidth=1.5, label='Uniform (diverges)')
ax.semilogy(x_fine, error_chebyshev, 'b-', linewidth=1.5, label='Chebyshev')
ax.semilogy(x_fine, error_bernstein, 'g-', linewidth=1.5, label='Bernstein')
ax.set_xlim(-1, 1)
ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('Absolute Error (log scale)', fontsize=11)
ax.set_title('(d) Error Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')

plt.suptitle(r'Approximating Runge Function: $f(x) = \frac{1}{1 + 25x^2}$', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the figure
plt.savefig('runge_phenomenon.png', dpi=300, bbox_inches='tight')
plt.savefig('runge_phenomenon.pdf', bbox_inches='tight')
print("Figure saved as runge_phenomenon.png and runge_phenomenon.pdf")
plt.close()
