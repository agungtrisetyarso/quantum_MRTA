import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, eigh
import seaborn as sns

plt.style.use('default')
sns.set_palette("coolwarm")

def random_hermitian(d, seed=42):
    np.random.seed(seed)
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    return (A + A.conj().T) / 2

def random_density_matrix(d, seed=42):
    np.random.seed(seed)
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = A @ A.conj().T
    return rho / np.trace(rho)

def compute_rho_star(H, sigma, lam=0.8):
    log_sigma = logm(sigma)
    G = log_sigma - (1.0 / lam) * H
    rho = expm(G)
    return rho / np.trace(rho)

def classical_projection(rho, sigma):
    eigvals, U = eigh(sigma)
    rho_diag = U.conj().T @ rho @ U
    rho_class = U @ np.diag(np.diag(rho_diag)) @ U.conj().T
    return rho_class

def plot_matrix(ax, mat, title):
    real_part = np.real(mat)
    im = ax.imshow(real_part, cmap='RdBu_r', vmin=-0.35, vmax=0.35)
    
    ax.set_title(title, fontsize=13, pad=12)
    ax.axis('off')
    
    # Add value annotations
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if abs(val) > 0.015:
                text = f'{val.real:.2f}' + (f'{val.imag:+.2f}j' if abs(val.imag) > 0.015 else '')
                ax.text(j, i, text, ha='center', va='center', fontsize=7.2,
                        color='black' if abs(val.real) > 0.08 else 'darkgray')
    return im

# ====================== Main ======================
d = 4
H = random_hermitian(d, seed=123)
sigma = random_density_matrix(d, seed=456)
H = H + 0.3 * random_hermitian(d, seed=789)   # ensure non-commuting

rho_star = compute_rho_star(H, sigma, lam=0.8)
rho_class = classical_projection(rho_star, sigma)

# Create figure with more space
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), gridspec_kw={'wspace': 0.25})

im1 = plot_matrix(axs[0], rho_star, r'$\rho^\star$ (Coherent)')
im2 = plot_matrix(axs[1], rho_class, r'Classical Projection')
im3 = plot_matrix(axs[2], sigma, r'$\sigma$ (Target)')

# Shared colorbar on the right
cbar = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.04, shrink=0.85)
cbar.set_label('Real Part of Matrix Elements', fontsize=11)

fig.suptitle('Wigner-like Visualization of Optimal State, Classical Projection, and Target\n'
             'Negative regions (blue) in $\\rho^\\star$ highlight the coherence resource', 
             fontsize=14, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('wigner_plot_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
