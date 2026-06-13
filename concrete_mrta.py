import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, eigh

# ====================== Concrete MRTA Instance ======================
H = np.array([
    [1.0, 2.5, 3.0, 4.0],
    [2.5, 1.5, 4.5, 2.0],
    [3.0, 4.5, 2.0, 1.0],
    [4.0, 2.0, 1.0, 3.5]
], dtype=complex)

sigma_raw = np.array([
    [0.10, 0.05, 0.05, 0.00],
    [0.05, 0.10, 0.00, 0.05],
    [0.05, 0.00, 0.10, 0.05],
    [0.00, 0.05, 0.05, 0.10]
], dtype=complex)
sigma = sigma_raw / np.trace(sigma_raw)

lam = 0.5

def compute_rho_star(H, sigma, lam):
    G = logm(sigma) - (1.0 / lam) * H
    rho = expm(G)
    return rho / np.trace(rho)

rho_star = compute_rho_star(H, sigma, lam)

def classical_projection(rho, sigma):
    _, U = eigh(sigma)
    rho_diag = U.conj().T @ rho @ U
    rho_class = U @ np.diag(np.diag(rho_diag)) @ U.conj().T
    return rho_class

rho_class = classical_projection(rho_star, sigma)

# ====================== Improved Plot Function ======================
def plot_matrix(ax, mat, title, is_H=False):
    real_part = np.real(mat)
    
    if is_H:
        # Lighter colormap for H matrix so text is clearly visible
        im = ax.imshow(real_part, cmap='YlOrRd', vmin=0.8, vmax=4.8)
    else:
        im = ax.imshow(real_part, cmap='RdBu_r', vmin=-0.45, vmax=0.45)
    
    ax.set_title(title, fontsize=14, pad=15)
    ax.axis('off')
    
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            if abs(val) > 0.015:
                text = f'{val.real:.2f}' + (f'{val.imag:+.2f}j' if abs(val.imag) > 0.02 else '')
                color = 'black' if is_H else ('black' if abs(val.real) > 0.1 else 'darkgray')
                ax.text(j, i, text, ha='center', va='center', fontsize=8.8, 
                        color=color, fontweight='bold' if is_H else 'normal')
    return im

# ====================== Create Figure ======================
fig, axs = plt.subplots(2, 2, figsize=(15, 12), 
                        gridspec_kw={'wspace': 0.35, 'hspace': 0.35})

im1 = plot_matrix(axs[0, 0], H, r'Cost Operator $H$', is_H=True)
im2 = plot_matrix(axs[0, 1], sigma, r'Target State $\sigma$')
im3 = plot_matrix(axs[1, 0], rho_star, r'Optimal Coherent State $\rho^\star$')
im4 = plot_matrix(axs[1, 1], rho_class, r'Classical Projection $\rho$ (commuting with $\sigma$)')

# Colorbar (shared - using H's colormap for reference)
cbar = fig.colorbar(im1, ax=axs, fraction=0.046, pad=0.02, shrink=0.75)
cbar.set_label('Real Part of Matrix Elements', fontsize=12)

fig.suptitle('Matrix Visualization for Concrete 2-Agent, 2-Task MRTA Instance\n'
             r'($\lambda = 0.5$, $[H, \sigma] \neq 0$)', 
             fontsize=16, y=0.96)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('mrta_concrete_matrices_fixed.png', dpi=300, bbox_inches='tight')
plt.show()
