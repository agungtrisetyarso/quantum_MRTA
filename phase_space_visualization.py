# =============================================================================
# Honest Phase-Space Visualization + True Coherence Measure
# =============================================================================

!pip install qutip numpy matplotlib seaborn -q

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.linalg import expm, logm

np.random.seed(42)

d = 4
lambda_reg = 0.5

def random_hermitian(dim): 
    H = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    return (H + H.conj().T)/2

def random_dm(dim):
    return qt.rand_dm(dim, density=1.0).full()

sigma = random_dm(d)
H = random_hermitian(d)

# Compute ρ★
G = logm(sigma) - (1.0/lambda_reg)*H
rho_star = expm(G)
rho_star /= np.trace(rho_star)

# Classical projection
evals, U = np.linalg.eigh(sigma)
rho_class = U @ np.diag(np.diag(U.conj().T @ rho_star @ U)) @ U.conj().T
rho_class /= np.trace(rho_class)

rho_star_qt = qt.Qobj(rho_star)
rho_class_qt = qt.Qobj(rho_class)

# ========================= TRUE COHERENCE MEASURE =========================
def l1_coherence(rho, U):
    """ℓ₁ coherence in σ's eigenbasis"""
    rho_diag_basis = U.conj().T @ rho @ U
    off_diag = rho_diag_basis - np.diag(np.diag(rho_diag_basis))
    return np.sum(np.abs(off_diag))

coh_star = l1_coherence(rho_star, U)
coh_class = l1_coherence(rho_class, U)

print(f"ℓ₁ coherence of ρ★ in σ basis : {coh_star:.4f}")
print(f"ℓ₁ coherence of classical proj: {coh_class:.4f}")

# ========================= PLOTTING =========================
fig = plt.figure(figsize=(18, 6))

# Husimi Q plots
x = np.linspace(-4, 4, 80)

plt.subplot(1, 4, 1)
Q = qt.qfunc(rho_star_qt, x, x)
plt.contourf(x, x, Q, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Husimi Q(α) — ρ★\n(Optimal Quantum Allocation)')
plt.xlabel('Re(α)'); plt.ylabel('Im(α)')

plt.subplot(1, 4, 2)
Q = qt.qfunc(rho_class_qt, x, x)
plt.contourf(x, x, Q, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Husimi Q(α) — Classical Projection')

plt.subplot(1, 4, 3)
Q = qt.qfunc(qt.Qobj(sigma), x, x)
plt.contourf(x, x, Q, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Husimi Q(α) — Target σ')

# True coherence bar
plt.subplot(1, 4, 4)
plt.bar(['ρ★', 'Classical'], [coh_star, coh_class], color=['blue', 'orange'])
plt.ylabel('ℓ₁ Off-Diagonal Coherence')
plt.title('True Quantum Resource\n(Corollary 1)')
plt.ylim(0, max(coh_star, coh_class)*1.1)

plt.tight_layout()
plt.show()
