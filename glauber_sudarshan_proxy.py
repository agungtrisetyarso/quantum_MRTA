# =============================================================================
# Colab Simulation: Glauber-Sudarshan P-Function in Information-Geometric Task Allocation
# Context: Demonstrates classical vs non-classical behavior of ρ★
# =============================================================================

!pip install qutip numpy matplotlib seaborn -q

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt
from scipy.linalg import expm, logm

np.random.seed(42)

print("=== Glauber-Sudarshan P-Function Simulation ===\n")

# ========================= PARAMETERS =========================
n_agents = 2
m_tasks = 2
d = m_tasks ** n_agents  # 4 (two qubits)
lambda_reg = 0.5

# ========================= HELPERS =========================
def random_hermitian(dim):
    H = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    return (H + H.conj().T) / 2

def random_density_matrix(dim):
    return qt.rand_dm(dim, density=1.0).full()

def compute_rho_star(sigma, H, lam):
    G = logm(sigma) - (1.0 / lam) * H
    rho = expm(G)
    return rho / np.trace(rho)

# ========================= SETUP STATES =========================
sigma = random_density_matrix(d)           # Target correlation state
H = random_hermitian(d)                    # Cost operator

rho_star = compute_rho_star(sigma, H, lambda_reg)   # Optimal coherent state

# Classical projection (commutes with σ)
evals, U = np.linalg.eigh(sigma)
rho_class = U @ np.diag(np.diag(U.conj().T @ rho_star @ U)) @ U.conj().T
rho_class /= np.trace(rho_class)

# Convert to QuTiP
rho_star_qt = qt.Qobj(rho_star)
rho_class_qt = qt.Qobj(rho_class)
sigma_qt = qt.Qobj(sigma)

print("States generated successfully.")

# ========================= GLAUBER-SUDARSHAN P-FUNCTION =========================
def plot_p_function(rho, title, alpha_max=5, npts=100):
    """Plot Glauber-Sudarshan P-function (via Husimi Q + deconvolution or sampling)"""
    fig = plt.figure(figsize=(8, 6))
    x = np.linspace(-alpha_max, alpha_max, npts)
    y = np.linspace(-alpha_max, alpha_max, npts)
    X, Y = np.meshgrid(x, y)
    alpha = X + 1j * Y
    
    # For small systems, we use a sampled approximation or Husimi for visualization
    # True P-function is often singular; we use a regularized/smoothed version
    Q = qt.qfunc(rho, x, y, method='clenshaw')  # Husimi Q is smooth
    # For illustration, we show Husimi Q as a proxy (P is more singular)
    
    plt.contourf(X, Y, Q, levels=80, cmap='RdBu_r')
    plt.colorbar(label='Quasi-Probability (Husimi Q proxy)')
    plt.xlabel(r'Re($\alpha$)')
    plt.ylabel(r'Im($\alpha$)')
    plt.title(title)
    plt.grid(False)
    return fig

print("\nGenerating phase-space distributions...")

plt.figure(figsize=(15, 5))

# Optimal Coherent State ρ★
plt.subplot(1, 3, 1)
x = np.linspace(-4, 4, 80)
Q_star = qt.qfunc(rho_star_qt, x, x)
plt.contourf(x, x, Q_star, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Glauber-Sudarshan Proxy (ρ★)\nOptimal Quantum Allocation')
plt.xlabel(r'Re($\alpha$)')
plt.ylabel(r'Im($\alpha$)')

# Classical Projection
plt.subplot(1, 3, 2)
Q_class = qt.qfunc(rho_class_qt, x, x)
plt.contourf(x, x, Q_class, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Classical Projection\n(commutes with σ)')
plt.xlabel(r'Re($\alpha$)')

# Target σ
plt.subplot(1, 3, 3)
Q_sigma = qt.qfunc(sigma_qt, x, x)
plt.contourf(x, x, Q_sigma, 80, cmap='RdBu_r')
plt.colorbar()
plt.title('Target State σ')
plt.xlabel(r'Re($\alpha$)')

plt.tight_layout()
plt.show()

# ========================= ANALYSIS =========================
print("\n=== Analysis ===")
print("In the Glauber-Sudarshan P-representation:")
print("• Classical states → P(α) is a valid non-negative probability distribution")
print("• Non-classical states → P(α) becomes singular or negative (distributions of deltas/derivatives)")

# Commutation & Advantage
comm_norm = np.linalg.norm(rho_star @ sigma - sigma @ rho_star, 'fro')
print(f"\n||[ρ★, σ]|| = {comm_norm:.6f} → Quantum advantage exists" if comm_norm > 1e-5 else "No quantum advantage (classical case)")

print("\nKey Paper Insight:")
print("The true resource for advantage in this framework is coherence in σ's eigenbasis,")
print("not necessarily non-classicality in the Glauber-Sudarshan / Wigner sense.")
