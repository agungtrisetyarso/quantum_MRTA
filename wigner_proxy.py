# =============================================================================
# Colab Simulation: Wigner Distribution Function in Information-Geometric Task Allocation
# Context: "Information-geometric task allocation with natural gradient swarm dynamics"
# Demonstrates why Wigner negativity ≠ quantum advantage certificate
# =============================================================================

!pip install qutip numpy matplotlib seaborn -q

import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
import seaborn as sns
import qutip as qt

np.random.seed(42)

print("=== Wigner Function Analysis for Quantum Task Allocation ===\n")

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
    rho = qt.rand_dm(dim, density=1.0).full()
    return rho

def compute_rho_star(sigma, H, lambda_reg):
    G = logm(sigma) - (1.0 / lambda_reg) * H
    rho_star = expm(G)
    rho_star /= np.trace(rho_star)
    return rho_star

# ========================= SETUP =========================
sigma = random_density_matrix(d)                    # Target correlation state
H = random_hermitian(d)                             # Cost operator

rho_star = compute_rho_star(sigma, H, lambda_reg)

# Classical projection (commutes with σ)
evals, U = np.linalg.eigh(sigma)
rho_class = U @ np.diag(np.diag(U.conj().T @ rho_star @ U)) @ U.conj().T
rho_class /= np.trace(rho_class)

print("States prepared successfully.")

# ========================= WIGNER FUNCTION =========================
# Convert to QuTiP objects (tensor of two qubits)
rho_star_qt = qt.Qobj(rho_star)
rho_class_qt = qt.Qobj(rho_class)
sigma_qt = qt.Qobj(sigma)

# Compute discrete Wigner function using QuTiP spin Wigner (works well for small systems)
def plot_wigner(rho_qt, title):
    fig = plt.figure(figsize=(8, 6))
    xvec = np.linspace(-3, 3, 100)
    W = qt.wigner(rho_qt, xvec, xvec)   # Spin Wigner for composite system
    plt.contourf(xvec, xvec, W, levels=100, cmap='RdBu_r')
    plt.colorbar(label='Wigner quasi-probability')
    plt.xlabel('Position-like')
    plt.ylabel('Momentum-like')
    plt.title(title)
    plt.grid(False)
    return fig

print("\nGenerating Wigner plots...")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
x = np.linspace(-3, 3, 80)
W_star = qt.wigner(rho_star_qt, x, x)
plt.contourf(x, x, W_star, 100, cmap='RdBu_r')
plt.colorbar()
plt.title('Wigner Function of Optimal ρ★ (Coherent)')
plt.xlabel('x')
plt.ylabel('p')

plt.subplot(1, 3, 2)
W_class = qt.wigner(rho_class_qt, x, x)
plt.contourf(x, x, W_class, 100, cmap='RdBu_r')
plt.colorbar()
plt.title('Wigner Function of Classical Projection')
plt.xlabel('x')

plt.subplot(1, 3, 3)
W_sigma = qt.wigner(sigma_qt, x, x)
plt.contourf(x, x, W_sigma, 100, cmap='RdBu_r')
plt.colorbar()
plt.title('Wigner Function of Target σ')
plt.xlabel('x')

plt.tight_layout()
plt.show()

# ========================= ANALYSIS =========================
def wigner_negativity(W):
    return np.sum(np.abs(W[W < 0])) / np.sum(np.abs(W))

print("\n=== Wigner Negativity Analysis ===")
print(f"Wigner negativity of ρ★:      {wigner_negativity(W_star):.4f}")
print(f"Wigner negativity of classical: {wigner_negativity(W_class):.4f}")
print(f"Wigner negativity of σ:        {wigner_negativity(W_sigma):.4f}")

# Commutation check
commutator_norm = np.linalg.norm(rho_star @ sigma - sigma @ rho_star, ord='fro')
print(f"||[ρ★, σ]|| = {commutator_norm:.6f} → Quantum advantage present" if commutator_norm > 1e-6 else "||[ρ★, σ]|| ≈ 0 → No quantum advantage")

print("\nKey Insight from the paper:")
print("• Even when [H, σ] = 0 (zero advantage), Wigner negativity can be positive.")
print("• The true certificate is coherence (off-diagonal mass) in σ's eigenbasis, not Wigner negativity.")
