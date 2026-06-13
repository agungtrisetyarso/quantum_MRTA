# =============================================================================
# Colab Simulation: Information-Geometric Task Allocation with Natural Gradient Swarm Dynamics
# Small-scale example: n=2 agents, m=2 tasks (d=4)
# =============================================================================

!pip install numpy scipy matplotlib seaborn -q

import numpy as np
from scipy.linalg import expm, logm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("Simulation for Information-Geometric Task Allocation (n=2 agents, m=2 tasks)\n")

# ========================= PARAMETERS =========================
n_agents = 2
m_tasks = 2
d = m_tasks ** n_agents  # 4
lambda_reg = 0.5  # Regularization strength

# ========================= HELPER FUNCTIONS =========================
def random_density_matrix(dim, rank=None):
    """Generate a random full-rank density matrix."""
    if rank is None:
        rank = dim
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = A @ A.conj().T
    rho = rho / np.trace(rho)
    # Make it full rank by adding small noise if needed
    rho += 1e-4 * np.eye(dim)
    rho = rho / np.trace(rho)
    return rho

def umegaki_relative_entropy(rho, sigma):
    """Compute Umegaki relative entropy S(rho || sigma)."""
    # Avoid log(0) by adding small epsilon
    eps = 1e-12
    rho_log = logm(rho + eps * np.eye(len(rho)))
    sigma_log = logm(sigma + eps * np.eye(len(sigma)))
    return np.real(np.trace(rho @ (rho_log - sigma_log)))

def objective(rho, H, sigma, lambda_reg):
    """f(rho) = tr(H rho) + lambda * S(rho || sigma)"""
    return np.real(np.trace(H @ rho)) + lambda_reg * umegaki_relative_entropy(rho, sigma)

# ========================= SETUP =========================
# Random full-rank target state σ (desired correlations)
sigma = random_density_matrix(d)

# Random Hermitian cost operator H (costs + constraints)
H = np.random.randn(d, d) + 1j * np.random.randn(d, d)
H = (H + H.conj().T) / 2  # Make Hermitian

# ========================= CLOSED-FORM OPTIMAL STATE =========================
print("Computing closed-form optimal state ρ★ ...")
G = logm(sigma) - (1.0 / lambda_reg) * H
rho_star = expm(G)
rho_star = rho_star / np.trace(rho_star)

print(f"Trace of ρ★: {np.trace(rho_star):.6f} (should be ≈1)")
print(f"Objective value at ρ★: {objective(rho_star, H, sigma, lambda_reg):.6f}")

# ========================= CLASSICAL BASELINE (commuting with σ) =========================
# Diagonalize σ to get classical subspace
evals, evecs = np.linalg.eigh(sigma)
U = evecs  # σ = U diag(evals) U†

# Project to diagonal in σ's eigenbasis
rho_classical = np.diag(np.diag(U.conj().T @ rho_star @ U))
rho_classical = U @ rho_classical @ U.conj().T
rho_classical = rho_classical / np.trace(rho_classical)

v_star = objective(rho_star, H, sigma, lambda_reg)
v_sigma = objective(rho_classical, H, sigma, lambda_reg)
delta = v_sigma - v_star

print(f"\nQuantum advantage Δ = {delta:.6f}")
print(f"Coherence (off-diagonal ℓ1 mass) in σ basis: {np.sum(np.abs(U.conj().T @ rho_star @ U - np.diag(np.diag(U.conj().T @ rho_star @ U)))):.6f}")

# ========================= VISUALIZATION =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot σ
sns.heatmap(np.abs(sigma), ax=axes[0,0], cmap="viridis", annot=True, fmt=".2f")
axes[0,0].set_title("Target State |σ|")

# Plot H
sns.heatmap(np.abs(H), ax=axes[0,1], cmap="viridis", annot=True, fmt=".2f")
axes[0,1].set_title("Cost Operator |H|")

# Plot ρ★
sns.heatmap(np.abs(rho_star), ax=axes[1,0], cmap="viridis", annot=True, fmt=".2f")
axes[1,0].set_title("Optimal Coherent State |ρ★|")

# Plot classical projection
sns.heatmap(np.abs(rho_classical), ax=axes[1,1], cmap="viridis", annot=True, fmt=".2f")
axes[1,1].set_title("Classical Projection (commutes with σ)")

plt.tight_layout()
plt.show()

# ========================= SIMPLE NATURAL GRADIENT FLOW SIMULATION =========================
print("\n=== Simulating Natural Gradient Flow (simple gradient descent on manifold) ===")

# Start from maximally mixed state
rho_current = np.eye(d) / d
history = [objective(rho_current, H, sigma, lambda_reg)]

eta = 0.1  # learning rate
steps = 50

for step in range(steps):
    # Approximate gradient (for demo; in practice use quantum Fisher metric)
    grad = H + lambda_reg * (logm(rho_current) - logm(sigma))
    # Simple update with projection back to density matrix (not true natural gradient)
    rho_current = rho_current - eta * grad
    rho_current = (rho_current + rho_current.conj().T) / 2  # Hermitian
    rho_current = rho_current - np.trace(rho_current) * np.eye(d) / d  # Trace correction (approx)
    rho_current = np.maximum(rho_current, 0)  # Rough positivity
    rho_current = rho_current / np.trace(rho_current)
    history.append(objective(rho_current, H, sigma, lambda_reg))

plt.figure(figsize=(8, 5))
plt.plot(history, 'b-o', label='Objective value')
plt.axhline(y=v_star, color='r', linestyle='--', label='Optimal ρ★')
plt.xlabel('Iteration')
plt.ylabel('f(ρ)')
plt.title('Simple Gradient Flow toward Optimal Allocation')
plt.legend()
plt.grid(True)
plt.show()

print("\nSimulation complete! Key insights:")
print("• Closed-form ρ★ provides the global optimum")
print("• Positive Δ demonstrates quantum advantage when [H, σ] ≠ 0")
print("• Coherence (off-diagonals) carries the advantage")
