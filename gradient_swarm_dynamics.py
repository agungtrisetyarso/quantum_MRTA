import numpy as np
from scipy.linalg import expm, logm, eigh
import matplotlib.pyplot as plt

# ================== Setup ==================
def random_hermitian(d, seed=42):
    np.random.seed(seed)
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    return (A + A.conj().T) / 2

def random_density(d, seed=42):
    np.random.seed(seed)
    A = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    rho = A @ A.conj().T
    return rho / np.trace(rho)

def compute_rho_star(H, sigma, lam=0.5):
    G = logm(sigma) - (1.0/lam) * H
    rho = expm(G)
    return rho / np.trace(rho)

# Simple natural gradient step (discrete mirror descent style)
def natural_gradient_step(rho, H, sigma, lam, eta=0.1):
    log_rho = logm(rho)
    grad = H + lam * (log_rho - logm(sigma))
    # Approximate natural gradient flow step
    update = -eta * (rho @ grad + grad @ rho) / 2
    rho_new = rho + update
    # Project back to density matrix
    rho_new = (rho_new + rho_new.conj().T) / 2
    eigvals, U = eigh(rho_new)
    rho_new = U @ np.diag(np.maximum(eigvals, 0)) @ U.conj().T
    return rho_new / np.trace(rho_new)

# ================== Simulation ==================
d = 4
H = random_hermitian(d, seed=123)
sigma = random_density(d, seed=456)
H += 0.4 * random_hermitian(d, seed=789)  # Non-commuting case

rho_star = compute_rho_star(H, sigma, lam=0.8)
rho_class = sigma  # Start from target (classical-like)

# Run swarm-like iterations (simulating distributed updates)
obj_values = []
coherence_values = []

rho = np.eye(d, dtype=complex) / d   # Initial state

for k in range(300):
    rho = natural_gradient_step(rho, H, sigma, lam=0.8, eta=0.15)
    
    obj = np.real(np.trace(H @ rho) + 0.8 * (np.trace(rho @ logm(rho)) - np.trace(rho @ logm(sigma))))
    obj_values.append(obj)
    
    # Coherence mass
    _, U = eigh(sigma)
    rho_basis = U.conj().T @ rho @ U
    off_diag = np.sum(np.abs(rho_basis - np.diag(np.diag(rho_basis))))
    coherence_values.append(off_diag)

# Plot convergence
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(obj_values, label='Natural Gradient Flow')
plt.axhline(y=np.real(np.trace(H @ rho_star)), color='r', linestyle='--', label=r'$f(\rho^\star)$')
plt.xlabel('Iteration')
plt.ylabel('Objective Value $f(\\rho)$')
plt.title('Convergence of Natural Gradient Swarm Dynamics')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(coherence_values, label='Coherence Mass')
plt.xlabel('Iteration')
plt.ylabel('$\\ell_1$ Off-diagonal Coherence')
plt.title('Emergence of Coherence')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('swarm_validation.png', dpi=300)
plt.show()

print(f"Final objective: {obj_values[-1]:.4f}")
print(f"Optimal objective: {np.real(np.trace(H @ rho_star)):.4f}")
print(f"Advantage gap Δ: {obj_values[-1] - np.real(np.trace(H @ rho_star)):.4f}")
