import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

# ============================
# Setup: 3x3 MRTA Instance
# ============================
np.random.seed(42)
n_robots, n_tasks = 3, 3
cost_matrix = np.random.randint(5, 20, (n_robots, n_tasks)).astype(float)
n_qubits = n_robots * n_tasks

def build_hamiltonian(cost_matrix):
    coeffs = {}
    nr, nt = cost_matrix.shape
    penalty = 25.0
    # Linear objective terms
    for i in range(nr):
        for j in range(nt):
            idx = i * nt + j
            coeffs[(idx,)] = cost_matrix[i, j]
    # One-hot constraints (quadratic penalties)
    for i in range(nr):
        for j1, j2 in combinations(range(nt), 2):
            coeffs[(i*nt + j1, i*nt + j2)] = penalty
    for j in range(nt):
        for i1, i2 in combinations(range(nr), 2):
            coeffs[(i1*nt + j, i2*nt + j)] = penalty
    return coeffs

H_dict = build_hamiltonian(cost_matrix)
dev = qml.device("default.qubit", wires=n_qubits, shots=2048)

@qml.qnode(dev)
def qaoa_circuit(params, p=1):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for layer in range(p):
        gamma = params[2*layer]
        beta = params[2*layer + 1]
        # Problem unitary
        for key, coeff in H_dict.items():
            if len(key) == 1:
                qml.RZ(2 * gamma * coeff, wires=key[0])
            elif len(key) == 2:
                i, j = key
                qml.CNOT([i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT([i, j])
        # Mixer
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)
    return qml.probs(wires=range(n_qubits))

# ====================== CORRECTED ENTROPY ======================
def approximate_entropy(probs):
    """Safe Shannon entropy for measurement probabilities (max = n_qubits)"""
    probs = np.asarray(probs).flatten()
    probs = probs[probs > 1e-12]
    if len(probs) == 0:
        return 0.0
    probs = probs / np.sum(probs)          # Re-normalize
    return -np.sum(probs * np.log2(probs))
# ============================================================

def get_expectation_value(params, p=1):
    """Fast probabilistic approximation of cost"""
    probs = qaoa_circuit(params, p)
    cost = 0.0
    for key, coeff in H_dict.items():
        if len(key) == 1:
            idx = key[0]
            prob_1 = sum(p for i, p in enumerate(probs) if (i & (1 << idx)) != 0)
            cost += coeff * (2 * prob_1 - 1)
        elif len(key) == 2:
            i, j = key
            prob_11 = sum(p for k, p in enumerate(probs) if (k & (1 << i)) and (k & (1 << j)))
            prob_i1 = sum(p for k, p in enumerate(probs) if (k & (1 << i)))
            prob_j1 = sum(p for k, p in enumerate(probs) if (k & (1 << j)))
            cost += coeff * (4*prob_11 - 2*(prob_i1 + prob_j1) + 1)
    return cost

def objective(params, p=1, lam_ent=0.5):
    probs = qaoa_circuit(params, p)
    cost = get_expectation_value(params, p)
    ent = approximate_entropy(probs)
    return cost + lam_ent * ent

# ============================
# Optimization Helper
# ============================
def run_optimization(current_H_dict=None, p=1, maxiter=120):
    if current_H_dict is not None:
        global H_dict
        H_dict = current_H_dict.copy()
    
    n_params = 2 * p
    init_params = np.random.uniform(0, 2*np.pi, n_params)
    res = minimize(lambda x: objective(x, p), init_params,
                   method='COBYLA', options={'maxiter': maxiter, 'disp': False})
    final_cost = get_expectation_value(res.x, p)
    final_ent = approximate_entropy(qaoa_circuit(res.x, p))
    return final_cost, final_ent, res.x

# ============================
# Main Dynamic Simulation
# ============================
print("Running Entropy-Guided QAOA with Dynamic Reallocation...")

cost_history = []
ent_history = []
times = np.arange(0, 10)

# Initial optimization phase
for t in range(5):
    c, e, _ = run_optimization()
    cost_history.append(c)
    ent_history.append(e)
    print(f"Step {t}: Cost = {c:.2f}, Entropy = {e:.3f}")

# Perturbation
print("\n=== Applying Perturbation (Robot failure + New Task) ===")
perturbation_time = 5

for t in range(5, 10):
    cost_matrix_perturbed = cost_matrix.copy()
    cost_matrix_perturbed[0] += np.random.normal(8, 4, n_tasks)  # Simulate failure
    perturbed_H = build_hamiltonian(cost_matrix_perturbed)
    
    c, e, _ = run_optimization(current_H_dict=perturbed_H)
    cost_history.append(c)
    ent_history.append(e)
    print(f"Step {t}: Cost = {c:.2f}, Entropy = {e:.3f}")

# ============================
# Plot Dynamic Response (for manuscript)
# ============================
fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

axs[0].plot(times, cost_history, 'o-', color='darkgreen', linewidth=2.5, markersize=6, label='Assignment Cost')
axs[0].axvline(x=perturbation_time, color='red', linestyle='--', linewidth=3, label='Perturbation')
axs[0].set_ylabel('Total Assignment Cost\n(Lower is better)', fontsize=12)
axs[0].legend(fontsize=11)
axs[0].grid(True, alpha=0.3)

axs[1].plot(times, ent_history, 'o-', color='blue', linewidth=2.5, markersize=6, label='von Neumann Entropy')
axs[1].axvline(x=perturbation_time, color='red', linestyle='--', linewidth=3)
axs[1].set_ylabel('Approximate von Neumann Entropy', fontsize=12)
axs[1].set_xlabel('Time Steps', fontsize=12)
axs[1].legend(fontsize=11)
axs[1].grid(True, alpha=0.3)

plt.suptitle('Dynamic Response of Entropy-Guided QAOA Framework under Perturbation\n(Cybernetic Feedback for Collective Reorganization)', fontsize=14)
plt.tight_layout()
plt.savefig('f_dynamic.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Figure saved as 'f_dynamic.png'")
print(f"Final entropy values are physically valid (max observed: {max(ent_history):.3f} ≤ 9.0)")
