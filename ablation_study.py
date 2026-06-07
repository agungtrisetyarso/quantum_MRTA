import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize

# ============================
# Setup
# ============================
n_robots = 4
n_tasks = 4
np.random.seed(42)
cost_matrix = np.random.randint(1, 10, (n_robots, n_tasks))
n_qubits = n_robots * n_tasks

print("Cost Matrix:")
print(cost_matrix)

# Hamiltonian
def build_mrta_hamiltonian(cost_matrix):
    coeffs = {}
    n_robots, n_tasks = cost_matrix.shape
    penalty = 20.0
    for i in range(n_robots):
        for j in range(n_tasks):
            idx = i * n_tasks + j
            coeffs[(idx,)] = float(cost_matrix[i, j])
    for i in range(n_robots):
        for j1, j2 in combinations(range(n_tasks), 2):
            idx1 = i * n_tasks + j1
            idx2 = i * n_tasks + j2
            coeffs[(idx1, idx2)] = penalty
    for j in range(n_tasks):
        for i1, i2 in combinations(range(n_robots), 2):
            idx1 = i1 * n_tasks + j
            idx2 = i2 * n_tasks + j
            coeffs[(idx1, idx2)] = penalty
    return coeffs

H_dict = build_mrta_hamiltonian(cost_matrix)

# QAOA Circuit
dev = qml.device("default.qubit", wires=n_qubits, shots=2048)

@qml.qnode(dev)
def qaoa_circuit(params, p=2, measure_cost=True):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for layer in range(p):
        gamma = params[2*layer]
        beta = params[2*layer + 1]
        for key, coeff in H_dict.items():
            if len(key) == 1:
                qml.RZ(2 * gamma * coeff, wires=key[0])
            elif len(key) == 2:
                i, j = key
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT(wires=[i, j])
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)
    if measure_cost:
        H = qml.Hamiltonian(list(H_dict.values()), 
                           [qml.PauliZ(k[0]) if len(k)==1 else qml.PauliZ(k[0]) @ qml.PauliZ(k[1]) 
                            for k in H_dict.keys()])
        return qml.expval(H)
    else:
        return qml.probs(wires=range(n_qubits))

def approximate_entropy(probs):
    probs = np.asarray(probs)
    probs = probs[probs > 1e-12]
    return -np.sum(probs * np.log2(probs))

# ============================
# Objective Functions
# ============================
def objective_with_entropy(params, p=2, lambda_ent=0.5):
    cost = qaoa_circuit(params, p=p, measure_cost=True)
    probs = qaoa_circuit(params, p=p, measure_cost=False)
    S = approximate_entropy(probs)
    return float(cost) + lambda_ent * S

def objective_without_entropy(params, p=2):
    cost = qaoa_circuit(params, p=p, measure_cost=True)
    return float(cost)

# ============================
# Ablation Study
# ============================
n_starts = 12
p = 2
results_with = []
results_without = []

print("Running Ablation Study...")

for start in range(n_starts):
    init_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    # With Entropy Regularization
    res_with = minimize(objective_with_entropy, init_params, args=(p, 0.5), 
                       method='COBYLA', tol=1e-4, options={'maxiter': 100})
    final_cost_with = qaoa_circuit(res_with.x, p=p, measure_cost=True)
    probs = qaoa_circuit(res_with.x, p=p, measure_cost=False)
    S_with = approximate_entropy(probs)
    results_with.append({'cost': float(final_cost_with), 'entropy': S_with})
    
    # Without Entropy Regularization
    res_without = minimize(objective_without_entropy, init_params, args=(p,), 
                          method='COBYLA', tol=1e-4, options={'maxiter': 100})
    final_cost_without = qaoa_circuit(res_without.x, p=p, measure_cost=True)
    probs = qaoa_circuit(res_without.x, p=p, measure_cost=False)
    S_without = approximate_entropy(probs)
    results_without.append({'cost': float(final_cost_without), 'entropy': S_without})

# ============================
# Results & Visualization
# ============================
costs_with = [r['cost'] for r in results_with]
costs_without = [r['cost'] for r in results_without]
ent_with = [r['entropy'] for r in results_with]
ent_without = [r['entropy'] for r in results_without]

print(f"\n=== ABLATION RESULTS ({n_starts} starts) ===")
print(f"With Entropy  → Best Cost: {np.min(costs_with):.2f} | Mean Entropy: {np.mean(ent_with):.3f}")
print(f"Without Entropy → Best Cost: {np.min(costs_without):.2f} | Mean Entropy: {np.mean(ent_without):.3f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.boxplot([costs_with, costs_without], labels=['With Entropy Reg.', 'Without'])
plt.ylabel("Final Cost (lower is better)")
plt.title("Cost Comparison")

plt.subplot(1, 2, 2)
plt.boxplot([ent_with, ent_without], labels=['With Entropy Reg.', 'Without'])
plt.ylabel("von Neumann Entropy")
plt.title("Entropy Comparison")

plt.tight_layout()
plt.show()
