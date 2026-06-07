import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize

# ============================
# 1. MRTA Instance
# ============================
n_robots = 4
n_tasks = 4
np.random.seed(42)
cost_matrix = np.random.randint(1, 10, (n_robots, n_tasks))
n_qubits = n_robots * n_tasks

print("Cost Matrix:")
print(cost_matrix)

# ============================
# 2. Build Hamiltonian Coefficients
# ============================
def build_mrta_hamiltonian(cost_matrix):
    coeffs = {}
    n_robots, n_tasks = cost_matrix.shape
    penalty = 20.0
    
    # Objective
    for i in range(n_robots):
        for j in range(n_tasks):
            idx = i * n_tasks + j
            coeffs[(idx,)] = float(cost_matrix[i, j])
    
    # Constraints
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

# ============================
# 3. QAOA Circuit + Cost Measurement
# ============================
dev = qml.device("default.qubit", wires=n_qubits, shots=4096)

@qml.qnode(dev)
def qaoa_circuit(params, p=2, measure_cost=True):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(p):
        gamma = params[2*layer]
        beta = params[2*layer + 1]
        
        # Problem Hamiltonian
        for key, coeff in H_dict.items():
            if len(key) == 1:
                qml.RZ(2 * gamma * coeff, wires=key[0])
            elif len(key) == 2:
                i, j = key
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT(wires=[i, j])
        
        # Mixer
        for i in range(n_qubits):
            qml.RX(2 * beta, wires=i)
    
    if measure_cost:
        # Measure expectation value of H_P (objective + penalties)
        H = qml.Hamiltonian(list(H_dict.values()), [qml.PauliZ(i) if len(k)==1 else qml.PauliZ(k[0])@qml.PauliZ(k[1]) for k in H_dict.keys()])
        return qml.expval(H)
    else:
        return qml.probs(wires=range(n_qubits))

# ============================
# 4. Entropy from probabilities
# ============================
def approximate_entropy(probs):
    probs = np.asarray(probs)
    probs = probs[probs > 1e-12]
    return -np.sum(probs * np.log2(probs))

# ============================
# 5. Variational Optimization with Entropy Regularization
# ============================
def objective(params, p=2, lambda_ent=0.5):
    cost = qaoa_circuit(params, p=p, measure_cost=True)
    # Get probabilities for entropy
    probs = qaoa_circuit(params, p=p, measure_cost=False)
    S = approximate_entropy(probs)
    return cost + lambda_ent * S

# Run optimization for multiple starts
n_starts = 15
p = 2
results = []

for start in range(n_starts):
    init_params = np.random.uniform(0, 2*np.pi, 2*p)
    res = minimize(objective, init_params, args=(p, 0.5), method='COBYLA', tol=1e-4)
    
    final_cost = qaoa_circuit(res.x, p=p, measure_cost=True)
    probs = qaoa_circuit(res.x, p=p, measure_cost=False)
    S = approximate_entropy(probs)
    
    results.append({'cost': final_cost, 'entropy': S, 'success': res.success})

# ============================
# 6. Results
# ============================
costs = [r['cost'] for r in results]
entropies = [r['entropy'] for r in results]

print(f"\n=== Optimization Results ({n_starts} starts) ===")
print(f"Best Cost: {np.min(costs):.4f}")
print(f"Mean Entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(costs, bins=8, alpha=0.8, color='darkgreen')
plt.xlabel("Final Cost")
plt.ylabel("Frequency")
plt.title("Cost Distribution")

plt.subplot(1, 2, 2)
plt.hist(entropies, bins=8, alpha=0.8, color='royalblue')
plt.xlabel("von Neumann Entropy")
plt.ylabel("Frequency")
plt.title("Entropy Distribution")
plt.tight_layout()
plt.show()
