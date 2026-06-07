import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from scipy.optimize import minimize
import os

# ============================
# Load Real TWPC-MRTA Data
# ============================
data_path = "Data"  # Adjust if needed

# Example: Load one instance (Uniform or Cluster)
# Files are usually CSV with robot-task costs, time windows, etc.
instance_file = "uniform_4_4.csv"  # Change to actual filename after ls

if os.path.exists(f"Data/{instance_file}"):
    df = pd.read_csv(f"Data/{instance_file}")
    print("Loaded instance:", instance_file)
    print(df.head())
else:
    print("Please check available files in Data/ folder")
    # Fallback to synthetic if real file not found
    n_robots, n_tasks = 4, 4
    cost_matrix = np.random.randint(1, 10, (n_robots, n_tasks))
else:
    # Example parsing (adapt to actual file structure)
    # Assuming columns like 'robot', 'task', 'cost', etc.
    n_robots = df['robot'].nunique() if 'robot' in df.columns else 4
    n_tasks = df['task'].nunique() if 'task' in df.columns else 4
    cost_matrix = np.zeros((n_robots, n_tasks))
    # Fill cost_matrix based on your file format

n_qubits = n_robots * n_tasks

# ============================
# Hamiltonian & QAOA (same as before)
# ============================
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
# Ablation Study
# ============================
def objective_with(params, p=2, lambda_ent=0.5):
    cost = qaoa_circuit(params, p=p, measure_cost=True)
    probs = qaoa_circuit(params, p=p, measure_cost=False)
    S = approximate_entropy(probs)
    return float(cost) + lambda_ent * S

def objective_without(params, p=2):
    return float(qaoa_circuit(params, p=p, measure_cost=True))

n_starts = 10
p = 2
results_with = []
results_without = []

for start in range(n_starts):
    init_params = np.random.uniform(0, 2*np.pi, 2*p)
    
    # With Entropy
    res_with = minimize(objective_with, init_params, args=(p, 0.5), method='COBYLA', tol=1e-4, options={'maxiter': 80})
    results_with.append({'cost': float(qaoa_circuit(res_with.x, p=p, measure_cost=True)), 
                        'entropy': approximate_entropy(qaoa_circuit(res_with.x, p=p, measure_cost=False))})
    
    # Without Entropy
    res_without = minimize(objective_without, init_params, args=(p,), method='COBYLA', tol=1e-4, options={'maxiter': 80})
    results_without.append({'cost': float(qaoa_circuit(res_without.x, p=p, measure_cost=True)), 
                           'entropy': approximate_entropy(qaoa_circuit(res_without.x, p=p, measure_cost=False))})

# ============================
# Visualization
# ============================
costs_with = [r['cost'] for r in results_with]
costs_without = [r['cost'] for r in results_without]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.boxplot([costs_with, costs_without], labels=['With Entropy', 'Without'])
plt.ylabel("Final Cost (lower better)")
plt.title("Ablation on Real TWPC-MRTA Instance")

plt.subplot(1, 2, 2)
plt.boxplot([[r['entropy'] for r in results_with], [r['entropy'] for r in results_without]], 
            labels=['With Entropy', 'Without'])
plt.ylabel("von Neumann Entropy")
plt.title("Entropy Comparison")
plt.tight_layout()
plt.show()

print("Ablation completed on real benchmark data.")
