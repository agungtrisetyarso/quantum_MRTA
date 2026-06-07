import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# MRTA Setup
n_robots = 4
n_tasks = 4
np.random.seed(42)
cost_matrix = np.random.randint(1, 10, (n_robots, n_tasks))
n_qubits = n_robots * n_tasks

dev = qml.device("default.qubit", wires=n_qubits, shots=4096)

@qml.qnode(dev)
def qaoa_circuit(params, p=2):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for layer in range(p):
        gamma = params[2*layer]
        beta = params[2*layer + 1]
        for i in range(n_qubits):          # Simplified placeholder Hamiltonian
            qml.RZ(gamma, wires=i)
        for i in range(n_qubits):
            qml.RX(beta, wires=i)
    return qml.probs(wires=range(n_qubits))

def approximate_entropy(probs):
    probs = np.array(probs)               # Ensure NumPy array
    probs = probs[probs > 1e-12]
    return -np.sum(probs * np.log2(probs))

# Simulation
n_runs = 30
entropies = []

for run in range(n_runs):
    params = np.random.uniform(0, 2*np.pi, 4)
    probs = qaoa_circuit(params)
    S = approximate_entropy(probs)
    entropies.append(S)

print(f"Mean Approximate Entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")

plt.figure(figsize=(10,6))
plt.hist(entropies, bins=12, alpha=0.8, color='royalblue', edgecolor='black')
plt.xlabel("Approximate von Neumann Entropy")
plt.ylabel("Frequency")
plt.title("Entropy Distribution in QAOA for MRTA (30 runs)")
plt.grid(True)
plt.show()
