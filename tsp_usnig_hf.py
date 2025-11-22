import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. Generate 10 cities
# --------------------------------------------------------
np.random.seed(10)
n = 10
cities = np.random.rand(n, 2)

# Distance matrix
D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        D[i, j] = np.linalg.norm(cities[i] - cities[j])

# --------------------------------------------------------
# 2. Hopfield–Tank Weight Matrix (Textbook Version)
# --------------------------------------------------------
gamma = 500
N = n * n
theta = np.full(N, -gamma/2)
W = np.zeros((N, N))

def idx(i, k):
    return i*n + k  # neuron representing "city i at position k"

# Build weights
for i in range(n):
    for k in range(n):
        p = idx(i, k)
        for j in range(n):
            for l in range(n):
                q = idx(j, l)

                # Distance term (column k → k+1)
                if l == (k+1) % n:
                    W[p, q] -= D[i, j]

                # Rook constraints (same city OR same position)
                if (i == j) or (k == l):
                    W[p, q] -= gamma

# --------------------------------------------------------
# 3. Good random initial configuration (valid permutation)
# --------------------------------------------------------
x0 = np.zeros(N)
perm = np.random.permutation(n)
for k in range(n):
    x0[idx(perm[k], k)] = 1

# --------------------------------------------------------
# 4. Hopfield Update
# --------------------------------------------------------
def hopfield_update(x, steps=150):
    x = x.copy()
    for _ in range(steps):
        for neuron in np.random.permutation(N):
            h = np.dot(W[neuron], x) - theta[neuron]
            x[neuron] = 1 if h > 0 else 0
    return x

x_final = hopfield_update(x0, steps=150)
state = x_final.reshape((n, n))

# --------------------------------------------------------
# 5. Extract tour (argmax per position)
# --------------------------------------------------------
tour = []
for k in range(n):
    tour.append(np.argmax(state[:, k]))

# --------------------------------------------------------
# 6. REPAIR TOUR (Ensure Hamiltonian Cycle)
# --------------------------------------------------------
# remove duplicates while preserving order
seen = set()
fixed = []
for c in tour:
    if c not in seen:
        fixed.append(c)
        seen.add(c)

# add missing cities
for c in range(n):
    if c not in seen:
        fixed.append(c)

# close the cycle
fixed.append(fixed[0])

print("\nFinal repaired tour:", fixed)

# --------------------------------------------------------
# 7. Plot initial and final tours
# --------------------------------------------------------

# --------------------------------------------------------
# 7. Plot initial and final tours
# --------------------------------------------------------

# --- Initial: Only city scatter plot, NO LINES ---
plt.figure(figsize=(6,6))
plt.scatter(cities[:,0], cities[:,1], c='blue', s=80)
for i,(x,y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, color='red')

plt.title("Initial City Layout (Unconnected)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# --- Final Hopfield + Repaired Tour ---
plt.figure(figsize=(6,6))
plt.scatter(cities[:,0], cities[:,1], c='blue', s=80)
for i,(x,y) in enumerate(cities):
    plt.text(x, y, str(i), fontsize=12, color='red')

# draw final connected tour
for k in range(n):
    c1, c2 = fixed[k], fixed[k+1]
    plt.plot([cities[c1,0], cities[c2,0]],
             [cities[c1,1], cities[c2,1]],
             'k-', linewidth=2)

plt.title("Final Hopfield Tour (Connected Path)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
