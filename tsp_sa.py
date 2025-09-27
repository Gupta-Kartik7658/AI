"""
Simple Travelling Salesman Problem solver using Simulated Annealing (2-opt neighbor).
- Uses a logistic acceptance probability like in the MATLAB example you provided.
- Produces an initial random tour and attempts to improve it by reversing random segments (2-opt).
- Minimal, well-commented, and easy to adapt for related permutation problems (like jigsaw placement
  where the energy is a measure of mismatch between adjacent pieces).

How to run:
    python tsp_simulated_annealing.py

The script will display two subplots: the initial random tour and the final tour found by SA.
"""

import numpy as np
import matplotlib.pyplot as plt


def pairwise_distances(coords):
    """Return an (N,N) distance matrix for N points in coords (shape (N,2))."""
    # Broadcasting: coords[:, None, :] - coords[None, :, :] gives shape (N,N,2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))
    return D


def path_cost_tour(s, D):
    """Compute total length (cost) of tour `s` using distance matrix D.

    s is a 1-D array-like of indices, length N. The tour is closed (returns to start).
    """
    N = len(s)
    cost = 0.0
    for k in range(N):
        a = s[k]
        b = s[(k + 1) % N]  # next city, wrap around using modulo
        cost += D[a, b]
    return cost


def two_opt_neighbor(s):
    """Return a new permutation that results from reversing a random segment of s.

    This implements the "second neighbor operator" used in the MATLAB code.
    """
    N = len(s)
    i, j = np.random.choice(N, size=2, replace=False)
    if i > j:
        i, j = j, i
    new_s = s.copy()
    # reverse the slice between i and j (inclusive)
    new_s[i : j + 1] = new_s[i : j + 1][::-1]
    return new_s


def simulated_annealing(coords, T0=1000.0, iter_max=50000, seed=100):
    """Run simulated annealing and return the final tour and history.

    - coords: numpy array of shape (N,2) giving point coordinates
    - T0: initial temperature scale (Tm in MATLAB)
    - iter_max: number of iterations
    - seed: RNG seed for reproducibility

    Returns: (s, ds, s_init, d_history, p_history, D)
    """
    np.random.seed(seed)

    N = coords.shape[0]
    D = pairwise_distances(coords)

    # initial random tour
    s = np.random.permutation(N)
    s_init = s.copy()
    ds = path_cost_tour(s, D)

    d_history = [ds]  # store cost at each iteration
    p_history = []    # store acceptance-probabilities computed

    for i in range(1, iter_max + 1):
        snext = two_opt_neighbor(s)
        dsnext = path_cost_tour(snext, D)

        # energy difference (positive when snext is better - smaller cost)
        E = ds - dsnext

        # cooling schedule like MATLAB: T = Tm / i
        T = T0 / i

        # logistic acceptance probability (mirrors the original MATLAB code)
        # note: when E is large positive => pE -> 1; when large negative => pE -> 0
        pE = 1.0 / (1.0 + np.exp(-E / T))

        # accept if improvement or with probability pE
        if E > 0 or np.random.rand() < pE:
            s = snext
            ds = dsnext

        d_history.append(ds)
        p_history.append(pE)

    return s, ds, s_init, d_history, p_history, D


def plot_tour(coords, s, title=None):
    """Plot a closed tour given by permutation s on coordinates coords."""
    tour = np.concatenate([s, [s[0]]])  # append first to close loop
    plt.plot(coords[tour, 0], coords[tour, 1], '-o', markersize=3)
    if title:
        plt.title(title)
    plt.axis('equal')


if __name__ == '__main__':
    # Example run with 200 nodes (reduce N if your machine is slow)
    N = 200
    coords = np.random.rand(N, 2)  # random points in unit square

    # Run SA (reduce iter_max to make it faster during testing)
    s_final, cost_final, s_init, d_hist, p_hist, D = simulated_annealing(
        coords, T0=1000.0, iter_max=20000, seed=100
    )
    print(f'Final tour cost: {cost_final:.2f}')

    # Plot initial and final tours side-by-side
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_tour(coords, s_init, title='Initial tour')

    plt.subplot(1, 2, 2)
    plot_tour(coords, s_final, title=f'Final tour (cost={cost_final:.2f})')

    plt.tight_layout()
    plt.show()
