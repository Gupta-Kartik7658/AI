import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Optional

# -------------------------
# Helpers: splitting + permute
# -------------------------
def split_into_blocks(image: np.ndarray, n_blocks_side: int = 4) -> List[np.ndarray]:
    """Split HxW image into (n_blocks_side**2) square blocks (row-major order)."""
    h, w = image.shape
    assert h == w, "image must be square"
    block_size = h // n_blocks_side
    blocks = []
    for br in range(n_blocks_side):
        for bc in range(n_blocks_side):
            y0, y1 = br * block_size, (br + 1) * block_size
            x0, x1 = bc * block_size, (bc + 1) * block_size
            blocks.append(image[y0:y1, x0:x1].copy())
    return blocks

def permute_image_blocks(image: np.ndarray, perm_flat: List[int], n_blocks_side: int = 4) -> np.ndarray:
    """Build a permuted full image from a flat permutation (length n_blocks_side**2)."""
    n_blocks = n_blocks_side
    h, w = image.shape
    block_size = h // n_blocks
    perm = np.array(perm_flat).reshape((n_blocks, n_blocks))
    new_image = np.zeros_like(image)
    # original blocks in row-major order
    orig_blocks = []
    for br in range(n_blocks):
        for bc in range(n_blocks):
            y0 = br * block_size; y1 = y0 + block_size
            x0 = bc * block_size; x1 = x0 + block_size
            orig_blocks.append(image[y0:y1, x0:x1])
    # place according to perm
    for r in range(n_blocks):
        for c in range(n_blocks):
            idx = int(perm[r, c])
            block = orig_blocks[idx]
            y0 = r * block_size; y1 = y0 + block_size
            x0 = c * block_size; x1 = x0 + block_size
            new_image[y0:y1, x0:x1] = block
    return new_image

# -------------------------
# Edge error lookup
# 0 = up, 1 = right, 2 = left, 3 = down
# -------------------------
def compute_edge_errors_k(blocks: List[np.ndarray], k: int = 1) -> np.ndarray:
    """
    blocks: list of (block_size x block_size) numpy arrays, length n_blocks (should be 16)
    k: number of rows/cols to use from the edge (1 is only the boundary line)
    Returns: error[n_blocks, n_blocks, 4] with mapping:
             error[i,j,0] = mse(top_k_rows_of_i, bottom_k_rows_of_j)
             error[i,j,1] = mse(right_k_cols_of_i, left_k_cols_of_j)
             error[i,j,2] = mse(left_k_cols_of_i, right_k_cols_of_j)
             error[i,j,3] = mse(bottom_k_rows_of_i, top_k_rows_of_j)
    """
    n_blocks = len(blocks)
    edges = []
    for b in blocks:
        edges.append({
            0: b[0:k, :].astype(np.float64),   # up: top k rows
            3: b[-k:, :].astype(np.float64),  # down: bottom k rows
            1: b[:, -k:].astype(np.float64),  # right: right k cols
            2: b[:, 0:k].astype(np.float64)   # left: left k cols
        })

    error = np.zeros((n_blocks, n_blocks, 4), dtype=np.float64)
    for i in range(n_blocks):
        for j in range(n_blocks):
            if i == j:
                error[i, j, :] = np.inf
                continue
            # up of i vs down of j
            diff_up = edges[i][0] - edges[j][3]
            error[i, j, 0] = np.mean(diff_up.ravel() ** 2)
            # right of i vs left of j
            diff_right = edges[i][1] - edges[j][2]
            error[i, j, 1] = np.mean(diff_right.ravel() ** 2)
            # left of i vs right of j
            diff_left = edges[i][2] - edges[j][1]
            error[i, j, 2] = np.mean(diff_left.ravel() ** 2)
            # down of i vs up of j
            diff_down = edges[i][3] - edges[j][0]
            error[i, j, 3] = np.mean(diff_down.ravel() ** 2)
    return error

# -------------------------
# Energy function for a permutation state
# -------------------------
def energy_of_state(state: List[int], error_lookup: np.ndarray, n_blocks_side: int = 4) -> float:
    n = n_blocks_side
    total = 0.0
    for r in range(n):
        for c in range(n):
            pos = r * n + c
            b = int(state[pos])
            # right neighbor
            if c < n - 1:
                neighbor = int(state[r * n + (c + 1)])
                total += error_lookup[b, neighbor, 1]
            # down neighbor
            if r < n - 1:
                neighbor = int(state[(r + 1) * n + c])
                total += error_lookup[b, neighbor, 3]
    return float(total)

# -------------------------
# Node class
# -------------------------
class Node:
    def __init__(self, state: List[int], parent: Optional['Node'], energy: float):
        self.state = list(state)
        self.parent = parent
        self.energy = float(energy)

    def __repr__(self):
        return f"Node(energy={self.energy:.3f}, state={self.state})"

# -------------------------
# *** NEW *** Neighbor operator: swap two random blocks
# -------------------------
def swap_neighbor(state: List[int]) -> List[int]:
    """Pick two random distinct indices and swap the blocks at those positions."""
    n = len(state)
    i, j = random.sample(range(n), 2)  # Efficiently gets 2 unique indices
    new_state = state.copy()
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state

# -------------------------
# Simulated annealing search agent
# -------------------------
def simulated_annealing(
    init_state: List[int],
    error_lookup: np.ndarray,
    max_iters: int = 20000,
    Tm: float = 1000.0,
    alpha: float = 0.995,
    verbose: bool = True,
    print_every: int = 500
):
    current_state = list(init_state)
    ds = energy_of_state(current_state, error_lookup)
    best_state = current_state
    best_energy = ds
    
    energies = [ds]

    for it in range(1, max_iters + 1):
        # *** CHANGED *** Using the new neighbor operator
        candidate_state = swap_neighbor(current_state)
        dsnext = energy_of_state(candidate_state, error_lookup)

        E = ds - dsnext
        T = Tm * (alpha ** it)

        # Stop if temperature is negligible
        if T < 1e-6:
            break

        if E > 0: # Better state found
            accepted = True
            pE = 1.0
        else: # Worse state, may still accept
            pE = math.exp(E / T)
            accepted = random.random() < pE

        if accepted:
            current_state = candidate_state
            ds = dsnext
            
            # Keep track of the best state found so far
            if ds < best_energy:
                best_energy = ds
                best_state = current_state

        energies.append(best_energy) # Track the best energy over time

        if verbose and (it % print_every == 0):
            print(
                f"iter {it:5d} | best_E={best_energy:.2f} | curr_E={ds:.2f} | T={T:.3f} | pE={pE:.4f} | "
                f"{'ACCEPT' if accepted else 'reject'}"
            )

    final_node = Node(state=best_state, parent=None, energy=best_energy)
    return final_node, energies


# -------------------------
# Example usage (put your own image path)
# -------------------------
if __name__ == "__main__":
    # load the 512x512 image
    try:
        # NOTE: Update this path to your file location
        with open("C:/Users/hp/Desktop/College/SEM5/AI/image.txt", "r") as f:
            l = f.read().split(",")
        l = [int(x.strip().replace("[", "").replace("]", "")) for x in l if x.strip() != ""]
        x = [l[i * 512:(i + 1) * 512] for i in range(512)]
        arr = np.array(x, dtype=np.uint8)
    except FileNotFoundError:
        print("File not found. Creating a placeholder gradient image for demonstration.")
        x = np.linspace(0, 255, 512)
        y = np.linspace(0, 255, 512)
        xv, yv = np.meshgrid(x, y)
        arr = (xv + yv).astype(np.uint8)


    # split into 16 blocks
    blocks = split_into_blocks(arr, n_blocks_side=4)
    # compute lookup table (k=1 is recommended to start)
    error_lookup = compute_edge_errors_k(blocks, k=1)
    print("error_lookup shape:", error_lookup.shape)

    # Scramble the initial state to create a puzzle
    # Start from a solved state for demonstration, or scramble it.
    init_state = list(range(16))
    random.shuffle(init_state) # Start with a random permutation
    print(f"Starting with a random state: {init_state}")

    # run simulated annealing with updated parameters
    final_node, energies = simulated_annealing(
        init_state=init_state,
        error_lookup=error_lookup,
        max_iters=20000,
        Tm=1000.0,
        alpha=0.995,
        verbose=True,
        print_every=500
    )

    print("\nFinal node:", final_node)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the final permuted image
    final_image = permute_image_blocks(arr, final_node.state)
    axes[0].imshow(final_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Final Assembled Image (Energy={final_node.energy:.2f})")
    axes[0].axis("off")

    # Display the energy convergence plot
    axes[1].plot(energies)
    axes[1].set_title("Energy Convergence")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Best Energy")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()