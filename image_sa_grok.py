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
# Edge error lookup (your requested direction mapping)
# 0 = up, 1 = right, 2 = left, 3 = down
# -------------------------
def compute_edge_errors_k(blocks: List[np.ndarray], k: int = 1) -> np.ndarray:
    """
    blocks: list of (block_size x block_size) numpy arrays, length n_blocks (should be 16)
    k: number of rows/cols to use from the edge (1 is only the boundary line)
    Returns: error[n_blocks, n_blocks, 4] with mapping:
            error[i,j,0] = mse(top_k_rows_of_i, bottom_k_rows_of_j)   (i.up vs j.down)
            error[i,j,1] = mse(right_k_cols_of_i, left_k_cols_of_j)   (i.right vs j.left)
            error[i,j,2] = mse(left_k_cols_of_i, right_k_cols_of_j)   (i.left vs j.right)
            error[i,j,3] = mse(bottom_k_rows_of_i, top_k_rows_of_j)   (i.down vs j.up)
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
                error[i, j, :] = np.inf  # discourage matching a block to itself
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
# - state is flat list length 16 (row-major positions)
# - evaluate each adjacency exactly once:
#   for every cell (r,c): check right neighbor (c<3) and down neighbor (r<3)
#   this ensures edges between neighbors are counted once
# -------------------------
def energy_of_state(state: List[int], error_lookup: np.ndarray, n_blocks_side: int = 4) -> float:
    """
    state: list length n_blocks_side**2 listing block indices placed at each position (row-major).
    error_lookup: array shape (n_blocks, n_blocks, 4) as returned above.
    """
    n = n_blocks_side
    total = 0.0
    for r in range(n):
        for c in range(n):
            pos = r * n + c
            b = int(state[pos])
            # right neighbor
            if c < n - 1:
                neighbor = int(state[r * n + (c + 1)])
                # add error for b's right matching neighbor's left -> error[b,neighbor,1]
                total += error_lookup[b, neighbor, 1]
            # down neighbor
            if r < n - 1:
                neighbor = int(state[(r + 1) * n + c])
                # add error for b's down matching neighbor's up -> error[b,neighbor,3]
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
# New neighbor operator: swap two random positions
# -------------------------
def swap_neighbor(state: List[int]) -> List[int]:
    """Pick two distinct random indices and swap them, return new state (copy)."""
    n = len(state)
    i = random.randrange(n)
    j = random.randrange(n)
    while j == i:
        j = random.randrange(n)
    new_state = state.copy()
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state, i, j

# -------------------------
# Updated simulated annealing (add best-tracking)
# -------------------------
def simulated_annealing(
    init_state: List[int],
    error_lookup: np.ndarray,
    max_iters: int = 10000,
    Tm: float = 5000.0,
    alpha: float = 0.999,
    verbose: bool = True,
    print_every: int = 500
):
    current_state = list(init_state)
    ds = energy_of_state(current_state, error_lookup)
    current_node = Node(state=current_state, parent=None, energy=ds)

    # Track the best seen
    best_node = current_node
    energies = [ds]

    for it in range(1, max_iters + 1):
        candidate_state, i, j = swap_neighbor(current_state)  # Use new neighbor
        dsnext = energy_of_state(candidate_state, error_lookup)

        E = ds - dsnext
        T = Tm * (alpha ** it)

        if E > 0:
            accepted = True
            pE = 1.0
        else:
            pE = math.exp(E / T)
            accepted = random.random() < pE

        if accepted:
            current_state = candidate_state
            ds = dsnext
            current_node = Node(state=current_state.copy(), parent=current_node, energy=ds)
            
            # Update best if improved
            if ds < best_node.energy:
                best_node = current_node

        energies.append(ds)

        if verbose and (it % print_every == 0):
            print(
                f"iter {it:4d} | ds={ds:.3f} | E={E:.3f} | T={T:.3f} | pE={pE:.4f} | "
                f"{'ACCEPT' if accepted else 'reject'}"
            )

    return best_node, energies  # Return best_node instead of current_node

# -------------------------
# small util: flat_to_matrix and matrix_to_flat
# -------------------------
def flat_to_matrix(perm_flat: List[int], n: int = 4) -> np.ndarray:
    return np.array(perm_flat).reshape((n, n))

def matrix_to_flat(perm_matrix: List[List[int]]) -> List[int]:
    return list(np.array(perm_matrix).reshape(-1))

# -------------------------
# Updated example usage
# -------------------------
if __name__ == "__main__":
    # load the 512x512 image (same method you used)
    with open("C:/Users/hp/Desktop/College/SEM5/AI/image.txt", "r") as f:
        l = f.read().split(",")
    l = [int(x.strip().replace("[", "").replace("]", "")) for x in l if x.strip() != ""]
    x = [l[i * 512:(i + 1) * 512] for i in range(512)]
    arr = np.array(x, dtype=np.uint8)

    # split into 16 blocks
    blocks = split_into_blocks(arr, n_blocks_side=4)
    
    # compute lookup table with k=3
    error_lookup = compute_edge_errors_k(blocks, k=3)
    print("error_lookup shape:", error_lookup.shape)

    # initial state: shuffled permutation
    init_state = list(range(16))
    random.shuffle(init_state)  # Crucial: start shuffled

    # run simulated annealing with updated params
    best_node, energies = simulated_annealing(
        init_state=init_state,
        error_lookup=error_lookup,
        max_iters=10000,
        Tm=5000.0,
        alpha=0.999,
        verbose=True,
        print_every=500
    )

    print("\nBest node:", best_node)

    # Display the best permuted image
    best_image = permute_image_blocks(arr, best_node.state)
    plt.figure(figsize=(6, 6))
    plt.imshow(best_image, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Best reconstructed image (energy={best_node.energy:.3f})")
    plt.axis("off")
    plt.show()