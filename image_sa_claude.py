import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import List, Optional, Tuple

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
# Edge error lookup with improved k handling
# -------------------------
def compute_edge_errors_k(blocks: List[np.ndarray], k: int = 2) -> np.ndarray:
    """
    blocks: list of (block_size x block_size) numpy arrays, length n_blocks (should be 16)
    k: number of rows/cols to use from the edge (2-3 is often better than 1)
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
# Energy function (unchanged)
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
                total += error_lookup[b, neighbor, 1]
            # down neighbor
            if r < n - 1:
                neighbor = int(state[(r + 1) * n + c])
                total += error_lookup[b, neighbor, 3]
    return float(total)

# -------------------------
# Node class (unchanged)
# -------------------------
class Node:
    def __init__(self, state: List[int], parent: Optional['Node'], energy: float):
        self.state = list(state)
        self.parent = parent
        self.energy = float(energy)

    def __repr__(self):
        return f"Node(energy={self.energy:.3f}, state={self.state})"

# -------------------------
# Improved neighbor operators
# -------------------------
def reverse_subsequence_neighbor(state: List[int]) -> Tuple[List[int], int, int]:
    """Pick random i<j and reverse state[i:j+1], return new state (copy)."""
    n = len(state)
    i = random.randrange(0, n - 1)
    j = random.randrange(i + 1, n)
    new_state = state.copy()
    new_state[i : j + 1] = reversed(new_state[i : j + 1])
    return new_state, i, j

def swap_neighbor(state: List[int]) -> Tuple[List[int], int, int]:
    """Swap two random positions."""
    n = len(state)
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        j = (j + 1) % n
    new_state = state.copy()
    new_state[i], new_state[j] = new_state[j], new_state[i]
    return new_state, i, j

def relocate_neighbor(state: List[int]) -> Tuple[List[int], int, int]:
    """Move one element to a different position."""
    n = len(state)
    i = random.randrange(n)  # source
    j = random.randrange(n)  # destination
    if i == j:
        j = (j + 1) % n
    new_state = state.copy()
    element = new_state.pop(i)
    new_state.insert(j, element)
    return new_state, i, j

def mixed_neighbor(state: List[int]) -> Tuple[List[int], int, int]:
    """Randomly choose between different neighbor operators."""
    op = random.choice([0, 1, 2])
    if op == 0:
        return swap_neighbor(state)
    elif op == 1:
        return reverse_subsequence_neighbor(state)
    else:
        return relocate_neighbor(state)

# -------------------------
# Improved simulated annealing with better cooling schedule
# -------------------------
def simulated_annealing_improved(
    init_state: List[int],
    error_lookup: np.ndarray,
    max_iters: int = 10000,
    T_init: float = None,
    T_final: float = 0.01,
    cooling_schedule: str = 'exponential',  # 'exponential', 'linear', 'logarithmic'
    alpha: float = 0.99,
    neighbor_op: str = 'mixed',  # 'reverse', 'swap', 'relocate', 'mixed'
    verbose: bool = True,
    restart_threshold: int = 1000  # restart if no improvement for this many iterations
):
    """
    Improved simulated annealing with better parameters and cooling schedules.
    """
    # Auto-determine initial temperature if not provided
    if T_init is None:
        # Sample some random moves to estimate energy differences
        sample_size = min(100, max_iters // 10)
        energy_diffs = []
        current_energy = energy_of_state(init_state, error_lookup)
        
        for _ in range(sample_size):
            if neighbor_op == 'reverse':
                candidate_state, _, _ = reverse_subsequence_neighbor(init_state)
            elif neighbor_op == 'swap':
                candidate_state, _, _ = swap_neighbor(init_state)
            elif neighbor_op == 'relocate':
                candidate_state, _, _ = relocate_neighbor(init_state)
            else:  # mixed
                candidate_state, _, _ = mixed_neighbor(init_state)
            
            candidate_energy = energy_of_state(candidate_state, error_lookup)
            energy_diffs.append(abs(candidate_energy - current_energy))
        
        # Set initial temperature to accept ~80% of moves initially
        avg_diff = np.mean(energy_diffs) if energy_diffs else 1000
        T_init = -avg_diff / math.log(0.8)
        print(f"Initial temperature: {T_init:.2f}")
    
    current_state = list(init_state)
    ds = energy_of_state(current_state, error_lookup)
    current_node = Node(state=current_state, parent=None, energy=ds)
    best_node = Node(state=current_state, parent=None, energy=ds)
    
    accepted_count = 0
    no_improvement_count = 0
    
    print(f"Initial energy: {ds:.3f}")
    
    for it in range(1, max_iters + 1):
        # Choose neighbor operator
        if neighbor_op == 'reverse':
            candidate_state, i, j = reverse_subsequence_neighbor(current_state)
        elif neighbor_op == 'swap':
            candidate_state, i, j = swap_neighbor(current_state)
        elif neighbor_op == 'relocate':
            candidate_state, i, j = relocate_neighbor(current_state)
        else:  # mixed
            candidate_state, i, j = mixed_neighbor(current_state)
        
        dsnext = energy_of_state(candidate_state, error_lookup)
        
        # Energy difference (positive means improvement)
        delta_E = ds - dsnext
        
        # Temperature schedule
        if cooling_schedule == 'exponential':
            T = T_init * (alpha ** it)
        elif cooling_schedule == 'linear':
            T = T_init * (1 - it / max_iters)
            T = max(T, T_final)
        elif cooling_schedule == 'logarithmic':
            T = T_init / math.log(it + 1)
        else:  # default to exponential
            T = T_init * (alpha ** it)
        
        T = max(T, T_final)  # Ensure minimum temperature
        
        # Acceptance probability
        if delta_E > 0:
            # Improvement - always accept
            pE = 1.0
            accepted = True
        else:
            # Deterioration - accept with probability
            pE = math.exp(delta_E / T) if T > 0 else 0.0
            accepted = random.random() < pE
        
        if accepted:
            current_state = candidate_state
            ds = dsnext
            current_node = Node(state=current_state.copy(), parent=current_node, energy=ds)
            accepted_count += 1
            
            # Print only accepted moves
            if verbose:
                print(f"iter {it:5d} | energy={ds:.3f} | ΔE={delta_E:.3f} | T={T:.3f} | ACCEPTED")
            
            # Track best solution
            if ds < best_node.energy:
                best_node = Node(state=current_state.copy(), parent=None, energy=ds)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1
        
        # Restart mechanism
        if no_improvement_count >= restart_threshold and it < max_iters * 0.8:
            print(f"Restarting at iteration {it} (no improvement for {no_improvement_count} steps)")
            current_state = random.sample(range(16), 16)
            ds = energy_of_state(current_state, error_lookup)
            no_improvement_count = 0
    
    final_acceptance_rate = accepted_count / max_iters
    print(f"\nFinal Results:")
    print(f"Accepted moves: {accepted_count}/{max_iters} ({final_acceptance_rate:.3f})")
    print(f"Best energy: {best_node.energy:.3f}")
    
    return best_node

# -------------------------
# Utility functions (unchanged)
# -------------------------
def flat_to_matrix(perm_flat: List[int], n: int = 4) -> np.ndarray:
    return np.array(perm_flat).reshape((n, n))

def matrix_to_flat(perm_matrix: List[List[int]]) -> List[int]:
    return list(np.array(perm_matrix).reshape(-1))

# -------------------------
# Enhanced example usage with multiple runs
# -------------------------
def solve_puzzle_multiple_runs(image_path: str = None, n_runs: int = 3):
    """Run the solver multiple times and return the best result."""
    
    # Load image (modify this section for your image loading method)
    if image_path:
        # If you have an actual image file
        from PIL import Image
        img = Image.open(image_path).convert('L')
        arr = np.array(img, dtype=np.uint8)
    else:
        # Your existing method - adjust path as needed
        try:
            with open("C:/Users/hp/Desktop/College/SEM5/AI/image.txt", "r") as f:
                l = f.read().split(",")
            l = [int(x.strip().replace("[", "").replace("]", "")) for x in l if x.strip() != ""]
            x = [l[i * 512:(i + 1) * 512] for i in range(512)]
            arr = np.array(x, dtype=np.uint8)
        except FileNotFoundError:
            print("Image file not found. Creating a test pattern.")
            arr = create_test_image()
    
    # Split into blocks and compute error lookup
    blocks = split_into_blocks(arr, n_blocks_side=4)
    error_lookup = compute_edge_errors_k(blocks, k=4)  # Using k=2 for better edge matching
    
    best_result = None
    best_energy = float('inf')
    
    for run in range(n_runs):
        print(f"\n{'='*30} Run {run + 1}/{n_runs} {'='*30}")
        
        # Start with random permutation for diversity
        if run == 0:
            init_state = list(range(16))  # Identity for first run
        else:
            init_state = random.sample(range(16), 16)  # Random for others
        
        # Run simulated annealing
        result = simulated_annealing_improved(
            init_state=init_state,
            error_lookup=error_lookup,
            max_iters=8000,
            cooling_schedule='exponential',
            alpha=0.995,
            neighbor_op='mixed',
            verbose=True,
            restart_threshold=2000
        )
        
        if result.energy < best_energy:
            best_energy = result.energy
            best_result = (result, arr)
    
    return best_result

def create_test_image(size: int = 512) -> np.ndarray:
    """Create a test image with clear block boundaries for testing."""
    arr = np.zeros((size, size), dtype=np.uint8)
    block_size = size // 4
    
    # Create a pattern with different intensities for each block
    for i in range(4):
        for j in range(4):
            y0, y1 = i * block_size, (i + 1) * block_size
            x0, x1 = j * block_size, (j + 1) * block_size
            # Create gradient pattern
            for y in range(y0, y1):
                for x in range(x0, x1):
                    arr[y, x] = (i * 60 + j * 15 + (y-y0) + (x-x0)) % 255
    
    return arr

if __name__ == "__main__":
    # Run multiple attempts to find best solution
    result = solve_puzzle_multiple_runs(n_runs=3)
    
    if result:
        best_node, original_image = result
        
        print(f"\n{'='*50}")
        print(f"FINAL SOLUTION")
        print(f"{'='*50}")
        print(f"Best energy: {best_node.energy:.3f}")
        print(f"Final permutation: {best_node.state}")
        
        # Create the final reconstructed image
        final_image = permute_image_blocks(original_image, best_node.state)
        
        # Display only original and final images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Final reconstructed image
        axes[1].imshow(final_image, cmap="gray", vmin=0, vmax=255)
        axes[1].set_title(f"Reconstructed (Energy: {best_node.energy:.3f})")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Print final block arrangement
        perm_matrix = flat_to_matrix(best_node.state)
        print("\nFinal block arrangement:")
        for i in range(4):
            print([perm_matrix[i, j] for j in range(4)])