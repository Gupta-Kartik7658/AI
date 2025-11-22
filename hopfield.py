import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# --------------------------------------------------------
# 1. PLOT FUNCTION (with grid lines)
# --------------------------------------------------------
def plot_board(board, title="Board"):
    plt.figure(figsize=(5,5))
    plt.imshow(board, cmap="gray_r", extent=(0,8,8,0))

    # Proper grid on cell boundaries
    plt.xticks(np.arange(0,9,1))
    plt.yticks(np.arange(0,9,1))
    plt.grid(color='black', linewidth=1)

    # Force grid lines to appear above image
    plt.gca().set_axisbelow(False)

    plt.title(title)
    plt.show()


# --------------------------------------------------------
# 2. HOPFIELD WEIGHT MATRIX FOR 8×8 ROOK PROBLEM
# --------------------------------------------------------
def build_weights():
    n = 8
    N = n * n
    W = np.zeros((N, N))

    for i in range(n):
        for j in range(n):
            p = i * n + j
            for k in range(n):
                for l in range(n):
                    q = k * n + l
                    if p != q:
                        # Same row OR same column → weight -2
                        if i == k or j == l:
                            W[p, q] = -2
    return W

# --------------------------------------------------------
# 3. HOPFIELD UPDATE
# --------------------------------------------------------
def hopfield_update(x, W, theta=-1, steps=10):
    x = x.copy()
    N = len(x)
    for _ in range(steps):
        for p in range(N):
            h = np.dot(W[p], x) - theta
            x[p] = 1 if h > 0 else 0
    return x

# --------------------------------------------------------
# 4. USER PROVIDES INVALID 8×8 CONFIGURATION
# --------------------------------------------------------

# Example INVALID configuration (modify to test your own!)
initial_board = np.array([
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,1],
    [0,1,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0]
], dtype=int)

print("Initial board:\n", initial_board)
plot_board(initial_board, "Initial (Invalid) Configuration")

# --------------------------------------------------------
# 5. RUN HOPFIELD NETWORK
# --------------------------------------------------------
W = build_weights()

x0 = initial_board.flatten()
x_final = hopfield_update(x0, W, theta=-1, steps=20)

final_board = x_final.reshape((8,8))

print("Final board:\n", final_board)
plot_board(final_board, "Final (Hopfield Output) Configuration")
