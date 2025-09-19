from collections import deque

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

def get_successors(node):
    state = node.state
    successors = []
    n = len(state)
    pos = state.index("_")   # empty stone

    # East rabbit moves/jumps
    if pos - 1 >= 0 and state[pos - 1] == "E":
        new_state = state.copy()
        new_state[pos], new_state[pos - 1] = new_state[pos - 1], "_"
        successors.append(Node(new_state, node))

    if pos - 2 >= 0 and state[pos - 2] == "E" and state[pos - 1] in ("E", "W"):
        new_state = state.copy()
        new_state[pos], new_state[pos - 2] = new_state[pos - 2], "_"
        successors.append(Node(new_state, node))

    # West rabbit moves/jumps
    if pos + 1 < n and state[pos + 1] == "W":
        new_state = state.copy()
        new_state[pos], new_state[pos + 1] = new_state[pos + 1], "_"
        successors.append(Node(new_state, node))

    if pos + 2 < n and state[pos + 2] == "W" and state[pos + 1] in ("E", "W"):
        new_state = state.copy()
        new_state[pos], new_state[pos + 2] = new_state[pos + 2], "_"
        successors.append(Node(new_state, node))

    return successors

def bfs(start_state, goal_state):
    start_node = Node(start_state)
    queue = deque([(start_node, 0)])
    visited = set()
    nodes_explored = 0
    current_depth = 0
    c = 0
    while queue:
        depth = queue[0][1]  # depth of the first node in queue
        if depth != current_depth:
            current_depth = depth
            print(f"\nFrontier at depth {current_depth}:")
            print(["".join(n.state) for (n, d) in queue])
            c+=len(["".join(n.state) for (n, d) in queue])
   
        node, depth = queue.popleft()
        tstate = tuple(node.state)
        if tstate in visited:
            continue
        visited.add(tstate)
        nodes_explored += 1

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("\nTotal nodes explored", nodes_explored)
            return path[::-1]

        for successor in get_successors(node):
            queue.append((successor, depth + 1))
    return None

def dfs(start_state, goal_state):
    start_node = Node(start_state)
    stack = [(start_node, 0)]  # (node, depth)
    visited = set()
    nodes_explored = 0
    current_depth = 0

    while stack:
        # we’ll look at the top of the stack to know current depth
        depth = stack[-1][1]
        if depth != current_depth:
            current_depth = depth
            print(f"\nFrontier at depth {current_depth}:")
            print(["".join(n.state) for (n, d) in stack])

        node, depth = stack.pop()  # LIFO
        tstate = tuple(node.state)
        if tstate in visited:
            continue
        visited.add(tstate)
        nodes_explored += 1

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("\nTotal nodes explored (DFS):", nodes_explored)
            return path[::-1]

        # push successors onto stack
        for successor in reversed(get_successors(node)):  
            stack.append((successor, depth + 1))
    return None

# initial and goal states
start_state = ["E", "E", "E", "_", "W", "W", "W"]
goal_state  = ["W", "W", "W", "_", "E", "E", "E"]

solution = bfs(start_state, goal_state)
if solution:
    print("\nSolution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
