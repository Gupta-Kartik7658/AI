import heapq
import random

class Node:
    def __init__(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g           # distance to root
        self.h = h           # estimated distance to goal
        self.f = g + h       # evaluation function

    def __lt__(self, other):
        return self.f < other.f  # compare on f, not g

def heuristic(state, goal_state):
    # number of misplaced tiles (ignoring blank)
    return sum(1 for i, tile in enumerate(state)
               if tile != 0 and tile != goal_state[i])

def get_successors(node, goal_state):
    successors = []
    index = node.state.index(0)
    quotient = index // 3
    remainder = index % 3
    moves = []

    # Row constrained moves
    if quotient == 0:
        moves = [3]
    if quotient == 1:
        moves = [-3, 3]
    if quotient == 2:
        moves = [-3]
    # Column constrained moves
    if remainder == 0:
        moves += [1]
    if remainder == 1:
        moves += [-1, 1]
    if remainder == 2:
        moves += [-1]

    for move in moves:
        im = index + move
        if 0 <= im < 9:
            new_state = list(node.state)
            new_state[index], new_state[im] = new_state[im], new_state[index]
            h_val = heuristic(new_state, goal_state)
            successor = Node(new_state, node, node.g + 1, h_val)
            successors.append(successor)
    return successors

def search_agent(start_state, goal_state):
    start_h = heuristic(start_state, goal_state)
    start_node = Node(start_state, None, g=0, h=start_h)

    frontier = []
    heapq.heappush(frontier, (start_node.f, start_node))
    visited = set()

    while frontier:
        _, node = heapq.heappop(frontier)
        tstate = tuple(node.state)
        if tstate in visited:
            continue
        visited.add(tstate)

        if node.state == goal_state:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]

        for successor in get_successors(node, goal_state):
            heapq.heappush(frontier, (successor.f, successor))
    return None

start_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
goal_state  = [0, 1, 2, 3, 4, 5, 6, 7, 8]

solution = search_agent(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")
