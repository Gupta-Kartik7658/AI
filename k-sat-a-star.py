import heapq
import numpy as np
import random
import math

# ---------- Node and helpers ----------

class Node:
    def __init__(self, state, bias, frequency_table, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.bias = bias
        self.frequency_table = frequency_table
        self.g = g
        self.h = h
        self.f = g + h  # for A*
    def __lt__(self, other):
        return self.f < other.f

def heuristic1(state, clauses):
    h = 0 
    for clause in clauses: 
        if any((state[abs(lit)-1] ^ (lit < 0)) for lit in clause): 
            h += 1 
    return h 


def heuristic2(state, bias, frequency_table):
    h = 0
    for i in range(len(state)):
        s_i = 1 if state[i] == 1 else -1
        h += s_i * (bias[i+1] + 0.5 * np.log(1 + frequency_table[i+1]))
    return h

def isGoalState(state, clauses):
    for clause in clauses:
        if not any((state[abs(lit)-1] ^ (lit < 0)) for lit in clause):
            return False
    return True

def get_successors(node):
    successors = []
    for i in range(len(node.state)):
        new_state = list(node.state)
        new_state[i] = 1 - new_state[i]  # flip the bit
        new_h = heuristic2(new_state, node.bias, node.frequency_table)
        successors.append(Node(new_state, node.bias, node.frequency_table,
                               parent=node, g=node.g + 1, h=new_h))
    return successors

def generate_clauses(k, m, n):
    clauses = []
    for _ in range(m):
        clause = []
        vars_in_clause = random.sample(range(1, n+1), k)
        for var in vars_in_clause:
            sign = random.choice([1, -1])
            clause.append(sign * var)
        clauses.append(clause)
    return clauses

# ---------- A* search (alpha=beta=1) ----------

import itertools
counter = itertools.count()

def astar_search(start_state, clauses, bias, frequency_table):
    start_h = heuristic2(start_state, bias, frequency_table)
    start_node = Node(start_state, bias, frequency_table, g=0, h=start_h)

    frontier = []
    heapq.heappush(frontier, (start_node.f, next(counter), start_node))
    visited = set()
    step = 0

    while frontier:
        print("\n" + "="*40)
        print(f"[A*] Step {step}: Frontier")
        for f, _, n in frontier:
            print(f"  State={n.state} g={n.g} h={n.h:.3f} f={n.f:.3f}")
        step += 1

        _, _, node = heapq.heappop(frontier)
        print(f"\nChosen node: {node.state} g={node.g} h={node.h:.3f} f={node.f:.3f}")

        tstate = tuple(node.state)
        if tstate in visited:
            print("Already visited. Skipping.")
            continue
        visited.add(tstate)
        print(f"Visited states: {len(visited)}")

        if isGoalState(node.state, clauses):
            print("\nGoal reached!")
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]

        successors = get_successors(node)
        print(f"Expanding {len(successors)} successors:")
        for s in successors:
            print(f"  Successor={s.state} g={s.g} h={s.h:.3f} f={s.f:.3f}")
            heapq.heappush(frontier, (s.f, next(counter), s))

    print("\nNo solution found.")
    return None





k, m, n = 3, 4, 5
clauses = [[-1, 5, -3], [-4, 3, 5], [4, 3, -5], [-4, -5, -3]]
print("Clauses:", clauses)

bias = {}
frequency_table = {}
for clause in clauses:
    for var in clause:
        if abs(var) not in bias:
            bias[abs(var)] = int(var/abs(var))
            frequency_table[abs(var)] = 1
        else:
            bias[abs(var)] += int(var/abs(var))
            frequency_table[abs(var)] += 1
for i in range(1, n+1):
    bias.setdefault(i, 0)
    frequency_table.setdefault(i, 0)

initial_state = [1]*n

# print("\nRunning A*:")
# path_a = astar_search(initial_state, clauses, bias, frequency_table)
# print("A* path:", path_a)

print("\nRunning Hill climbing:")
path_h = astar_search(initial_state, clauses, bias, frequency_table)
print("Hill path:", path_h)


