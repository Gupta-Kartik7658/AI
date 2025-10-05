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


def heuristic2(state, bias, frequency_table, lam=0.5, base=math.e):
    h = 0
    for i in range(len(state)):
        s_i = 1 if state[i] == 1 else -1
        h += s_i * (bias[i+1] + lam * (math.log(1 + frequency_table[i+1], base)))
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

# ---------- Hill climbing (max-h) ----------

def hill_climb(start_state, clauses, bias, frequency_table):
    """Greedy hill climbing with max-h."""
    current = Node(start_state, bias, frequency_table, g=0,
                   h=heuristic2(start_state, bias, frequency_table))
    visited = set()
    step = 0

    while True:
        print("\n" + "="*40)
        print(f"[Hill] Step {step}: Current={current.state} h={current.h:.3f}")
        visited.add(tuple(current.state))
        print(f"Visited states: {len(visited)}")

        if isGoalState(current.state, clauses):
            print("\nGoal reached!")
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]

        successors = get_successors(current)
        if not successors:
            print("No successors.")
            return None

        # choose best by h
        best = max(successors, key=lambda s: s.h)
        print(f"Best successor: {best.state} h={best.h:.3f}")

        if tuple(best.state) in visited:
            print("Stuck in local maximum.")
            return None

        current = best
        step += 1



# ---------- Example usage ----------

k, m, n = 3, 10 , 10
# clauses = generate_clauses(k, m, n)
clauses = [[-8, -10, -7], [-2, -6, 10], [-5, 2, -4], [-8, -9, -3], [-10, 5, -9], [1, -8, 3], [-1, -2, 5], [-3, 5, -7], [5, 10, -1], [4, 8, -1]]
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

print("\nRunning Hill climbing:")
path_h = hill_climb(initial_state, clauses, bias, frequency_table)
for i in path_h:
    print(i)


