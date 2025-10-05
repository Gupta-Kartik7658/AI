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
        h += s_i * (bias[i+1]) + 0.5 * np.log(1 + frequency_table[i+1])
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



# ---------- Beam search (fixed beam width) ----------

def beam_search(start_state, clauses, bias, frequency_table, beam_width=3):
    """Beam search with fixed beam width."""
    # initial beam
    start_h = heuristic2(start_state, bias, frequency_table)
    start_node = Node(start_state, bias, frequency_table, h=start_h)
    beam = [start_node]
    visited = set()
    step = 0

    while beam:
        print("\n" + "="*40)
        print(f"[Beam] Step {step}: Beam states:")
        for n in beam:
            print(f"  {n.state} h={n.h:.3f}")
        step += 1

        # Check goal in current beam
        for n in beam:
            if isGoalState(n.state, clauses):
                print("\nGoal reached!")
                path = []
                node = n
                while node:
                    path.append(node.state)
                    node = node.parent
                return path[::-1]

        # Expand all nodes in beam
        all_successors = []
        for n in beam:
            visited.add(tuple(n.state))
            successors = get_successors(n)
            all_successors.extend(successors)   

        if not all_successors:
            print("No successors left.")
            return None

        # Select top beam_width successors by heuristic
        all_successors.sort(key=lambda s: s.h, reverse=True)  # max-h first
        new_beam = []
        for s in all_successors:
            if tuple(s.state) not in visited:
                new_beam.append(s)
            if len(new_beam) >= beam_width:
                break

        if not new_beam:
            print("No new states to expand.")
            return None

        beam = new_beam



k, m, n = 3, 10 , 10
# clauses = generate_clauses(k, m, n)
clauses =[[-8, -10, -7], [-2, -6, 10], [-5, 2, -4], [-8, -9, -3], [-10, 5, -9], [1, -8, 3], [-1, -2, 5], [-3, 5, -7], [5, 10, -1], [4, 8, -1]]
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

print("\nRunning Beam Searching:")
path_h = beam_search(initial_state, clauses, bias, frequency_table, beam_width=3)
for i in path_h:
    print(i)