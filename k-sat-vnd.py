import random
import numpy as np
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


def get_successors_k(node, k):
    """Generate successors flipping k bits at once."""
    successors = []
    nvars = len(node.state)
    import itertools
    for indices in itertools.combinations(range(nvars), k):
        new_state = list(node.state)
        for idx in indices:
            new_state[idx] = 1 - new_state[idx]
        new_h = heuristic2(new_state, node.bias, node.frequency_table)
        successors.append(Node(new_state, node.bias, node.frequency_table,
                               parent=node, h=new_h))
    return successors

def variable_neighborhood_descent(start_state, clauses, bias, frequency_table, max_k=2):
    """Variable Neighborhood Descent algorithm."""
    current = Node(start_state, bias, frequency_table,
                   h=heuristic2(start_state, bias, frequency_table))
    visited = set()
    step = 0
    k = 1  # start neighborhood size

    while k <= max_k:
        print("\n" + "="*40)
        print(f"[VND] Step {step}: Current={current.state} h={current.h:.3f} k={k}")
        visited.add(tuple(current.state))

        if isGoalState(current.state, clauses):
            print("\nGoal reached!")
            path = []
            node = current
            while node:
                path.append(node.state)
                node = node.parent
            return path[::-1]

        successors = get_successors_k(current, k)
        if not successors:
            print("No successors at k=", k)
            k += 1
            continue

        # choose best by h
        best = max(successors, key=lambda s: s.h)
        print(f"Best successor: {best.state} h={best.h:.3f}")

        if best.h > current.h and tuple(best.state) not in visited:
            current = best
            # restart neighborhood size
            k = 1
        else:
            # no improvement in current neighborhood, try larger
            k += 1
        step += 1

    print("No improvement found with any neighborhood.")
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

print("\nRunning variable neighborhood descent:")
path_h = variable_neighborhood_descent(initial_state, clauses, bias, frequency_table)
print("Hill path:", path_h)