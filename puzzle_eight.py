from collections import deque
import numpy as np
import random

# Author : Pratik Shah
# Date : August 20, 2024
# Place : IIIT Vadodara

# Course : CS307 Artificial Intelligence
# Exercise : Puzzle Eight Solver
# Learning Objective : Revisit concepts of basic data structures, BFS and DFS

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


def get_successors(node):
    successors = []
    value = 0
    index = node.state.index(0)
    quotient = index//3
    remainder = index%3
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

    # moves = [-1, 1, 3, -3]
    for move in moves:
        im = index+move
        if im >= 0 and im < 9:
            new_state = list(node.state)
            temp = new_state[im]
            new_state[im] = new_state[index]
            new_state[index] = temp
            successor = Node(new_state, node)
            successors.append(successor)            
    return successors



def bfs(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)
    queue = deque([start_node])
    visited = set()
    nodes_explored = 0
    while queue:
        node = queue.popleft()
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        # print(node.state)
        nodes_explored = nodes_explored + 1
        if node.state == list(goal_node.state):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print('Total nodes explored', nodes_explored)
            return path[::-1]
        for successor in get_successors(node):
            queue.append(successor)
    # print('Total nodes explored', nodes_explored)
    return None

start_state = [7,2,4,5,0,6,8,3,1]
s_node = Node(start_state)
# D = 20
# d = 0
# while d <= D:
#     goal_state = random.choice(list(get_successors(s_node))).state
#     s_node = Node(goal_state)
#     d = d+1
#     # print(goal_state)

goal_state = [0, 1, 2, 3, 4, 5, 6, 7, 8]
solution = bfs(start_state, goal_state)
if solution:
    print("Solution found:")
    for step in solution:
        print(step)
else:
    print("No solution found.")