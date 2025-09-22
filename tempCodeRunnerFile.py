
def hill_climb(start_state, clauses, bias, frequency_table):
    """Greedy hill climbing with max-h."""
    current = Node(start_state, bias, frequency_table, g=0, h=heuristic2(start_state, bias, frequency_table))
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
