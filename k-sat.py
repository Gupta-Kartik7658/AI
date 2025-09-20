import random

def generate_k_sat(k, m, n):
    formula = []

    for _ in range(m):
        # pick k distinct variables
        variables = random.sample(range(1, n + 1), k)
        # randomly negate variables
        clause = [var if random.choice([True, False]) else -var for var in variables]
        formula.append(clause)

    return formula


def format_formula(formula):
    """Format CNF formula into human-readable string."""
    result = []
    for clause in formula:
        literals = []
        for var in clause:
            if var > 0:
                literals.append(f"x{var}")
            else:
                literals.append(f"¬x{abs(var)}")
        result.append("(" + " v ".join(literals) + ")")
    return " ∧ ".join(result)



k = 3   # literals per clause
m = 3   # number of clauses
n = 5   # number of variables

formula = generate_k_sat(k, m, n)
print("Generated formula in CNF:")
print(format_formula(formula))