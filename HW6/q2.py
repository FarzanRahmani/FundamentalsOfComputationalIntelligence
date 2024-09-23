import numpy as np

# Define the target polynomials
polynomials = [
    lambda x: 2*x - 4,
    lambda x: x**2 - 8*x + 4,
    lambda x: 4*x**3 - 5*x**2 + x - 1,
    lambda x: 186*x**3 - 7.22*x**2 + 15.5*x - 13.2,
]

# Number of bits before and after the floating point
bits_before_fp = 3 # or more can be
bits_after_fp = 6 # more or less depend on problem
total_bits = bits_before_fp + bits_after_fp

def float_to_binary(solution):
    # Convert the positive float solution to a binary bit string
    if solution < 0:
        raise ValueError("Only positive numbers are allowed.")
    binary_string = format(int(solution * (2 ** bits_after_fp)), f'0{total_bits}b')
    return '0' + binary_string[:bits_before_fp] + '.' + binary_string[bits_before_fp:]

def binary_to_float(binary_string):
    # Convert the binary bit string to a float
    integer_part, fractional_part = binary_string[1:].split('.')
    return int(integer_part, 2) + int(fractional_part, 2) / (2 ** bits_after_fp)

def fitness(solution, polynomial):
    # Fitness function: evaluate how close the solution is to being a root
    # return abs(polynomial(solution))
    epsilon = 0.000001 # for numerical stability
    return 1 / ( abs(polynomial(solution)) + epsilon )

def initialize_population(population_size):
    # Initialize a population of potential solutions as binary bit strings
    return [float_to_binary(np.random.uniform(0, 10)) for _ in range(population_size)]

def select_parents(population, fitness_scores):
    # Select two parents based on their fitness scores
    probabilities = fitness_scores / fitness_scores.sum()
    parents_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[parents_indices[0]], population[parents_indices[1]]]

def crossover(parent1, parent2):
    # 00000.0000
    # Two-point crossover operation on binary bit strings
    crossover_point1 = np.random.randint(1, total_bits)
    crossover_point2 = np.random.randint(crossover_point1, total_bits+1)
    
    # Remove dot before crossover
    parent1_without_dot = parent1[:bits_before_fp+1] + parent1[bits_before_fp+2:]
    parent2_without_dot = parent2[:bits_before_fp+1] + parent2[bits_before_fp+2:]

    child_without_dot = (
        parent1_without_dot[:crossover_point1] +
        parent2_without_dot[crossover_point1:crossover_point2] +
        parent1_without_dot[crossover_point2:]
    )

    # Add dot back to the child
    child = child_without_dot[:bits_before_fp+1] + '.' + child_without_dot[bits_before_fp+1:]

    return child

def mutate(solution, mutation_rate):
    # Bit flip mutation with mutation probability
    solution_without_dot = solution[:bits_before_fp+1] + solution[bits_before_fp+2:]
    mutated_solution_without_dot = ''.join([bit if np.random.rand() > mutation_rate else '1' if bit == '0' else '0' for bit in solution_without_dot])
    mutated_solution_with_dot = mutated_solution_without_dot[:bits_before_fp+1] + '.' + mutated_solution_without_dot[bits_before_fp+1:]

    mutated_solution_with_dot = '0' + mutated_solution_with_dot[1:] # in this question we only consider positive answers
    return mutated_solution_with_dot

def genetic_algorithm(polynomial, population_size=100, generations=1000, mutation_rate=0.01):
    population = initialize_population(population_size)

    found_root = False
    for generation in range(generations):
        float_population = [binary_to_float(solution) for solution in population]
        fitness_scores = np.array([fitness(solution, polynomial) for solution in float_population])

        # Check if any solution is a good enough root
        best_solution = population[np.argmin(fitness_scores)]
        if fitness(binary_to_float(best_solution), polynomial) >= 100: # 0.01 difference from orginal root
            print(f"Root found in generation {generation}: {binary_to_float(best_solution)}")
            found_root = True
            print("Value of polynomial with this solution(error)", polynomial(binary_to_float(best_solution)))
            break

        parents = np.array([select_parents(population, fitness_scores) for _ in range(population_size // 2)])

        # Crossover and mutation
        children = np.array([crossover(parent1, parent2) for parent1, parent2 in parents])
        children = np.array([mutate(child, mutation_rate) for child in children])
        
        # population[:population_size // 2] = parents.flatten()
        # population[population_size // 2:] = children # updated population
        # -------------
        # Sort parents based on fitness
        sorted_indices = np.argsort(fitness_scores) # small to big
        tmp = np.array(population)
        sorted_parents = tmp[sorted_indices]

        # Choose the top 50 parents with better fitness
        selected_parents = sorted_parents[population_size // 2:]

        # Flatten the selected parents to update the population
        population[:population_size // 2] = selected_parents.flatten()
        population[population_size // 2:] = children  # updated population

    if (not found_root): # No exact root found.
        best_answer_found = population[np.argmin(fitness_scores)]
        print("Best solution in 1000 generations:", binary_to_float(best_answer_found))
        print("Fitness Score (1/error): ", np.argmin(fitness_scores))
        print("Value of polynomial with this solution(error)", polynomial(binary_to_float(best_answer_found)))

# Test the genetic algorithm on each polynomial
for idx, polynomial in enumerate(polynomials):
    print(f"\nTesting Polynomial {idx + 1}:")
    genetic_algorithm(polynomial)
