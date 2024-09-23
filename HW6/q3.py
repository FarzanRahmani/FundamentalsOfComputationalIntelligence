import numpy as np
import random

# Function to initialize a random population
def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        chromosome = np.random.permutation(range(1, 37))  # Generate a random permutation of numbers 1 to 36
        population.append(chromosome)
    return population

# Function to evaluate fitness based on constraints
def calculate_fitness(chromosome):
    square = chromosome.reshape((6, 6))  # Reshape the chromosome into a 6x6 square
    
    # Count the number of odd and even numbers in each row and column
    row_even_count = np.sum(square % 2 == 0, axis=1)
    row_odd_count = 6 - row_even_count
    col_even_count = np.sum(square % 2 == 0, axis=0)
    col_odd_count = 6 - col_even_count
    
    # Calculate violations (absolute differences) from the ideal count (3 even and 3 odd in each row/column)
    row_violations = np.abs(row_even_count - 3) + np.abs(row_odd_count - 3)
    col_violations = np.abs(col_even_count - 3) + np.abs(col_odd_count - 3)
    
    # Total violations in the square
    total_violations = np.sum(row_violations) + np.sum(col_violations)
    
    return 1 / (1 + total_violations)  # Fitness is inversely proportional to violations

# Genetic Algorithm
def genetic_algorithm(population_size, mutation_rate, num_generations):
    population = initialize_population(population_size)
    
    for generation in range(num_generations):
        # Evaluate fitness for each individual in the population
        fitness_scores = [calculate_fitness(chromosome) for chromosome in population]
        
        # Select parents for reproduction (using roulette wheel selection)
        selected_indices = np.random.choice(range(population_size), size=population_size, replace=True,
                                            p=[score / sum(fitness_scores) for score in fitness_scores])
        selected_population = [population[i] for i in selected_indices]
        
        # Create offspring through crossover (using two-point crossover)
        new_population = []
        for i in range(0, population_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            crossover_points = sorted(random.sample(range(36), 2))
            child1 = np.concatenate((parent1[:crossover_points[0]], parent2[crossover_points[0]:crossover_points[1]],
                                     parent1[crossover_points[1]:]))
            child2 = np.concatenate((parent2[:crossover_points[0]], parent1[crossover_points[0]:crossover_points[1]],
                                     parent2[crossover_points[1]:]))
            new_population.extend([child1, child2])
        
        # Apply mutation to some individuals in the population
        for chromosome in new_population:
            if random.random() < mutation_rate:
                index1, index2 = random.sample(range(36), 2)
                chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
        
        population = new_population
    
    # Return the best solution found
    best_chromosome = max(population, key=calculate_fitness)
    return best_chromosome.reshape((6, 6))

# Set parameters and run the genetic algorithm
population_size = 100
mutation_rate = 0.1
num_generations = 500

best_solution = genetic_algorithm(population_size, mutation_rate, num_generations)
print("Best Solution:")
print(best_solution)
