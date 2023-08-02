from machine_learning.neural_network import Network
from evaluation.cartpole import get_fitness_cartpole
from evaluation.mountaincar import get_fitness_mountaincar
from machine_learning.GA_utils import *


def genetic_algorithm(population_size, generations, in_size, out_size, layers, mutation_rate, mutation_amount, crossover_rate, problem_type):
    if problem_type == None:
        raise Exception("Please specify a problem type.")
    elif problem_type == 'cartpole':
        get_fitness = get_fitness_cartpole
    elif problem_type == 'mountaincar_cont':
        get_fitness = get_fitness_mountaincar
    else:
        raise Exception("Invalid problem type.")

    # Initialize population
    population = [Network(in_size, out_size, layers)
                  for _ in range(population_size)]

    print("🧬 Genetic Algorithm Started 🧬")

    for generation in range(generations):
        # Evaluate fitness
        if generation == generations - 1:
            fitness = get_fitness(population, max_steps=250)
        else:
            fitness = get_fitness(population, max_steps=1000)

        # Log best fitness of the generation
        best_fitness = max(fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness} 🏆")

        if generation == generations - 1:
            break

        # Select parents
        parents = select_parents(population, fitness)

        # Crossover
        offspring = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            if np.random.rand() < crossover_rate:
                offspring.append(crossover(parent1, parent2))
            else:
                offspring.append(parent1)
                offspring.append(parent2)

        # Mutate
        for child in offspring:
            mutate(child, mutation_rate, mutation_amount)

        # Select next generation
        population = select_next_population(
            parents, offspring, population, population_size)

    best_agent = best_individual(population, fitness)
    print(f"🎖️ Genetic Algorithm Completed: Best Agent Found 🎖️")
    return best_agent
