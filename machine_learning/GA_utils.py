import numpy as np
from machine_learning.neural_network import Network
import gym


def mutate(network, mutation_rate=0.02, mutation_amount=0.05):
    weights, biases = network.model
    mutated_weights = []
    mutated_biases = []

    # Iterate through the weights and biases and apply mutations
    for weight_matrix, bias_vector in zip(weights, biases):
        # Create a mask to identify the elements to be mutated
        weight_mask = np.random.rand(*weight_matrix.shape) < mutation_rate
        bias_mask = np.random.rand(*bias_vector.shape) < mutation_rate

        # Generate random Gaussian noise to be added to the elements to be mutated
        weight_noise = np.random.randn(*weight_matrix.shape) * mutation_amount
        bias_noise = np.random.randn(*bias_vector.shape) * mutation_amount

        # Apply mutations using the masks
        mutated_weight_matrix = weight_matrix + weight_mask * weight_noise
        mutated_bias_vector = bias_vector + bias_mask * bias_noise

        mutated_weights.append(mutated_weight_matrix)
        mutated_biases.append(mutated_bias_vector)

    # Replace the original model with the mutated version
    network.model = (mutated_weights, mutated_biases)


def crossover(parent1, parent2):
    weights1, biases1 = parent1.model
    weights2, biases2 = parent2.model

    # Initializing offspring's weights and biases
    offspring_weights = []
    offspring_biases = []

    # Iterate through the parent's weights and biases and perform crossover
    for (weight_matrix1, bias_vector1), (weight_matrix2, bias_vector2) in zip(zip(weights1, biases1), zip(weights2, biases2)):
        # Crossover weights
        crossover_point = np.random.randint(0, weight_matrix1.shape[1])
        offspring_weight_matrix = np.hstack(
            (weight_matrix1[:, :crossover_point], weight_matrix2[:, crossover_point:]))

        # Crossover biases
        crossover_point = np.random.randint(0, bias_vector1.shape[0])
        offspring_bias_vector = np.concatenate(
            (bias_vector1[:crossover_point], bias_vector2[crossover_point:]))

        offspring_weights.append(offspring_weight_matrix)
        offspring_biases.append(offspring_bias_vector)

    # Create a new network with the crossed-over weights and biases
    offspring = Network(parent1.in_size, parent1.out_size, parent1.layers, model=(
        offspring_weights, offspring_biases))

    return offspring


def select_next_population(parents, offspring, population, population_size):
    new_pop = offspring
    i = 0
    while len(new_pop) < population_size:
        if i >= len(parents):
            print('This should not happen (pop added to pop)')
            new_pop.append(population[i])
        else:
            new_pop.append(parents[i])
        i += 1

    if len(new_pop) > population_size:
        print('This should not happen (too large pop)')
        new_pop = new_pop[:population_size]

    return new_pop


def select_parents(population, fitness, tournament_size=3):
    parents = []
    for _ in range(len(population)):
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False)
        tournament_fitness = [fitness[idx] for idx in tournament_indices]
        best_in_tournament_idx = tournament_indices[np.argmax(
            tournament_fitness)]
        parents.append(population[best_in_tournament_idx])
    return parents


def best_individual(population, fitness):
    # Finding the index of the best individual
    best_index = np.argmax(fitness)
    return population[best_index]  # Returning the best individual
