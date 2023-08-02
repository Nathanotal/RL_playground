from machine_learning.GA import genetic_algorithm
from core.utils import load_network
from agents.machine_agent import MachineAgent
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from agents.cart_agent import SimpleCartAgent
from core.core import visualize_agent, compare_agents
import gym


def try_agent():
    network = load_network('first_try_new_structure', 2, 1, [5, 5])
    # network = Network(2, 4, [5, 5], None)
    agent = MachineAgent(network, 1, 2, problem_type='mountaincar_cont')
    # env = gym.make('CartPole-v1')
    # agent = SimpleAgent(env)

    # agent = SimpleCartAgent()

    visualize_agent(agent, 1000, 'mountaincar_cont')


def train_network(name, save=True):
    trained_network = genetic_algorithm(population_size=250, generations=5, in_size=2, out_size=1, layers=[
                                        4, 2], mutation_rate=0.01, mutation_amount=0.1, crossover_rate=0.5, problem_type='mountaincar_cont')
    if save:
        trained_network.save(name)
    return trained_network


def main():
    network = train_network('first_try_new_structure', save=True)
    while True:
        try_agent()


if __name__ == "__main__":
    main()
