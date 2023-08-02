from machine_learning.neural_network import Network
import numpy as np


class MachineAgent():
    def __init__(self, network=None, action_size=None, observation_size=None, problem_type=None):
        if problem_type is None:
            raise Exception('Problem type not specified')
        if network is None:
            self.network = Network(observation_size, action_size, [5, 5])

        self.name = 'Machine'
        self.problem_type = problem_type
        self.network = network
        self.load_network(network)

    def load_network(self, network):
        self.network = network
        self.observation_size = len(network.model[0])
        self.action_size = len(network.model[-1])

    def get_action(self, state):
        output = self.network.predict(state)
        if self.problem_type == 'mountaincar_cont':
            action = self.parse_output_mountaincar(output)
        elif self.problem_type == 'cartpole':
            action = self.parse_output_cartpole(output)

        return action

    def parse_output_mountaincar(self, output):
        if output < -1:
            output = -1
        elif output > 1:
            output = 1

        return np.array([output])

    def parse_output_cartpole(self, output):
        action = np.argmax(output)

        return action

    def save(self):
        self.network.save()
