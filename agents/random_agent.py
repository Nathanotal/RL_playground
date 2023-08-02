import random

class RandomAgent():
    def __init__(self, env):
        self.name = "Random"
        self.action_size = env.action_space.n
    
    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action