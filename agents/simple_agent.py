class SimpleAgent():
    def __init__(self, env):
        self.name = 'Simple'
        self.action_size = env.action_space.n
    
    def get_action(self, state):
        pole_angle = state[2]
        action = 0 if pole_angle < 0 else 1
        return action