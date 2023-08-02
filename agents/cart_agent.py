class SimpleCartAgent():
    def __init__(self):
        self.name = 'Simple cart'

    def get_action(self, state):
        position = state[0]
        velocity = state[1]

        if position == 0 or velocity > 0:
            return [1]
        else:
            return [0]
