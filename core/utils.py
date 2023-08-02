from machine_learning.neural_network import Network

def load_network(name, in_size, out_size, layers):
    network = Network(in_size, out_size, layers, None)
    network.load(name)
    return network
