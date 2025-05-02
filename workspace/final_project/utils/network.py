class NetworkModel:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.bandwidth_per_link = 32
        self.energy_per_link_per_byte = 1.6e-10
        self.average_distance = 0

    def get_threshold(self):
        return self.num_gpus * self.bandwidth_per_link / self.average_distance

    def get_network_energy(self):
        return self.energy_per_link_per_byte * self.average_distance

# 1-D Mesh
class LinearModel(NetworkModel):
    def __init__(self, num_gpus):
        super().__init__(num_gpus)
        self.average_distance = (num_gpus + 1) / 3

class RingModel(NetworkModel):  
    def __init__(self, num_gpus):
        super().__init__(num_gpus)
        self.average_distance = (num_gpus * num_gpus) / (4 * (num_gpus - 1)) if num_gpus % 2 == 0 else (num_gpus + 1) / 4

# All-to-all
class FullyConnectedModel(NetworkModel):
    def __init__(self, num_gpus):
        super().__init__(num_gpus)
        self.average_distance = 1
        