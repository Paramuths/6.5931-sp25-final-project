import matplotlib.pyplot as plt
import numpy as np
def plot_cycle_transfer(zero_transfers, zero_cycles, data_cycles, network_class, num_gpus, bound=True, no_bound=True, **kwargs):
    if no_bound:
        plt.scatter(zero_transfers, data_cycles, color='red', label='no network bound')
    if bound:
        plt.scatter(zero_transfers, zero_cycles, color='blue', label='with network bound')
        
    plt.xlabel('Network Transfers (in Byte)')
    plt.ylabel('Cycles')
    plt.legend()
    plt.title('Cycles vs Network Transfers Plot')
    
    transfer_line = np.linspace(min(zero_transfers), max(zero_transfers), 1000)
    threshold = network_class(num_gpus).get_threshold()
    cycle_line = (1 / threshold) * transfer_line
    plt.plot(transfer_line, cycle_line, color='black', linestyle='--', label=f'network_transfers = {threshold} x cycles')
    plt.show()

def plot_energy_gpus(zero_energies, data_energies, workload, configs, **kwargs):
    num_gpus = [config['num_gpus'] for config in configs]
    energies_percentage = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(zero_energies, data_energies)]
    plt.scatter(num_gpus, energies_percentage, color='blue')
        
    plt.xlabel('Num GPUs')
    plt.ylabel('Energy Percentage')
    plt.title(f"Energy Percentage vs Num GPUs Plot on {workload['label']}")
    plt.show()

def plot_cycle_gpus(zero_cycles, data_cycles, workload, configs, **kwargs):
    num_gpus = [config['num_gpus'] for config in configs]
    cycles_percentage = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(zero_cycles, data_cycles)]
    plt.scatter(num_gpus, cycles_percentage, color='blue')
        
    plt.xlabel('Num GPUs')
    plt.ylabel('Cycle Percentage')
    plt.title(f"Cycle Percentage vs Num GPUs Plot on {workload['label']}")
    plt.show()

def plot_energy_pe_meshX(zero_energies, data_energies, workload, configs, **kwargs):
    num_gpus = [config['pe_meshX'] for config in configs]
    energies_percentage = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(zero_energies, data_energies)]
    plt.scatter(num_gpus, energies_percentage, color='blue')
        
    plt.xlabel('PE Mesh X')
    plt.ylabel('Energy Percentage')
    plt.title(f"Energy Percentage vs PE Mesh X Plot on {workload['label']}")
    plt.show()

def plot_cycle_pe_meshX(zero_cycles, data_cycles, workload, configs, **kwargs):
    num_gpus = [config['pe_meshX'] for config in configs]
    cycles_percentage = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(zero_cycles, data_cycles)]
    plt.scatter(num_gpus, cycles_percentage, color='blue')
        
    plt.xlabel('PE Mesh X')
    plt.ylabel('Cycle Percentage')
    plt.title(f"Cycle Percentage vs PE Mesh X Plot on {workload['label']}")
    plt.show()

def plot_energy_models(linear, ring):
    num_gpus = [config['num_gpus'] for config in linear['configs']]
    energies_percentage_linear = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(linear['zero_energies'], linear['data_energies'])]
    energies_percentage_ring = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(ring['zero_energies'], ring['data_energies'])]
    plt.scatter(num_gpus, energies_percentage_linear, color='blue', label='linear')
    plt.scatter(num_gpus, energies_percentage_ring, color='red', label='ring')
        
    plt.xlabel('Num GPUs')
    plt.ylabel('Energy Percentage')
    plt.legend()
    plt.title(f"Energy Percentage vs Num GPUs Plot on {linear['workload']['label']}")
    plt.show()

def plot_cycle_models(linear, ring):
    num_gpus = [config['num_gpus'] for config in linear['configs']]
    cycles_percentage_linear = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(linear['zero_cycles'], linear['data_cycles'])]
    cycles_percentage_ring = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(ring['zero_cycles'], ring['data_cycles'])]
    plt.scatter(num_gpus, cycles_percentage_linear, color='blue', label='linear')
    plt.scatter(num_gpus, cycles_percentage_ring, color='red', label='ring')
        
    plt.xlabel('Num GPUs')
    plt.ylabel('Cycle Percentage')
    plt.legend()
    plt.title(f"Cycle Percentage vs Num GPUs Plot on {linear['workload']['label']}")
    plt.show()