import matplotlib.pyplot as plt
import numpy as np
import os

plot_dir = "plots"
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'mathtext.fontset': 'cm',
})

def plot_energy_gpus(fc, conv):
    num_gpus = [config['num_gpus'] for config in fc['configs']]
    energies_percentage_fc = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(fc['zero_energies'], fc['data_energies'])]
    energies_percentage_conv = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(conv['zero_energies'], conv['data_energies'])]

    x = np.arange(len(num_gpus))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, energies_percentage_fc, width, label=fc['workload']['label'], color='steelblue')
    ax.bar(x + width/2, energies_percentage_conv, width, label=conv['workload']['label'], color='darkorange')

    ax.set_xlabel('Num GPUs')
    ax.set_ylabel('Energy Percentage')
    # ax.set_title(f"Energy Percentage vs Num GPUs Plot")
    ax.set_xticks(x)
    ax.set_xticklabels(num_gpus)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "energy_gpus.png"))
    plt.show()

def plot_cycle_gpus(fc, conv):
    num_gpus = [config['num_gpus'] for config in fc['configs']]
    cycles_percentage_fc = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(fc['zero_cycles'], fc['data_cycles'])]
    cycles_percentage_conv = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(conv['zero_cycles'], conv['data_cycles'])]

    x = np.arange(len(num_gpus))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, cycles_percentage_fc, width, label=fc['workload']['label'], color='steelblue')
    ax.bar(x + width/2, cycles_percentage_conv, width, label=conv['workload']['label'], color='darkorange')

    ax.set_xlabel('Num GPUs')
    ax.set_ylabel('Cycle Percentage')
    # ax.set_title(f"Cycle Percentage vs Num GPUs Plot")
    ax.set_xticks(x)
    ax.set_xticklabels(num_gpus)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "cycle_gpus.png"))
    plt.show()

def plot_energy_pe_meshX(result):
    num_pe_meshX = [config['pe_meshX'] for config in result['configs']]
    x = np.arange(len(num_pe_meshX))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, [energy * 1e6 for energy in result['zero_energies']], width, label='zero parallelism', color='forestgreen')
    ax.bar(x + width/2, [energy * 1e6 for energy in result['data_energies']], width, label='data paralleism', color='mediumpurple')

    ax.set_xlabel('Num PE Columns')
    ax.set_ylabel(r'Energy $(\mu J)$')
    # ax.set_title(f"Energy vs Num PE Columns Plot for {result['workload']['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(num_pe_meshX)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "energy_pes.png"))
    plt.show()

def plot_cycle_pe_meshX(result):
    num_pe_meshX = [config['pe_meshX'] for config in result['configs']]
    x = np.arange(len(num_pe_meshX))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, result['zero_cycles'], width, label='zero parallelism', color='forestgreen')
    ax.bar(x + width/2, result['data_cycles'], width, label='data paralleism', color='mediumpurple')

    ax.set_xlabel('Num PE Columns')
    ax.set_ylabel('Cycle')
    # ax.set_title(f"Cycle vs Num PE Columns Plot for {result['workload']['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(num_pe_meshX)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "cycle_pes.png"))
    plt.show()

def plot_energy_models(linear, ring, fc):
    num_gpus = [config['num_gpus'] for config in linear['configs']]
    energies_percentage_linear = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(linear['zero_energies'], linear['data_energies'])]
    energies_percentage_ring = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(ring['zero_energies'], ring['data_energies'])]
    energies_percentage_fc = [100 * zero_energy / data_energy for zero_energy, data_energy in zip(fc['zero_energies'], fc['data_energies'])]

    x = np.arange(len(num_gpus))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, energies_percentage_linear, width, label='linear', color='royalblue')
    ax.bar(x, energies_percentage_ring, width, label='ring', color='darkorange')
    ax.bar(x + width, energies_percentage_fc, width, label='fully connected', color='forestgreen')

    ax.set_xlabel('Num GPUs')
    ax.set_ylabel('Energy Percentage')
    # ax.set_title(f"Energy Percentage vs Num GPUs Plot on {linear['workload']['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(num_gpus)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "energy_models.png"))
    plt.show()

def plot_cycle_models(linear, ring, fc):
    num_gpus = [config['num_gpus'] for config in linear['configs']]
    cycles_percentage_linear = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(linear['zero_cycles'], linear['data_cycles'])]
    cycles_percentage_ring = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(ring['zero_cycles'], ring['data_cycles'])]
    cycles_percentage_fc = [100 * zero_cycle / data_cycle for zero_cycle, data_cycle in zip(fc['zero_cycles'], fc['data_cycles'])]

    x = np.arange(len(num_gpus))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, cycles_percentage_linear, width, label='linear', color='royalblue')
    ax.bar(x, cycles_percentage_ring, width, label='ring', color='darkorange')
    ax.bar(x + width, cycles_percentage_fc, width, label='fully connected', color='forestgreen')

    ax.set_xlabel('Num GPUs')
    ax.set_ylabel('Cycle Percentage')
    # ax.set_title(f"Cycle Percentage vs Num GPUs Plot on {linear['workload']['label']}")
    ax.set_xticks(x)
    ax.set_xticklabels(num_gpus)
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "cycle_models.png"))
    plt.show()