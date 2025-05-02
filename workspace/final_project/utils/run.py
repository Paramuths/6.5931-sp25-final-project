import math
from utils.loaders import *

def get_data_config(workload, pe_meshX, pe_meshY):
    assert workload['inputCh_size'] % pe_meshX == 0
    assert workload['outputCh_size'] % pe_meshY == 0
    
    arch_config = dict(
        gpu_meshX=workload['batch_size'],
        gpu_meshY=1,
        pe_meshX=pe_meshX, 
        pe_meshY=pe_meshY
    )

    config = dict(
        disk_factor_N=1,
        disk_factor_M=1,
        disk_factor_C=1,
        GPU_spatial_factor_M=1,
        GPU_spatial_factor_C=1,
        GPU_spatial_factor_N=workload['batch_size'],
        self_memory_factor_N=1,
        self_memory_factor_M=int(workload['outputCh_size'] / pe_meshY),
        self_memory_factor_C=int(workload['inputCh_size'] / pe_meshX),
        PE_spatial_factor_M=pe_meshY,
        PE_spatial_factor_C=pe_meshX,
        scratchpad_factor_N=1,
    )

    full_config = {
        **config,
        **arch_config,
        **workload,
    }

    return full_config

def get_data_energy(result):
    return result.energy - result.per_component_energy['disk']

def get_data_cycles(result):
    return result.cycles

def get_zero_config(workload, pe_meshX, pe_meshY):
    assert workload['inputCh_size'] % pe_meshX == 0
    assert workload['outputCh_size'] % pe_meshY == 0

    arch_config = dict(
        gpu_meshX=1,
        gpu_meshY=1,
        pe_meshX=pe_meshX, 
        pe_meshY=pe_meshY
    )

    config = dict(
        disk_factor_N=1,
        disk_factor_M=1,
        disk_factor_C=1,
        other_memories_factor_N=1,
        other_memories_factor_M=1,
        other_memories_factor_C=1,
        GPU_spatial_factor_M=1,
        GPU_spatial_factor_C=1,
        GPU_spatial_factor_N=1,
        self_memory_factor_N=1,
        self_memory_factor_M=int(workload['outputCh_size'] / pe_meshY),
        self_memory_factor_C=int(workload['inputCh_size'] / pe_meshX),
        PE_spatial_factor_M=pe_meshY,
        PE_spatial_factor_C=pe_meshX,
        scratchpad_factor_N=1,
    )

    full_config = {
        **config,
        **arch_config,
        **workload,
        'batch_size': 1, # overwrite batch_size from workload
    }

    return full_config

def get_weight_size(workload):
    return workload['inputCh_size'] * workload['outputCh_size'] * workload['filter_height'] * workload['filter_width']

def get_zero_network_transfer(workload):
    return (workload['batch_size'] - 1) * get_weight_size(workload)
    
def get_zero_energy(result, network_model, workload):
    return workload['batch_size'] * (result.energy - result.per_component_energy['disk']) \
        + get_zero_network_transfer(workload) * network_model.get_network_energy()

def get_zero_cycles(result, network_model, workload):
    network_transfer = get_zero_network_transfer(workload)
    cycles = result.cycles
    threshold = network_model.get_threshold()
    return cycles if (network_transfer / cycles) <= threshold else math.ceil(network_transfer / threshold)

# return num cycles and energy
def get_zero_result(workload, num_gpus, pe_meshX, pe_meshY, network_class, **kwargs):
    workload['batch_size'] = num_gpus # simplify the simulation
    zero_config = get_zero_config(workload, pe_meshX, pe_meshY)
    zero_result = run_timeloop_model(
        zero_config,
        architecture='designs/architecture/arch.yaml',
        mapping='designs/architecture/map.yaml',
        problem='layer_shapes/workload.yaml'
    )
    network_model = network_class(num_gpus)
    cycles = get_zero_cycles(zero_result, network_model, workload)
    energy = get_zero_energy(zero_result, network_model, workload)
    return cycles, energy 

# return num cycles and energy
def get_data_result(workload, num_gpus, pe_meshX, pe_meshY, **kwargs):
    workload['batch_size'] = num_gpus # simplify the simulation
    data_config = get_data_config(workload, pe_meshX, pe_meshY)
    data_result = run_timeloop_model(
        data_config,
        architecture='designs/architecture/arch.yaml',
        mapping='designs/architecture/map.yaml',
        problem='layer_shapes/workload.yaml'
    )
    cycles = get_data_cycles(data_result)
    energy = get_data_energy(data_result)
    return cycles, energy 

def run(workload, config):
    zero_cycle, zero_energy = get_zero_result(workload, **config)
    zero_transfer = get_zero_network_transfer(workload)
    data_cycle, data_energy = get_data_result(workload, **config)
    return dict(zero_cycle=zero_cycle, 
            zero_energy=zero_energy, 
            zero_transfer=zero_transfer, 
            data_cycle=data_cycle, 
            data_energy=data_energy,
           )

def run_configs(workload, configs):
    zero_cycles = []
    zero_energies = []
    zero_transfers = []
    data_cycles = []
    data_energies = []
    for config in configs:
        result = run(workload, config)
        zero_cycles.append(result['zero_cycle'])
        zero_energies.append(result['zero_energy'])
        zero_transfers.append(result['zero_transfer'])
        data_cycles.append(result['data_cycle'])
        data_energies.append(result['data_energy'])
        
    return dict(zero_cycles=zero_cycles, 
                zero_energies=zero_energies, 
                zero_transfers=zero_transfers, 
                data_cycles=data_cycles, 
                data_energies=data_energies,
                configs=configs,
                workload=workload
               )