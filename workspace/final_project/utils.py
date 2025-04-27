import math

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