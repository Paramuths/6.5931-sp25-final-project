THRESHOLD = 1
def get_weight_size(workload):
    return workload['inputCh_size'] * workload['outputCh_size'] * workload['filter_height'] * workload['filter_width']
    
def get_data_config(workload):
    pe_spatial_c = 4 if workload['inputCh_size'] % 4 == 0 else 1
    pe_spatial_m = 4 if workload['outputCh_size'] % 4 == 0 else 1
    
    arch_config = dict(
        gpu_meshX=workload['batch_size'],
        gpu_meshY=1,
        pe_meshX=4, 
        pe_meshY=4
    )

    config = dict(
        disk_factor_N=1,
        disk_factor_M=1,
        disk_factor_C=1,
        GPU_spatial_factor_M=1,
        GPU_spatial_factor_C=1,
        GPU_spatial_factor_N=workload['batch_size'],
        self_memory_factor_N=1,
        self_memory_factor_M=int(workload['outputCh_size'] / pe_spatial_m),
        self_memory_factor_C=int(workload['inputCh_size'] / pe_spatial_c),
        PE_spatial_factor_M=pe_spatial_m,
        PE_spatial_factor_C=pe_spatial_c,
        scratchpad_factor_N=1,
    )

    full_config = {
        **config,
        **arch_config,
        **workload
    }

    return full_config

def get_data_energy(result):
    return result.energy - result.per_component_energy['disk']

def get_data_cycles(result):
    return result.cycles

def get_zero_config(workload):
    pe_spatial_c = 4 if workload['inputCh_size'] % 4 == 0 else 1
    pe_spatial_m = 4 if workload['outputCh_size'] % 4 == 0 else 1

    arch_config = dict(
        gpu_meshX=1,
        gpu_meshY=1,
        pe_meshX=4, 
        pe_meshY=4
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
        self_memory_factor_M=int(workload['outputCh_size'] / pe_spatial_m),
        self_memory_factor_C=int(workload['inputCh_size'] / pe_spatial_c),
        PE_spatial_factor_M=pe_spatial_m,
        PE_spatial_factor_C=pe_spatial_c,
        scratchpad_factor_N=1,
    )

    full_config = {
        **config,
        **arch_config,
        **workload,
        'batch_size': 1, # overwrite workload batch_size
    }

    return full_config
    
def get_zero_energy(result, workload):
    access_energy_factor = 1e-9 # Change this
    return workload['batch_size'] * (result.energy - result.per_component_energy['disk']) \
        + (workload['batch_size'] - 1) * get_weight_size(workload) * access_energy_factor
    
def get_zero_network_hops(workload):
    return (workload['batch_size'] - 1) * get_weight_size(workload)

def get_zero_cycles(result, workload):
    num_hops = get_zero_network_hops(workload)
    cycles = result.cycles

    return cycles if (num_hops / cycles) <= THRESHOLD else int(num_hops / THRESHOLD)