conv2 = dict(
    inputCh_size = 4,
    outputCh_size = 8,
    filter_height = 5,
    filter_width = 5,
    ofmap_height = 28,
    ofmap_width = 28,
    batch_size = 16,
)

def get_weight_size(workload):
    return workload['inputCh_size'] * workload['outputCh_size'] * workload['filter_height'] * workload['filter_width']