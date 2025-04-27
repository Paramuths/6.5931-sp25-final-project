conv1 = dict(
    inputCh_size = 24,
    outputCh_size = 24,
    filter_height = 5,
    filter_width = 5,
    ofmap_height = 28,
    ofmap_width = 28,
    label='conv1',
)

conv2 = dict(
    inputCh_size = 96,
    outputCh_size = 96,
    filter_height = 3,
    filter_width = 3,
    ofmap_height = 10,
    ofmap_width = 10,
    label='conv2',
)

fc1 = dict(
    inputCh_size = 48,
    outputCh_size = 32,
    filter_height = 1,
    filter_width = 1,
    ofmap_height = 1,
    ofmap_width = 1,
    label='fc1',
)

fc2 = dict(
    inputCh_size = 96,
    outputCh_size = 64,
    filter_height = 1,
    filter_width = 1,
    ofmap_height = 1,
    ofmap_width = 1,
    label='fc2',
)

workloads = [
    conv1,
    conv2,
    fc1,
    fc2,
]