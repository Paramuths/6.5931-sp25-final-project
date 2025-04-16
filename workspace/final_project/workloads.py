conv1 = dict(
    inputCh_size = 1,
    outputCh_size = 4,
    filter_height = 5,
    filter_width = 5,
    ofmap_height = 28,
    ofmap_width = 28,
    batch_size = 16,
)

conv2 = dict(
    inputCh_size = 4,
    outputCh_size = 8,
    filter_height = 5,
    filter_width = 5,
    ofmap_height = 28,
    ofmap_width = 28,
    batch_size = 16,
)

conv3 = dict(
    inputCh_size = 8,
    outputCh_size = 16,
    filter_height = 5,
    filter_width = 5,
    ofmap_height = 28,
    ofmap_width = 28,
    batch_size = 16,
)

depthwise_layer = dict(
    inputCh_size = 1,
    outputCh_size = 72,
    filter_height = 3,
    filter_width = 3,
    ofmap_height = 5,
    ofmap_width = 5,
    batch_size = 16,
)

fc1 = dict(
    inputCh_size = 196, # original 392
    outputCh_size = 64, # original 256
    filter_height = 1,
    filter_width = 1,
    ofmap_height = 1,
    ofmap_width = 1,
    batch_size = 16,
)

fc2 = dict(
    inputCh_size = 256,
    outputCh_size = 10,
    filter_height = 1,
    filter_width = 1,
    ofmap_height = 1,
    ofmap_width = 1,
    batch_size = 16,
)

medium_layer = dict(
    inputCh_size = 32,
    outputCh_size = 32,
    filter_height = 3,
    filter_width = 3,
    ofmap_height = 10,
    ofmap_width = 10,
    batch_size = 16,
)

pointwise_layer = dict(
    inputCh_size = 12,
    outputCh_size = 20,
    filter_height = 1,
    filter_width = 1,
    ofmap_height = 5,
    ofmap_width = 5,
    batch_size = 16,
)

small_layer = dict(
    inputCh_size = 16,
    outputCh_size = 16,
    filter_height = 3,
    filter_width = 3,
    ofmap_height = 5,
    ofmap_width = 5,
    batch_size = 16,
)

tiny_layer = dict(
    inputCh_size = 8,
    outputCh_size = 8,
    filter_height = 3,
    filter_width = 3,
    ofmap_height = 5,
    ofmap_width = 5,
    batch_size = 16,
)

workloads = [
    conv1,
    conv2,
    conv3,
    depthwise_layer,
    fc1,
    fc2,
    medium_layer,
    pointwise_layer,
    small_layer,
    tiny_layer
]

labels = [
    'conv1',
    'conv2',
    'conv3',
    'depthwise_layer',
    'fc1',
    'fc2',
    'medium_layer',
    'pointwise_layer',
    'small_layer',
    'tiny_layer',
]