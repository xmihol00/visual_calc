
def calculate(layers):
    current_stride = 1
    receptive_field = 1
    for kernel_size, stride in layers:
        receptive_field += (kernel_size - 1) * current_stride
        current_stride *= stride
    
    return receptive_field

print(calculate([[3, 1], [2, 2], [2, 1], [2, 2], [1, 1]]))
