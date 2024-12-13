from numba.pycc import CC
from numba import types
import numpy as np

cc = CC('compiled_feature_extraction')

# Signature: function_name(input_type, input_type) -> tuple_of_three_floats
# Here, types.float64[:] denotes a 1D float64 array
# The return type is a UniTuple of three float64 values.
@cc.export('compute_metrics', types.UniTuple(types.float64, 3)(types.float64[:], types.float64[:], types.int32))
def compute_metrics(real, imag, sample_size):
    total_size = real.size

    # Handle case where sample size exceeds total size
    if sample_size >= total_size:
        sampled_real = real
        sampled_imag = imag
        n = total_size
    else:
        # Efficient sampling with linspace-like indices
        sampled_indices = [i * (total_size // sample_size) for i in range(sample_size)]
        sampled_real = real[sampled_indices]
        sampled_imag = imag[sampled_indices]
        n = sample_size
    
    sum_real = 0.0
    sum_imag = 0.0
    for i in range(n):
        sum_real += real[i]
        sum_imag += imag[i]
    mean_real = sum_real / n
    mean_imag = sum_imag / n

    sum_var_real = 0.0
    sum_var_imag = 0.0
    sum_magnitude = 0.0
    for i in range(n):
        diff_real = real[i] - mean_real
        diff_imag = imag[i] - mean_imag
        sum_var_real += diff_real * diff_real
        sum_var_imag += diff_imag * diff_imag
        sum_magnitude += (real[i]*real[i] + imag[i]*imag[i])**0.5

    var_real = sum_var_real / n
    var_imag = sum_var_imag / n
    mean_magnitude = sum_magnitude / n
    return var_real, var_imag, mean_magnitude

if __name__ == "__main__":
    cc.compile()
