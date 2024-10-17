import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cupy as cp
import time


def vector_sum_cpu(vector):
    result = 0
    start = time.time()
    for x in vector:
        result += x
    end = time.time()
    return result, end - start


def vector_sum_gpu_lib(vector):
    start = time.time()
    vector_gpu = cp.asarray(vector)
    result_gpu = cp.sum(vector_gpu)
    result_cpu = cp.asnumpy(result_gpu)
    end = time.time()
    return result_cpu, end - start


vector_sum_kernel = ("""
__global__ void vectorSumKernel(float* result, float* vec, int len) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < len) {
        atomicAdd(result, vec[tid]);
    }
}
""")


def vector_sum_gpu(vector):
    start_time = time.time()

    res = np.zeros(1, dtype=np.float32)
    vec_gpu = cuda.mem_alloc(vector.nbytes)
    result_gpu = cuda.mem_alloc(res.nbytes)

    cuda.memcpy_htod(vec_gpu, vector)
    cuda.memcpy_htod(result_gpu, res)

    mod = SourceModule(vector_sum_kernel)
    vector_sum = mod.get_function("vectorSumKernel")

    block_size = 256
    grid_size = (len(vector) + block_size - 1) // block_size

    vector_sum(result_gpu, vec_gpu, np.int32(len(vector)), block=(block_size, 1, 1), grid=(grid_size, 1))
    cuda.Context.get_current().synchronize()

    cuda.memcpy_dtoh(res, result_gpu)
    end_time = time.time()

    return res[0], end_time - start_time


# Случайный эксперимент
vec_n = 10000
vector = np.random.uniform(1.0, 9.0, size=vec_n)
result_cpu, time_cpu = vector_sum_cpu(vector)
result_gpu_lib, time_gpu_lib = vector_sum_gpu_lib(vector)
result_gpu, time_gpu = vector_sum_gpu(vector.astype(np.float32))

print(f"Время на CPU: {time_cpu} секунд\n",
      "Проверка корректности сложения:", result_cpu)
print(f"Время на GPU, библиотечная реализация: {time_gpu_lib} секунд\n",
      "Проверка корректности сложения:", result_gpu_lib)
print(f"Время на GPU: {time_gpu} секунд\n",
      "Проверка корректности сложения:", result_gpu)
