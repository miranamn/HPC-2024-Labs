import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.pyplot as plt
from pycuda.compiler import SourceModule
import numpy as np
import cupy as cp
import time


def matrix_mult_cpu(m_a, m_b):
    if m_a.shape[1] != m_b.shape[0]:
        raise ValueError("Внутренние размерности не совпадают")

    start = time.time()
    res = [[sum(m_a[i, k] * m_b[k, j] for k in range(m_a.shape[1]))
            for j in range(m_b.shape[1])] for i in range(m_a.shape[0])]
    res = np.array(res)
    end = time.time()

    result_correct = np.dot(m_a, m_b)
    is_correct = 1 if np.allclose(res, result_correct) else 0
    return is_correct, end - start


def matrix_mult_gpu_lib(m_a, m_b):
    if m_a.shape[1] != m_b.shape[0]:
        raise ValueError("Внутренние размерности не совпадают")

    matrix_a_gpu = cp.asarray(m_a)
    matrix_b_gpu = cp.asarray(m_b)
    start = time.time()

    res_gpu = cp.dot(matrix_a_gpu, matrix_b_gpu)
    res = cp.asnumpy(res_gpu)

    end = time.time()
    result_correct = np.dot(m_a, m_b)
    is_correct = 1 if np.allclose(res, result_correct) else 0
    return is_correct, end - start


matrix_mul_kernel = ("""
__global__ void matrixMulKernel(float* A, float* B, float* C, int Acols, int Bcols) {
    int i0 = Acols * (blockDim.y * blockIdx.y + threadIdx.y);
    int j0 = blockDim.x * blockIdx.x + threadIdx.x;

    if (j0 < Bcols) {
        float sum = 0;
        for (int k = 0; k < Acols; k++) sum += A[i0 + k] * B[k * Bcols + j0];
        int ind = Bcols * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
        C[ind] = sum;
    }
}
""")


def matrix_mult_gpu(m_a, m_b):
    if m_a.shape[1] != m_b.shape[0]:
        raise ValueError("Внутренние размерности не совпадают")

    block_size = (16, 16, 1)
    grid_size = ((m_b.shape[1] + block_size[0] - 1) // block_size[0],
                 (m_a.shape[0] + block_size[1] - 1) // block_size[1])
    res = np.zeros((m_a.shape[0], m_b.shape[1]), dtype=np.float32)

    start = time.time()

    A = cuda.mem_alloc(m_a.nbytes)
    B = cuda.mem_alloc(m_b.nbytes)
    C = cuda.mem_alloc(res.nbytes)

    cuda.memcpy_htod(A, m_a)
    cuda.memcpy_htod(B, m_b)

    mod = SourceModule(matrix_mul_kernel)
    matrix_mul = mod.get_function("matrixMulKernel")
    matrix_mul(A, B, C, np.int32(m_a.shape[1]),
               np.int32(m_b.shape[1]), block=block_size, grid=grid_size)

    cuda.Context.synchronize()
    cuda.memcpy_dtoh(res, C)

    end = time.time()
    result_correct = np.dot(m_a, m_b)
    is_correct = 1 if np.allclose(res, result_correct) else 0
    return is_correct, end - start


# Случайный эксперимент
matrix_n = (100, 200)
matrix_k = (200, 150)
matrix_A = np.random.uniform(1.0, 9.0, size=matrix_n)
matrix_B = np.random.uniform(1.0, 9.0, size=matrix_k)

result_cpu, time_cpu = matrix_mult_cpu(matrix_A, matrix_B)
result_gpu_lib, time_gpu_lib = matrix_mult_gpu_lib(matrix_A, matrix_B)
result_gpu, time_gpu = matrix_mult_gpu(matrix_A.astype(np.float32), matrix_B.astype(np.float32))

print(f"Время на CPU: {time_cpu} секунд\n",
      "Проверка корректности перемножения:", result_cpu)
print(f"Время на GPU, библиотечная реализация: {time_gpu_lib} секунд\n",
      "Проверка корректности перемножения:", result_gpu_lib)
print(f"Время на GPU: {time_gpu} секунд\n",
      "Проверка корректности перемножения:", result_gpu)

time_cpu = []
time_gpu = []
size_list = []
speedup = []
for size in range(100, 2001, 200):
    matrix_A = np.random.uniform(1.0, 9.0, size=(size, size))
    matrix_B = np.random.uniform(1.0, 9.0, size=(size, size))
    result_cpu, time_cpu_val = matrix_mult_cpu(matrix_A, matrix_B)
    result_gpu, time_gpu_val = matrix_mult_gpu(matrix_A.astype(np.float32), matrix_B.astype(np.float32))
    print("Проверка корректности gpu:", result_gpu)
    print("Проверка корректности cpu:", result_cpu)
    time_cpu.append(time_cpu_val)
    time_gpu.append(time_gpu_val)
    size_list.append(size)
    speedup.append(time_cpu_val / time_gpu_val)

plt.figure(figsize=(10, 6))
plt.plot(size_list, time_cpu, label='CPU Time', marker='o')
plt.plot(size_list, time_gpu, label='GPU Time', marker='x')
plt.title('Time Comparison: CPU vs GPU')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(size_list, speedup, label='Speedup (CPU/GPU)', marker='o')
plt.title('Speedup: CPU vs GPU')
plt.xlabel('Matrix Size')
plt.ylabel('Speedup')
plt.grid(True)
plt.legend()
plt.show()
