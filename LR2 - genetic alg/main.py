import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import random
import math
import time

calculate_fitness_kernel = ("""
__global__ void calculateFitness(double* population, double* x, double* y, double* fitness, int populationSize, 
int polynomialOrder, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        double sumSquaredError = 0.0;
        for (int i = 0; i < points; ++i) {
            double approx = 0.0;
            for (int j = 0; j <= polynomialOrder; ++j) approx += population[idx * (polynomialOrder + 1) + j] * pow(x[i], j);
            double error = y[i] - approx;
            sumSquaredError += error * error;
        }
        fitness[idx] = sumSquaredError / points;
    }
}
""")


def gpu_fitness_evaluation(population, x, y, polynomial_order):
    population_size = population.shape[0]
    points = x.size
    population_bytes = population.nbytes
    x_bytes = x.nbytes
    y_bytes = y.nbytes
    fitness_bytes = population_size * np.dtype(np.float64).itemsize
    d_population = cuda.mem_alloc(population_bytes)
    d_x = cuda.mem_alloc(x_bytes)
    d_y = cuda.mem_alloc(y_bytes)
    cuda.mem_alloc(fitness_bytes)
    cuda.memcpy_htod(d_population, population)
    cuda.memcpy_htod(d_x, x)
    cuda.memcpy_htod(d_y, y)
    module = SourceModule(calculate_fitness_kernel)
    calculate_fitness = module.get_function("calculateFitness")
    fitness = np.zeros(population_size, dtype=np.float64)
    threads_per_block = 256
    blocks = (population_size + threads_per_block - 1) // threads_per_block

    calculate_fitness(d_population, d_x, d_y, cuda.Out(fitness),
                      np.int32(population_size), np.int32(polynomial_order), np.int32(points),
                      block=(threads_per_block, 1, 1), grid=(blocks, 1))

    return fitness


def tournament_selection(fitness_scores, selection_size=5):
    candidates = random.sample(range(len(fitness_scores)), selection_size)
    return min(candidates, key=lambda idx: fitness_scores[idx])


def crossover(parent1, parent2, crossover_rate=0.7):
    mask = np.random.rand(len(parent1)) < crossover_rate
    return np.where(mask, parent2, parent1)


def mutate(child, mutation_rate, mutation_range):
    if random.random() < mutation_rate:
        child += np.random.uniform(-mutation_range, mutation_range, child.size)
    return child


def genetic_algorithm(x, y, population_size, polynomial_order, generations, base_mutation_rate, err):
    population = np.random.uniform(-1, 1, (population_size, polynomial_order + 1))
    for generation in range(generations):
        # Evaluate the individual fitness of individuals in population
        fitness_scores = gpu_fitness_evaluation(population, x, y, polynomial_order)
        best_idx = np.argmin(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        print(f"Generation {generation}: Best fit = {best_fitness}")
        if best_fitness < err:
            return population[best_idx], generation, best_fitness
        mutation_rate = base_mutation_rate * math.exp(-generation / generations)
        new_population = [population[best_idx]]
        while len(new_population) < population_size:
            # Selection (Select the best-fit individuals for reproduction in next population)
            parent1 = population[tournament_selection(fitness_scores)]
            parent2 = population[tournament_selection(fitness_scores)]
            # . Crossover (birth of new individuals)
            child = crossover(parent1, parent2)
            # . Mutation
            mutation_range = (1.0 - generation / generations) * 10
            child = mutate(child, mutation_rate, mutation_range)
            new_population.append(child)
        population = np.array(new_population)
    return population[0], generations, best_fitness


def main():
    polynomial_order = 4
    population_size = 1000  # Размер популяции
    generations = 500
    base_mutation_rate = 0.95
    err = 1e-6
    points_count = 1000  # Набор точек на поверхности
    coefficients = np.random.uniform(-100, 100, polynomial_order + 1)

    x = np.random.uniform(-10, 10, points_count)
    y = np.array([sum(coefficients[j] * (x[i] ** j) for j in range(polynomial_order + 1)) for i in range(points_count)])
    start_time = time.time()
    solution, evaluated_generations, best_fitness = genetic_algorithm(
        x, y, population_size, polynomial_order, generations, base_mutation_rate, err)

    print(f"GPU Time: {time.time() - start_time} секунд")
    print("Original coefficients:", coefficients)
    print("Found coefficients:", solution)
    print("Fitness:", best_fitness)
    print("Evaluated generations:", evaluated_generations)


if __name__ == "__main__":
    main()
