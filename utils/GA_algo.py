# Genetic Algorithm (GA)
import numpy as np
from tqdm import tqdm

class GA:
    def __init__(self, obj_func, pop_size=40, max_iter=100, mutation_rate=0.1):
        self.obj_func = obj_func
        self.pop_size = pop_size
        self.generations = max_iter
        self.mutation_rate = mutation_rate
        self.dim = obj_func.dimension
        self.lower_bound = obj_func.search_range[0]
        self.upper_bound = obj_func.search_range[1]
        self.search_track = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def evaluate_population(self, population):
        return np.array([self.obj_func.evaluate(ind) for ind in population])

    def select_parents(self, population, fitness):
        idx = np.argsort(fitness)
        return population[idx[:2]]

    def crossover(self, p1, p2):
        alpha = np.random.rand(self.dim)
        child1 = alpha * p1 + (1 - alpha) * p2
        child2 = alpha * p2 + (1 - alpha) * p1
        return child1, child2

    def mutate(self, individual):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.uniform(self.lower_bound, self.upper_bound)
        return individual

    def run(self):
        pop = self.initialize_population()
        best_solution = None
        best_fitness = float('inf')
        with tqdm(total=self.generations, desc=f"GA Algorithm", unit="iter.") as pbar:
            for idx in range(self.generations):
                fitness = self.evaluate_population(pop)
                new_pop = []
                for _ in range(self.pop_size // 2):
                    parents = self.select_parents(pop, fitness)
                    child1, child2 = self.crossover(*parents)
                    new_pop.append(self.mutate(child1))
                    new_pop.append(self.mutate(child2))
                pop = np.clip(np.array(new_pop), self.lower_bound, self.upper_bound)
                current_best = np.min(fitness)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_solution = pop[np.argmin(fitness)]

                self.search_track.append(best_fitness)

                pbar.set_postfix({
                    "Iteration": f"{idx+1}/{self.generations}",
                    "Best fitness": f"{best_fitness}"})
                pbar.update(1)
            
        return best_solution, best_fitness

if __name__ == '__main__':
    pass