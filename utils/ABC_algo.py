# Artificial Bee Colony (ABC)
import numpy as np
from tqdm import tqdm

class ABC:
    def __init__(self, obj_func, food_count=40, limit=100, max_iter=100):
        self.obj_func = obj_func
        self.food_count = food_count
        self.limit = limit
        self.max_iter = max_iter
        self.dim = obj_func.dimension
        self.lower_bound = obj_func.search_range[0]
        self.upper_bound = obj_func.search_range[1]
        self.search_track = []

    def initialize_food_sources(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.food_count, self.dim))

    def evaluate(self, sources):
        return np.array([self.obj_func.evaluate(src) for src in sources])

    def run(self):
        food_sources = self.initialize_food_sources()
        fitness = self.evaluate(food_sources)
        trial = np.zeros(self.food_count)

        best_index = np.argmin(fitness)
        best_solution = food_sources[best_index].copy()
        best_fitness = fitness[best_index]

        with tqdm(total=self.max_iter, desc=f"ABC Algorithm", unit="iter.") as pbar:
            for idx in range(self.max_iter):
                # Employed bees phase
                for i in range(self.food_count):
                    phi = np.random.uniform(-1, 1, self.dim)
                    k = np.random.choice([j for j in range(self.food_count) if j != i])
                    candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = self.obj_func.evaluate(candidate)
                    if candidate_fitness < fitness[i]:
                        food_sources[i] = candidate
                        fitness[i] = candidate_fitness
                        trial[i] = 0
                    else:
                        trial[i] += 1

                # Onlooker bees phase
                prob = fitness.max() - fitness
                prob = prob / prob.sum()
                for i in range(self.food_count):
                    if np.random.rand() < prob[i]:
                        phi = np.random.uniform(-1, 1, self.dim)
                        k = np.random.choice([j for j in range(self.food_count) if j != i])
                        candidate = food_sources[i] + phi * (food_sources[i] - food_sources[k])
                        candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                        candidate_fitness = self.obj_func.evaluate(candidate)
                        if candidate_fitness < fitness[i]:
                            food_sources[i] = candidate
                            fitness[i] = candidate_fitness
                            trial[i] = 0
                        else:
                            trial[i] += 1

                # Scout bees phase
                for i in range(self.food_count):
                    if trial[i] > self.limit:
                        food_sources[i] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                        fitness[i] = self.obj_func.evaluate(food_sources[i])
                        trial[i] = 0

                # Update best solution
                best_index = np.argmin(fitness)
                if fitness[best_index] < best_fitness:
                    best_fitness = fitness[best_index]
                    best_solution = food_sources[best_index].copy()
                    # np.savetxt("best.txt", [best_fitness])
                    # np.savetxt("best_sol.txt", [best_solution])

                
                self.search_track.append(best_fitness)
                pbar.set_postfix({
                    "Iteration": f"{idx+1}/{self.max_iter}",
                    "Best fitness": f"{best_fitness}"})
                pbar.update(1)

        print("best_solution : ", best_solution)

        return best_solution, best_fitness

if __name__ == '__main__':
    pass
