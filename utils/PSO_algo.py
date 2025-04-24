# Particle Swarm Optimization (PSO)
import numpy as np
from tqdm import tqdm

class PSO:
    def __init__(self, obj_func, n_particles=40, max_iter=100, w=0.5, c1=2.0, c2=2.0):
        self.obj_func = obj_func
        self.n_particles = n_particles
        self.iterations = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.dim = obj_func.dimension
        self.lower_bound = obj_func.search_range[0]
        self.upper_bound = obj_func.search_range[1]

        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (n_particles, self.dim))
        self.velocities = np.zeros((n_particles, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.array([self.obj_func.evaluate(p) for p in self.positions])

        gbest_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[gbest_idx].copy()
        self.global_best_score = self.personal_best_scores[gbest_idx]

        self.search_track = []

    def run(self):
        with tqdm(total=self.iterations, desc=f"PSO Algorithm", unit="iter.") as pbar:
            for idx in range(self.iterations):
                r1 = np.random.rand(self.n_particles, self.dim)
                r2 = np.random.rand(self.n_particles, self.dim)

                cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
                social = self.c2 * r2 * (self.global_best_position - self.positions)
                self.velocities = self.w * self.velocities + cognitive + social
                self.positions += self.velocities
                self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

                scores = np.array([self.obj_func.evaluate(p) for p in self.positions])

                improved = scores < self.personal_best_scores
                self.personal_best_positions[improved] = self.positions[improved]
                self.personal_best_scores[improved] = scores[improved]

                gbest_idx = np.argmin(self.personal_best_scores)
                if self.personal_best_scores[gbest_idx] < self.global_best_score:
                    self.global_best_score = self.personal_best_scores[gbest_idx]
                    self.global_best_position = self.personal_best_positions[gbest_idx].copy()

                self.search_track.append(self.global_best_score)

                pbar.set_postfix({
                    "Iteration": f"{idx+1}/{self.iterations}",
                    "Best fitness": f"{self.global_best_score}"})
                pbar.update(1)

        return self.global_best_position, self.global_best_score

if __name__ == '__main__':
    pass