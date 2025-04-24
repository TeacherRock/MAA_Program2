import numpy as np

class BenchmarkFunction:
    def __init__(self, name, dimension, search_range, optimal_x=None, optimal_value=None):
        self.name = name
        self.dimension = dimension
        self.search_range = search_range
        self.optimal_x = optimal_x
        self.optimal_value = optimal_value

    def evaluate(self, x):
        raise NotImplementedError("Each function must implement its own evaluation method.")
    
    def evaluate_batch(self, X):
        return np.array([self.evaluate(x) for x in X])
    
if __name__ == "__main__":
    pass