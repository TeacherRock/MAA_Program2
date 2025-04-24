from utils.inverted_pendulum_car import Inv_Pendulum_Car
from utils.ABC_algo import ABC
from utils.GA_algo import GA
from utils.PSO_algo import PSO

import matplotlib.pyplot as plt
import numpy as np

def main():
    inverted_pendulum_car = Inv_Pendulum_Car(dimension=4)
    abc = ABC(inverted_pendulum_car, food_count=10, max_iter=10)
    solution, cost = abc.run()
    print(cost)
    return solution

def infernece(x):
    inverted_pendulum_car = Inv_Pendulum_Car(dimension=4)
    solution = inverted_pendulum_car.simulation(x)

    # theta, theta_dot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, solution.y[2], label='Pendulum angle θ')
    plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angle")
    plt.legend()
    plt.title("Cart-Pendulum System States")

    plt.subplot(2, 1, 2)
    plt.plot(solution.t, solution.y[3], label='Pendulum angular velocity θ_dot')
    plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.tight_layout()

    # x, x_dot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, solution.y[0], label='Cart position x')
    plt.legend()
    plt.title("Cart-Pendulum System States")

    plt.subplot(2, 1, 2)
    plt.plot(solution.t, solution.y[1], label='Cart velocity x_dot')
    plt.legend()
    plt.xlabel("Time [s]")
    plt.tight_layout()

    # x tracking peformance
    plt.figure(figsize=(10, 5))
    plt.plot(solution.t, solution.y[0], label='Cart position x')
    plt.plot(solution.t, [inverted_pendulum_car.x_target_func(t) for t in solution.t], '--', label='Reference x_target', alpha=0.6)
    plt.legend()
    plt.title("Cart Position Tracking with PID Control")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.grid(True)
    plt.tight_layout()

    # 轉換控制力資料
    # force_log = np.array(force_log)
    # t_force = force_log[:, 0]
    # f_values = force_log[:, 1]

    # # 畫出控制力 f(t)
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_force, f_values, label='Control Force f(t)', color='purple')
    # plt.xlabel("Time [s]")
    # plt.ylabel("Force [N]")
    # plt.title("Control Force vs Time")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    solution = main()
    object = Inv_Pendulum_Car(dimension=4)
    print(object.parameter_denormalize(solution))
    infernece(solution)