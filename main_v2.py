from utils.inverted_pendulum_car import Inv_Pendulum_Car
from utils.ABC_algo import ABC
from utils.GA_algo import GA
from utils.PSO_algo import PSO

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def find_solution(algorithm, controller_type="PID", save_folder="/reocrd/"):
    if controller_type == "PID":
        inverted_pendulum_car = Inv_Pendulum_Car(dimension=6)
        inverted_pendulum_car.controller_type = "PID"
    elif controller_type == "PD-like":
        inverted_pendulum_car = Inv_Pendulum_Car(dimension=4)
        inverted_pendulum_car.controller_type = "PD-like"

    if algorithm == "ABC":
        Algo = ABC(inverted_pendulum_car, food_count=10, max_iter=40)
        solution, cost = Algo.run()
    elif algorithm == "GA":
        Algo = GA(inverted_pendulum_car, pop_size=10, max_iter=40)
        solution, cost = Algo.run()
    else:
        Algo = PSO(inverted_pendulum_car, n_particles=10, max_iter=40)
        solution, cost = Algo.run()

    np.savetxt(save_folder + f"{controller_type}_{algorithm}_fitness.txt", Algo.search_track)
    
    # Draw Fitness Curve
    plt.figure()
    plt.plot(Algo.search_track)
    plt.xlabel("Time (s)")
    plt.title(f"{algorithm} Algorithm Fitness Curve")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_fitness_curve.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_fitness_curve.svg")

    return solution

def infernece(x, controller_type="PID", save_folder="/reocrd/"):
    if controller_type == "PID":
        inverted_pendulum_car = Inv_Pendulum_Car(dimension=6)
        inverted_pendulum_car.controller_type = "PID"
    elif controller_type == "PD-like":
        inverted_pendulum_car = Inv_Pendulum_Car(dimension=4)
        inverted_pendulum_car.controller_type = "PD-like"

    param = inverted_pendulum_car.parameter_denormalize(x)
    solution = inverted_pendulum_car.simulation()

    x_target = np.array([inverted_pendulum_car.x_target_func(t) for t in solution.t])

    x_track         = solution.y[0]  # 小車的位置
    x_dot_track     = solution.y[1]  # 小車的速度
    theta_track     = solution.y[2]  # 擺的角度
    theta_dot_track = solution.y[3]  # 擺的角速度

    x_final         = solution.y[0][-1]  # 小車的最終位置
    x_dot_final     = solution.y[1][-1]  # 小車的最終速度
    theta_final     = solution.y[2][-1]  # 擺的最終角度
    theta_dot_final = solution.y[3][-1]  # 擺的最終角速度

    x_error_avg = np.linalg.norm(x_target - x_track) / np.sqrt(len(x_track)) # RMSE of x
    x_error_final = np.abs(x_final - x_target[-1]) # Final error of x 

    theta_error_avg = np.linalg.norm(theta_track) / np.sqrt(len(theta_track)) # RMSE of theta
    theta_error_final = np.abs(theta_final) # Final error of theta

    print("x_error_avg :", x_error_avg)
    print("x_error_final :", x_error_final)
    print("theta_error_avg :", theta_error_avg)
    print("theta_error_final :", theta_error_final)

    print("return : ", (theta_error_avg + theta_error_final) + 20.0*(x_error_avg + 0*x_error_final))

    # theta, theta_dot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, solution.y[2], label='Inverted Pendulum angle θ')
    plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="Target angle")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Inverted Pendulum on a Cart : System States")

    plt.subplot(2, 1, 2)
    plt.plot(solution.t, solution.y[3], label='Inverted Pendulum angular velocity θ_dot')
    plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.tight_layout()

    plt.savefig(save_folder + f"{controller_type}_{algorithm}_theta_system_states.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_theta_system_states.svg")

    # theta
    plt.figure()
    plt.plot(solution.t, solution.y[2], label='Inverted Pendulum angle θ')
    plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="Target angle")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.title("Inverted Pendulum angle")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_theta.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_theta.svg")

    # x, x_dot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(solution.t, solution.y[0], label='Cart position x')
    plt.plot(solution.t, [inverted_pendulum_car.x_target_func(t) for t in solution.t], '--', label='Reference x_target', alpha=0.6)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Inverted Pendulum on a Cart : System States")

    plt.subplot(2, 1, 2)
    plt.plot(solution.t, solution.y[1], label='Cart velocity x_dot')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.tight_layout()
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_x_system_states.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_x_system_states.svg")

    # x
    plt.figure()
    plt.plot(solution.t, solution.y[0], label='Cart position x')
    plt.plot(solution.t, [inverted_pendulum_car.x_target_func(t) for t in solution.t], '--', label='Reference x_target', alpha=0.6)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Cart position x'")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_x.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_x.svg")

    # Control Force
    force_log = np.array(inverted_pendulum_car.force_log)
    t_force = force_log[:, 0]
    f_values = force_log[:, 1]

    plt.figure(figsize=(10, 4))
    plt.plot(t_force, f_values, label='Control Force f(t)', color='purple')
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Control Force vs Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_control_force.jpg")
    plt.savefig(save_folder + f"{controller_type}_{algorithm}_control_force.svg")

    # plt.show()

if __name__ == '__main__':
    find_flag = True
    find_flag = False

    # algorithm = "ABC"
    # algorithm = "PSO"
    algorithm = "GA"

    # controller_type="PID"
    controller_type="PD-like"

    save_folder = "./reocrd/" + controller_type + "/" + algorithm + "/"
    os.makedirs(save_folder, exist_ok=True)

    if find_flag:
        solution = find_solution(algorithm, controller_type, save_folder)

        with open(save_folder + "solution.txt", "w") as file:
            file.write(str(solution))

        with open(save_folder + "solution.pkl", "wb") as f:
            pickle.dump(solution, f)
    else:

        # with open(save_folder + "solution.txt", "r") as file:
        #     content = file.read()
        #     loaded_dict = eval(content)

        with open(save_folder + "solution.pkl", "rb") as f:
            solution = pickle.load(f)
        
        infernece(solution, controller_type, save_folder)