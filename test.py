from utils.inverted_pendulum_car import Inv_Pendulum_Car
from utils.ABC_algo import ABC
from utils.GA_algo import GA
from utils.PSO_algo import PSO

import matplotlib.pyplot as plt
import numpy as np
import pickle

inverted_pendulum_car = Inv_Pendulum_Car(dimension=6)
inverted_pendulum_car.controller_type = "PID"
inverted_pendulum_car.Kp_x = 5.0/2 # 3.9 
inverted_pendulum_car.Ki_x = 0.1/2 # 0.5
inverted_pendulum_car.Kd_x = 0.5/2 # 0.43

gain = 10.0/80.0
inverted_pendulum_car.Kp_theta = 4.5874 / gain # 0.8
inverted_pendulum_car.Ki_theta = 0.3940 / gain # 0.094
inverted_pendulum_car.Kd_theta = 2.8045 / gain # 0.5

# inverted_pendulum_car = Inv_Pendulum_Car(dimension=4)
# inverted_pendulum_car.controller_type = "PD-like"

# inverted_pendulum_car.Kp_x =  15.0 
# inverted_pendulum_car.Kv_x = -50.0

# inverted_pendulum_car.Kp_theta =  100.0
# inverted_pendulum_car.Kv_theta = -50.0

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

print("return : ", (theta_error_avg + theta_error_final) + 20 * (x_error_avg + x_error_final))

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
plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
plt.legend()
plt.title("Cart-Pendulum System States")

plt.subplot(2, 1, 2)
plt.plot(solution.t, solution.y[1], label='Cart velocity x_dot')
plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
plt.legend()
plt.xlabel("Time [s]")
plt.tight_layout()

# x tracking peformance
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='Cart position x')
plt.plot(solution.t, [inverted_pendulum_car.x_target_func(t) for t in solution.t], '--', label='Reference x_target', alpha=0.6)
plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
plt.legend()
plt.title("Cart Position Tracking with PID Control")
plt.xlabel("Time [s]")
plt.ylabel("Position [m]")
plt.grid(True)
plt.tight_layout()

# x tracking peformance
# plt.figure(figsize=(10, 5))
# plt.plot(solution.t, solution.y[0], label='Cart position x')
# plt.plot(solution.t, np.array([inverted_pendulum_car.x_target_func(t) for t in solution.t] - solution.y[0]), '--', label='x_error', alpha=0.6)
# plt.hlines(0.0, xmax=60.0, xmin=0.0, colors="r", label="target angular velocity")
# plt.legend()
# plt.title("Cart Position Tracking with PID Control")
# plt.xlabel("Time [s]")
# plt.ylabel("Position [m]")
# plt.grid(True)
# plt.tight_layout()

plt.show()