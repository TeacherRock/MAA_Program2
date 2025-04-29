import numpy as np
from scipy.integrate import solve_ivp
from utils.benchmark_func import BenchmarkFunction

class Inv_Pendulum_Car(BenchmarkFunction):
    def __init__(self, dimension):
        super().__init__("IPC", dimension, [-1.0, 1.0])

        # 參數設定
        self.g = 9.8      # 重力加速度 m/s^2
        self.M = 1.1      # 小車質量 kg
        self.mu_c = 0.1   # 小車摩擦係數
        self.mu_p = 0.01  # 擺摩擦係數

        self.m = 0.1     # 擺的質量 kg ( 0.1 < m< 0.3 )
        self.L = 0.5     # 擺長 m ( 0.5 < L < 1.5 )

        # 初始條件
        self.theta_0 = np.pi / 6 # 擺的初始角度 (+-)
        self.theta_dot_0 = 0.0   # 擺的初始角速度
        self.x_0 = 0.0       # 小車的初始位置
        self.x_dot_0 = 0.0   # 小車的初始速度
        self.y0 = [self.x_0, self.x_dot_0, self.theta_0, self.theta_dot_0]

        # 目標
        self.theta_target = 0.0      # 目標角度
        self.theta_dot_target = 0.0  # 擺的目標角速度

        self.f = 0.0     # 控制力 N
        self.f_max =  80.0 # 最大控制力 N ( |f| <  80.0 )
        self.f_min = -80.0 # 最小控制力 N ( |f| > -80.0 ) 

        # 目標訊號參數
        self.x_amplitude = 0.5  # ±0.5 m
        self.x_period    = 20.0    # 秒    

        self.force_log = [] # 儲存控制力的時間序列

        # 控制器參數
        ## PD-like ##
        self.Kp_x = 80.0
        self.Kv_x = 10.0
        self.Kp_theta = 10.0
        self.Kv_theta = 0.1

        self.controller_type = "PD-like"

        self.integral_error_x = 0.0
        self.integral_error_theta = 0.0
        self.last_error_x = 0.0
        self.last_error_theta = 0.0
        self.dt = 0.01

    def evaluate(self, x):

        # self.integral_error_x = 0.0
        # self.integral_error_theta = 0.0
        # self.last_error_x = 0.0
        # self.last_error_theta = 0.0

        self.integral_error_x = 0.0
        self.integral_error_theta = 0.0
        self.last_error_x = 0.0
        self.last_error_theta = 0.0

        _ = self.parameter_denormalize(x)
        solution = self.simulation()

        t = solution.t
        x_target = np.array([self.x_target_func(t) for t in solution.t])

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
        
        # return_value = (theta_error_avg + theta_error_final) + 10.0*(x_error_avg + x_error_final)
        return_value = (theta_error_avg + theta_error_final) + 20.0*(x_error_avg + 0*x_error_final)

        if return_value < 3.0:
            print("=================================================================")
            print("x_error_avg :", x_error_avg)
            print("x_error_final :", x_error_final)
            print("theta_error_avg :", theta_error_avg)
            print("theta_error_final :", theta_error_final)
            print("return : ", return_value)
            print("=================================================================")

        return return_value

    def simulation(self):
        # from [-1, 1] map to [0.0, 200.0]

        t_span = (0, 60)
        t_eval = np.linspace(*t_span, int(60/self.dt))
        solution = solve_ivp(self.pendulum_cart_dynamics, t_span, self.y0, t_eval=t_eval)

        return solution
    
    def parameter_denormalize(self, x):
        if self.controller_type == "PD-like":
            self.Kp_x = (x[0] - (-1)) / 2.0 *  30.0
            self.Kv_x = (x[1] - (-1)) / 2.0 * -100.0

            self.Kp_theta = (x[2] - (-1)) / 2.0 *  100.0
            self.Kv_theta = (x[3] - (-1)) / 2.0 * -100.0

            param = {"Kp_x" : self.Kp_x, 
                    "Kv_x" : self.Kv_x, 
                    "Kp_theta" : self.Kp_theta, 
                    "Kv_theta" : self.Kv_theta}
            
        elif self.controller_type == "PID":
            self.Kp_x = (x[0] - (-1)) / 2.0 * 5.0     # Kp_x: 0 ~ 
            self.Ki_x = (x[1] - (-1)) / 2.0 * 0.25      # Ki_x: 0 ~ 
            self.Kd_x = (x[2] - (-1)) / 2.0 * 0.5      # Kd_x: 0 ~ 

            self.Kp_theta = (x[3] - (-1)) / 2.0 * 80.0  # Kp_theta: 0 ~ 
            self.Ki_theta = (x[4] - (-1)) / 2.0 * 8.0  # Ki_theta: 0 ~ 
            self.Kd_theta = (x[5] - (-1)) / 2.0 * 80.0  # Kd_theta: 0 ~ 

            param = {"Kp_x": self.Kp_x,
                     "Ki_x": self.Ki_x,
                     "Kd_x": self.Kd_x,
                     "Kp_theta": self.Kp_theta,
                     "Ki_theta": self.Ki_theta,
                     "Kd_theta": self.Kd_theta}
 
        return param

    def parameter_normalize(self, param):
        x = np.zeros(self.dimension)
        x[0] = param[0] * 2.0 / 500.0 + (-1)
        x[1] = param[1] * 2.0 / 500.0 + (-1)
        x[2] = param[2] * 2.0 / 100.0 + (-1)
        x[3] = param[3] * 2.0 / 100.0 + (-1)

        return x

    def control_force(self, t, state):
        x, x_dot, theta, theta_dot = state

        # 根據時間產生參考命令
        x_target = self.x_target_func(t)
        theta_target = 0.0

        if self.controller_type == "PD-like":
            error_x = x_target - x
            error_theta = theta_target - theta

            u_x = self.Kv_x * (self.Kp_x * (error_x) - x_dot)
            u_theta = self.Kv_theta * (self.Kp_theta * (error_theta) - theta_dot)

            # print(f"(PD-like) u_x : {u_x}, u_theta : {u_theta}")
            f = u_x + u_theta

        elif self.controller_type == "PID":
            # 誤差
            error_x = x - x_target
            error_theta = theta - theta_target

            # 誤差積分項
            self.integral_error_x += 0.5 * (error_x + self.last_error_x) * self.dt
            self.integral_error_theta += 0.5 * (error_theta + self.last_error_theta) * self.dt

            # 誤差微分項
            derivative_error_x = (error_x - self.last_error_x) / self.dt
            derivative_error_theta = (error_theta - self.last_error_theta) / self.dt

            self.last_error_x = error_x
            self.last_error_theta = error_theta

            # PID 控制律
            u_x = (self.Kp_x * error_x +
                   self.Ki_x * self.integral_error_x +
                   self.Kd_x * derivative_error_x)

            u_theta = (self.Kp_theta * error_theta +
                       self.Ki_theta * self.integral_error_theta +
                       self.Kd_theta * derivative_error_theta)
            # print(f"(PID) u_x : {0.5 * u_x}, u_theta : {self.f_max * u_theta}")
            f = u_x + u_theta

        f = np.clip(f, self.f_min, self.f_max)  # 限制在 ±80N

        self.force_log.append((t, f))

        return f

    def x_target_func(self, t):
        # 矩形波：每半週期改變一次符號
        phase = (t % self.x_period)
        return self.x_amplitude if phase < (self.x_period / 2) else -self.x_amplitude
    
    # 定義系統微分方程 dx/dt = f(x, t)
    def pendulum_cart_dynamics(self, t, y):
        # print("t:", t, "y:", y)  # Debugging line
        x, x_dot, theta, theta_dot = y
        f = self.control_force(t, y)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        theta_dot2 = theta_dot ** 2
        sgn_x = np.sign(x_dot)

        # theta 的二階導數
        numerator_theta_ddot = ((self.M + self.m) * self.g * sin_theta 
                                - cos_theta * (f + self.m * self.L * theta_dot2 * sin_theta - (self.M + self.m) * self.mu_c * sgn_x)
                                - self.mu_p * (self.M + self.m) * theta_dot / self.m / self.L)
        denominator_theta_ddot = (4 / 3) * (self.M + self.m) * self.L - self.m * self.L * cos_theta ** 2
        theta_ddot = numerator_theta_ddot / denominator_theta_ddot

        # x 的二階導數
        x_ddot = (f + self.m * self.L * (theta_dot2 * sin_theta - theta_ddot * cos_theta)) / (self.M + self.m) - self.mu_c * sgn_x

        return [x_dot, x_ddot, theta_dot, theta_ddot]
