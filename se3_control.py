import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2
        self.gamma = self.k_drag / self.k_thrust

        # STUDENT CODE HERE
        
        # kp = 1, T = 6s
        # .6 .8 
        # scale different
        
        # 8 8 5 
        self.K_d = np.array([
            [13.0, 0, 0],
            [0, 13.0, 0],
            [0, 0, 11]
        ])
        
        # 15 15 11
        self.K_p = np.array([   
            [25, 0, 0],
            [0, 25, 0],
            [0, 0, 28]
        ]) 
        
        # 400 400 2.5
        self.K_R = np.array([
            [3500.0, 0, 0],
            [0, 3500, 0],
            [0, 0, 23]
        ])
        
        # 9 9 1
        self.K_omega = np.array([
            [130.0, 0, 0],
            [0, 130.0, 0],
            [0, 0, 8]
        ])
        # wb effect
        # z > ori > pos > vel
        # print('K_d', self.K_d)
        # print('K_p', self.K_p)
        # print('K_R', self.K_R)
        # print('K_omega', self.K_omega)

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        
        # step 1 
        r_dd_des = flat_output['x_ddot'] - self.K_d @ (state['v']  - flat_output['x_dot']) - self.K_p @ (state['x'] - flat_output['x'])
        F_des = self.mass * (r_dd_des + np.array([0, 0, self.g]))
        
        # step 2
        R = Rotation.from_quat(state['q']).as_matrix()
        b_3 = R @ np.array([0, 0, 1])
        u_1 = b_3 @ F_des
        
        # step 3
        b_3_des = F_des / np.linalg.norm(F_des)
        a_psi = np.array([np.cos(flat_output['yaw']), np.sin(flat_output['yaw']), 0])
        b_2_des = np.cross(b_3_des, a_psi)
        # b_2_des = b_2_des / np.linalg.norm(b_2_des)
        # b_1_des = np.cross(b_2_des, b_3_des)
        b_1_des = a_psi
        R_des = np.column_stack((b_1_des, b_2_des, b_3_des))

        # step 4
        e_R = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R[2, 1], e_R[0, 2], e_R[1, 0]]) 
        e_omega = state['w'] - np.zeros((3,))
        
        # step 5
        M_des = -self.K_R @ e_R - self.K_omega @ e_omega
        u_2 = self.inertia @ M_des
        
        # compute the ouput
        A = np.array([[1, 1, 1, 1],
                      [0, self.arm_length, 0, -self.arm_length],
                      [-self.arm_length, 0, self.arm_length, 0],
                      [self.gamma, -self.gamma, self.gamma, -self.gamma]])
        F = np.linalg.inv(A) @ np.array([u_1, u_2[0], u_2[1], u_2[2]])
        F[F < 0] = 0
        cmd_motor_speeds = np.sqrt(F / self.k_thrust)
        cmd_thrust = u_1
        cmd_moment = u_2
                      
        cmd_motor_speeds = cmd_motor_speeds.clip(self.rotor_speed_min, self.rotor_speed_max)

        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
