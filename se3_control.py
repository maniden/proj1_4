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
#using crazyfly 7
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

        # STUDENT CODE HERE

        # self.kp= (7.9*.3) * np.eye(3)
        #kp[2,2] += 11;
        #self.kd= (4.2 *.3)* np.eye(3)
        #self.kr= (135.0 *.3) * np.eye(3)
        #self.kw= (17.4*.3) * np.eye(3)
        self.kp = np.array([[4.2, 0, 0],
                            [0, 4.2, 0],
                            [0, 0, 25]])
                           # [0, 0, 3.9]])
        self.kd = np.array([[5.0, 0, 0],
                            [0, 5.0, 0],
                            [0, 0, 10.1]]) 
        self.kr = np.array([[40.0, 0, 0],
                            [0, 40.0, 0],
                            [0, 0, 40.5]]) 
        self.kw = np.array([[28, 0, 0],
                            [0, 28, 0],
                            [0, 0, 5]]) #23,5

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
        x,lv,q,w= (state['x'],state['v'],state['q'],state['w'])
        p,v,a,j,s,yaw,yawd=(flat_output['x'],flat_output['x_dot'],
                            flat_output['x_ddot'],flat_output['x_dddot'],flat_output['x_ddddot'],
                            flat_output['yaw'],flat_output['yaw_dot'])


        # 1.Use Eq(26) to compute the commanded acceleration  ̈rdes.
        rdT= a - self.kd @ (lv-v) - self.kp @ (x-p)

        # 1. calculate Fdes from Eq (33), (32), and (31).
        Fdes = self.mass * rdT + np.array([0, 0, self.mass * self.g])

        # 2. compute u1 from Eq (34)
        R= Rotation.from_quat(q).as_matrix()
        u1= np.dot(R[:,2], Fdes)

        # 3. determine Rdes from Eq (38) and the definitions of bides.
        # b3des = Fdes
        # ||Fdes||
        b3des= Fdes/np.linalg.norm(Fdes)
        apsi= np.array([np.cos(yaw), np.sin(yaw), 0])
        b2des= np.cross(b3des, apsi)/np.linalg.norm(np.cross(b3des, apsi))

        # Rdes = [b2des × b3des, b2des, b3des]
        Rdes= np.column_stack([np.cross(b2des,b3des),b2des,b3des])

        # 4. find the error orientation error vector eR from Eq (39) and substitute ω for eω.
        er = 0.5 * ((Rdes.T @ R) - (R.T @ Rdes))
        er = np.array([er[2, 1], er[0, 2], er[1, 0]])

        ew= w - 0.0
        # 5. compute u2 from Eq (40)
        u2= (self.inertia @ (-self.kr@er - self.kw@w))

        gamma= self.k_drag/self.k_thrust
        l=self.arm_length

        mat= np.array([
            [1, 1, 1, 1],
            [0, l, 0, -l],
            [-l, 0, l, 0],
            [gamma, -gamma, gamma, -gamma]
        ])
        u1 = np.array([u1])
        inp = np.linalg.inv(mat) @ np.concatenate((u1, u2))
        inp = np.clip(inp, 0, None)
        cmd_motor_speeds= np.sqrt(np.maximum(inp / self.k_thrust, 0))
        cmd_thrust= u1
        cmd_moment= u2
        cmd_q= Rotation.from_matrix(Rdes).as_quat()


        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
