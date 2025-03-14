from calendar import c
from os import path
from cycler import V
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from scipy.optimize import minimize

from .graph_search import graph_search

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.2, 0.2, 0.2])
        self.margin = 0.3

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        if self.path is not None:
            self.points = self._rdp(self.path[:], 0.25)
        # self.points = np.vstack((self.points, goal))
        # print(self.points)
        # self._plot_waypoints()
        # if self.points.shape[0] %2 == 0:
        #     self.points_1 = np.vstack(self.points[0], (self.points[0] + self.points[1])/2)
        #     self.points = np.vstack(self.points_1, self.points[1:])

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.
        
        # Based on the segment length, calculate the time duration for each segment
        self.T = np.zeros((self.points.shape[0],))
        for i in range(self.points.shape[0]-1):
            self.T[i+1] = np.linalg.norm(self.points[i+1]-self.points[i])
        self.T[1:-1] = np.log(self.T[1:-1]/2.125+1) 
        self.T[-1]*= 1.3
        # self.T[1] *= 1.2
        self.T_table = np.cumsum(self.T)
        # print(self.T)
        # print(self.T_table)
        
        # path_length = 0
        # for i in range(self.points.shape[0]-1):
        #     path_length += np.linalg.norm(self.points[i+1]-self.points[i])
        
        # v_design = path_length / self.T_table[-1]
        # print(v_design)
        v_design = 2.95
        
        # Solve for the polynomial coefficients for each segment
        self.poly_coeff = np.zeros((self.points.shape[0]-1,3,6))
        for i in range(self.points.shape[0]-1):
            state_dict = {
                'start': self.points[i],
                'end': self.points[i+1],
                'mid': None,
                'T': self.T[i+1],
                'v_start': np.zeros((3,)),
                'v_end': np.zeros((3,)),
                'a_start': np.zeros((3,)),
                'a_end': np.zeros((3,))
            }           
            
            # Compute mid point
            
                
            
            # Smooth Condition
            if i == 0:
                _, v_2 = self._velocity_generation(np.vstack((np.zeros((3,)), self.points[i:i+3])), v_design)
                state_dict['a_start'] = self._set_vector((self.points[i+1]-self.points[i]), 0)
                state_dict['v_end'] = self._set_vector((self.points[i+2]-self.points[i]), v_2)
                state_dict['a_end'] = self._set_vector((self.points[i+2]-self.points[i+1]) , 0)
                pass
            elif i < self.points.shape[0]-3:
                _, v_0, a_0 = self._poly(self.poly_coeff[i-1], self.T[i])
                _, v_2 = self._velocity_generation(self.points[i-1:i+3], v_design)
                state_dict['v_start'] = v_0
                state_dict['a_start'] = a_0
                state_dict['v_end'] = self._set_vector((self.points[i+2]-self.points[i]), v_2)
                # if np.linalg.norm(self.points[i+2]-self.points[i+1]) < np.linalg.norm(self.points[i+1]-self.points[i]):
                #     state_dict['a_end'] = np.zeros((3,))
                # else:
                #     state_dict['v_end'] = self._set_vector((self.points[i+2]-self.points[i]), 0.3)
                # state_dict['a_end'] = 

            elif i < self.points.shape[0]-2:
                _, v_0, a_0 = self._poly(self.poly_coeff[i-1], self.T[i])
                # v_1, v_2 = self._velocity_generation(self.points[i-2:i+2], v_design)
                state_dict['v_start'] = v_0
                state_dict['a_start'] = a_0
                state_dict['v_end'] = self._set_vector((self.points[i+2]-self.points[i+1]), v_design/2)
                state_dict['a_end'] = self._set_vector((self.points[i+2]-self.points[i+1]), -0.5)

            else:
                state_dict['mid'] = (self.points[i+1] + self.points[i])/2
                _, v_0, a_0 = self._poly(self.poly_coeff[i-1], self.T[i])
                state_dict['v_start'] = v_0
                state_dict['a_start'] = a_0
                state_dict['a_end'] = np.zeros((3,))

            self.poly_coeff[i] = self._solve_sol_2(state_dict)
            
        # print(self.poly_coeff)
        
        # self._plot_traj()

        # STUDENT CODE HERE

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        
        if t <= 0:
            # Handle the situation where t is negative
            x = self.points[0]
            pass
        elif t >= self.T_table[-1]:
            # Handle the situation where t is greater than the total time
            x = self.points[-1]
            pass
        else:
            # Determine which segment the time t is in
            index = np.sum(t > self.T_table) - 1
            
            # Calculate the time within the segment
            t_seg = t - self.T_table[index]
            # Determine which half of the segment the time t is in
            # if t_seg > self.T[index + 1]/2:
            #     # the second half of the segment
            #     t_seg = t_seg - self.T[index + 1]/2
            #     x, x_dot, x_ddot = self._poly(self.poly_coeff[index, :, 6:], t_seg)
            # else:
            #     # the first half of the segment
            #     x, x_dot, x_ddot = self._poly(self.poly_coeff[index, :, :6], t_seg)
            x, x_dot, x_ddot = self._poly(self.poly_coeff[index, :, :], t_seg)
                
            # Calculate the desired flat output for the segment
            

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    
    def _perpend_dist(self, p1, p2, p_mid):
        """
        Compute the perpendicular distance from each point in the path to the
        line segment between the start and end points. This function will be
        used to determine which points to keep as waypoints.
        
        Inputs
            p1, start point of the line segment, shape=(3,)
            p2, end point of the line segment, shape=(3,)
            p_mid, point to compute the distance from, shape=(3,)
        Outputs
            dist, perpendicular distance from p_mid to the line
        """
        return np.linalg.norm(np.cross(p2-p1, p1-p_mid))/np.linalg.norm(p2-p1)
    
    def _rdp(self, path, epsilon):
        """
        Apply the Ramer-Douglas-Peucker algorithm to reduce the number of points
        in the path. This function will be used to determine which points to keep
        as waypoints.
        
        For more information on the RDP algorithm, see:
        https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        
        Inputs
            path, dense path to reduce, shape=(n_pts,3)
            epsilon, maximum perpendicular distance to the line, m
        Outputs
            reduced_path, reduced path, shape=(n_pts,3)
        """
        d_max = 0
        index = 0
        end = len(path)
        for i in range(1, end-1):
            d = self._perpend_dist(path[0], path[-1], path[i])
            if d > d_max:
                index = i
                d_max = d
        if d_max > epsilon:
            rec1 = self._rdp(path[:index+1], epsilon)
            rec2 = self._rdp(path[index:], epsilon)
            return np.vstack((rec1[:-1], rec2))
        else:
            return np.vstack((path[0], path[-1]))
        
    
    def _poly(self, c, t):
        """
        Compute the polynomial value at time t using coefficients c.
        
        Inputs
            c, polynomial coefficients, shape=(3,6)
            t, time, s
        Outputs
            (x, x_dot, x_ddot), position, velocity, and acceleration at time t
        """        
        # Compute the polynomial value at time t
        A = np.array([[t**5, t**4, t**3, t**2, t, 1],
                      [5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0],
                      [20*t**3, 12*t**2, 6*t, 2, 0, 0]])
        x = A @ c.T
        
        # print(x[0].shape)
        return x[0], x[1], x[2]

    def _solve_sol_0(self, state_dict):
        """_
        Solve for the polynomial coefficients of a segment given the start and
        end conditions and the time duration by Solution #0
        
        Inputs
            start, start conditions, shape=(3,)
            end, end conditions, shape=(3,)
            T, time duration, s
        Outputs
            c, polynomial coefficients, shape=(n,)
        """
        v_start = state_dict['v_start']
        v_end = state_dict['v_end']
        start = state_dict['start']
        end = state_dict['end']
        T = state_dict['T']
        a_start = state_dict['a_start']
        a_end = state_dict['a_end']
            
        c = np.zeros((3,6))
        A = np.array(     [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                           [T**5.0, T**4.0, T**3.0, T**2.0, T, 1.0],
                           [5.0*T**4.0, 4.0*T**3.0, 3.0*T**2.0, 2.0*T, 1.0, 0.0],
                           [20.0*T**3.0, 12.0*T**2.0, 6.0*T, 2.0, 0.0, 0.0]])
        for i in range(3):
            b = np.array([start[i], v_start[i]*1.0, a_start[i], end[i], v_end[i]*1.0, a_end[i]])
            c[i] = np.linalg.solve(A,b)
        return c
    
    def _solve_sol_1(self, state_dict):
        """
        Solve for the polynomial coefficients of a segment given the start and
        end conditions and the time duration by Solution #1
        Inputs
            state_dict, dictionary of state variables
                start, start conditions, shape=(3,)
                end, end conditions, shape=(3,)
                mid, mid conditions, shape=(3,)
                T, time duration, s
                v_start, start velocity, shape=(3,)
                v_end, end velocity, shape=(3,)
                a_start, start acceleration, shape=(3,)
                a_end, end acceleration, shape=(3,)
        Outputs
            c, polynomial coefficients, shape=(3,12)
        """
        start = state_dict['start']
        end = state_dict['end']
        mid = state_dict['mid']
        t1 = state_dict['T']/2
        t2 = state_dict['T']/2
        v_start = state_dict['v_start']
        v_end = state_dict['v_end']
        a_start = state_dict['a_start']
        a_end = state_dict['a_end']
        c = np.zeros((3,12))
        # t = T/2
        
        # Boundary and continuity constraints (Sol 1)
        A = np.array(     [[      0,       0,      0,    0,  0,  1,       0,       0,      0,    0,  0,  0],
                           [      0,       0,      0,    0,  1,  0,       0,       0,      0,    0,  0,  0],
                           [      0,       0,      0,    2,  0,  0,       0,       0,      0,    0,  0,  0],
                           [      0,       0,      0,    0,  0,  0,    t2**5,    t2**4,   t2**3, t2**2,  t2,  1],
                           [      0,       0,      0,    0,  0,  0,  5*t2**4,  4*t2**3, 3*t2**2,  2*t2,  1,  0],
                           [      0,       0,      0,    0,  0,  0, 20*t2**3, 12*t2**2,    6*t2,    2,  0,  0],
                           [   t1**5,    t1**4,   t1**3, t1**2,  t1,  1,       0,       0,      0,    0,  0,  0],
                           [      0,       0,      0,    0,  0,  0,       0,       0,      0,    0,  0,  1],
                           [ 5*t1**4,  4*t1**3, 3*t1**2,  2*t1,  1,  0,       0,       0,      0,    0, -1,  0],
                           [20*t1**3, 12*t1**2,    6*t1,    2,  0,  0,       0,       0,      0,   -2,  0,  0],
                           [60*t1**2,    24*t1,      6,    0,  0,  0,       0,       0,     -6,    0,  0,  0],
                           [  120*t1,      24,      0,    0,  0,  0,       0,     -24,      0,    0,  0,  0]])
        
        for i in range(3):
            b = np.array([start[i], v_start[i], a_start[i], end[i], v_end[i], a_end[i], mid[i], mid[i], 0, 0, 0, 0])
            c[i] = np.linalg.solve(A,b)
        return c   
    
    def _solve_sol_2(self, state_dict):
        """
        Solve for the polynomial coefficients of a segment given the start and
        end conditions and the time duration. 
        
        Inputs
            state_dict, dictionary of state variables
                start, start conditions, shape=(3,)
                end, end conditions, shape=(3,)
                mid, mid conditions, shape=(3,)
                T, time duration, s
                v_start, start velocity, shape=(3,)
                v_end, end velocity, shape=(3,)
                a_start, start acceleration, shape=(3,)
                a_end, end acceleration, shape=(3,)
        Outputs
            c, polynomial coefficients, shape=(n,)
        """
        
        c = np.zeros((3,6))
        start = state_dict['start']
        end = state_dict['end']
        T = state_dict['T']
        v_start = state_dict['v_start']
        v_end = state_dict['v_end']
        a_start = state_dict['a_start']
        a_end = state_dict['a_end']
        #print(state_dict)
        
        # print(np.linalg.norm(v_start))
        # print(np.linalg.norm(v_end))
        
        
        # Boundary and continuity constraints (Sol 1)
        # A = cvxopt.matrix([[      0,       0,      0,    0,  0,  1,       0,       0,      0,    0,  0,  0],
                        #    [      0,       0,      0,    0,  1,  0,       0,       0,      0,    0,  0,  0],
                        #    [      0,       0,      0,    2,  0,  0,       0,       0,      0,    0,  0,  0],
                        #    [      0,       0,      0,    0,  0,  0,    t**5,    t**4,   t**3, t**2,  t,  1],
                        #    [      0,       0,      0,    0,  0,  0,  5*t**4,  4*t**3, 3*t**2,  2*t,  1,  0],
                        #    [      0,       0,      0,    0,  0,  0, 20*t**3, 12*t**2,    6*t,    2,  0,  0],
                        #    [   t**5,    t**4,   t**3, t**2,  t,  1,       0,       0,      0,    0,  0,  0],
                        #    [      0,       0,      0,    0,  0,  0,       0,       0,      0,    0,  0,  1],
                        #    [ 5*t**4,  4*t**3, 3*t**2,  2*t,  1,  0,       0,       0,      0,    0, -1,  0],
                        #    [20*t**3, 12*t**2,    6*t,    2,  0,  0,       0,       0,      0,   -2,  0,  0],
                        #    [60*t**2,    24*t,      6,    0,  0,  0,       0,       0,     -6,    0,  0,  0],
                        #    [  120*t,      24,      0,    0,  0,  0,       0,     -24,      0,    0,  0,  0]])
        
        # Boundary and continuity constraints (Sol 2)
        A = cvxopt.matrix([[0.0, 0.0, 0.0, T**5.0, 5.0*T**4.0, 20.0*T**3.0],
                   [0.0, 0.0, 0.0, T**4.0, 4.0*T**3.0, 12.0*T**2.0],
                   [0.0, 0.0, 0.0, T**3.0, 3.0*T**2.0, 6.0*T],
                   [0.0, 0.0, 2.0, T**2.0, 2.0*T, 2.0],
                   [0.0, 1.0, 0.0, T, 1.0, 0.0],
                   [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        # print(A)
        
        for i in range(3):
            # Boundary and continuity constraints (Sol 1)
            # b = cvxopt.matrix([start[i], 0, 0, end[i], 0, 0, mid[i], mid[i], 0, 0, 0, 0])
            
            # Boundary and continuity constraints (Sol 2)
            b = cvxopt.matrix([start[i], v_start[i], a_start[i], end[i], v_end[i], a_end[i]])
            
            # Cose Function
            H = 0.5 * cvxopt.matrix([[720.0*T**5.0, 360.0*T**4.0, 120.0*T**3.0, 0.0, 0.0, 0.0],
                               [360.0*T**4.0, 192.0*T**3.0,  72.0*T**2.0, 0.0, 0.0, 0.0],
                               [120.0*T**3.0,  72.0*T**2.0,       36.0*T, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            
            # print(H)
            f = cvxopt.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # Solve the quadratic programming problem
            sol = cvxopt.solvers.qp(P = H, q = f, A=A, b=b)

            # Extract the solution
            x = np.array(sol['x'])
            c[i] = x.flatten()
            # print(x)
            
        # print(self._poly(c, T))
        # exit(0)
        return c
    
    def _segment_time(self, T_total):
        pass
    
    
    
    
    
    
    
    def _plot_waypoints(self):
        """
        Plot the waypoints on the world plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(self.points[:, 0], self.points[:, 1], self.points[:, 2], 'ro-', label='Waypoints')
        ax.plot(self.path[:, 0], self.path[:, 1], self.path[:, 2], 'bo-', alpha=0.3, label='Path')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
        
    def _plot_traj(self):
        """
        Plot the trajectory on the world plot.
        """
        # print("Plotting trajectory...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        t = np.linspace(0, self.T_table[-1], 100)
        x = np.zeros((len(t),3))
        x_dot = np.zeros((len(t),3))
        x_ddot = np.zeros((len(t),3))
        for i in range(len(t)):
            flat_output = self.update(t[i])
            x[i] = flat_output['x']
            # print(x[i])
        ax.plot(x[:, 0], x[:, 1], x[:, 2], 'ro-', label='Waypoints')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    def _set_vector(self, vec, v):
        return vec * v / np.linalg.norm(vec)

    def _velocity_generation(self, p, v_base = 2.0):
        """
        Generate a velocity vector from 4 points.
        """
        v1 = p[1] - p[0]
        v2 = p[2] - p[1]
        v3 = p[3] - p[2]
        
        v1_std = v1/np.linalg.norm(v1)
        v2_std = v2/np.linalg.norm(v2)
        v3_std = v3/np.linalg.norm(v3)
        
        # negation to curvature, has larger curvature, smaller velocity
        curv_1 = np.dot(v1_std, v2_std)
        curv_2 = np.dot(v2_std, v3_std)
        
        vel_1 = v_base * np.exp((curv_1-.8)*1.4)
        vel_2 = v_base * np.exp((curv_2-.8)*1.4)
        
        return vel_1, vel_2




# def find_circle_3d(p1, p2, p3):
#     """
#     Finds the center and radius of the unique circle passing through three 3D points.
    
#     :param p1, p2, p3: NumPy arrays of shape (3,), representing three 3D points.
#     :return: center (numpy array of shape (3,)), radius (float), normal (numpy array of shape (3,))
#     """
#     # Compute midpoints of two segments
#     mid_AB = (p1 + p2) / 2
#     mid_BC = (p2 + p3) / 2

#     # Compute normal to the plane containing the three points
#     normal = np.cross(p2 - p1, p3 - p1)
#     normal /= np.linalg.norm(normal)  # Normalize the normal vector

#     # Compute perpendicular bisector direction vectors
#     v1 = p2 - p1
#     v2 = p3 - p1
#     bisector_dir1 = np.cross(normal, v1)  # Perpendicular to both normal and v1
#     bisector_dir2 = np.cross(normal, v2)  # Perpendicular to both normal and v2

#     A = np.vstack([bisector_dir1, bisector_dir2]).T  # Matrix system
#     b = np.vstack([mid_AB, mid_BC]).mean(axis=0)  # Average of two midpoints
#     b = np.vstack([mid_AB, mid_BC]).mean(axis=0)  # Average of two midpoints

#     center = np.linalg.lstsq(A, b, rcond=None)[0]  # Least squares solution

#     # Compute radius
#     radius = np.linalg.norm(center - p1)

#     return center, radius, normal

# def arc_midpoint_3d(p1, p2, center, radius):
#     """
#     Finds the midpoint of the arc between two points on a circle in 3D.
    
#     :param p1, p2: NumPy arrays of shape (3,), representing two points on the circle.
#     :param center: NumPy array of shape (3,), representing the center of the circle.
#     :param radius: The radius of the circle.
#     :return: NumPy array of shape (3,), representing the arc midpoint.
#     """
#     A = p1 - center
#     B = p2 - center

#     # Normalize vectors
#     A /= np.linalg.norm(A)
#     B /= np.linalg.norm(B)

#     # Compute the midpoint direction
#     mid_vec = A + B
#     mid_vec /= np.linalg.norm(mid_vec)  # Normalize

#     # Scale and shift back
#     mid_arc = center + radius * mid_vec

#     return mid_arc



if __name__ == "__main__":
    pass        