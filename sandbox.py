import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D, lineStyles
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World


from proj1_3.code.occupancy_map import OccupancyMap
from proj1_3.code.se3_control import SE3Control
from proj1_3.code.world_traj import WorldTraj

# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# Choose a test example file. You should write your own example files too!
filename = '../util/maze_2025_3.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()

# Help debug issues you may encounter with your choice of resolution and margin
# by plotting the occupancy grid after inflation by margin. THIS IS VERY SLOW!!
# fig = plt.figure('world')
# ax = Axes3Ds(fig)
# world.draw(ax)
# fig = plt.figure('occupancy grid')
# ax = Axes3Ds(fig)
# resolution = SET YOUR RESOLUTION HERE
# margin = SET YOUR MARGIN HERE
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# plt.show()

# Set simulation parameters.
t_final = 60
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

print()
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
print(exit.value)

# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.

collision_pts = world.path_collisions(state['x'], robot_radius)

stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= 0.05
no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

print()
print(f"Results:")
print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
print(f"  Flight time:     {flight_time:.1f} seconds")
print(f"  Flight distance: {flight_distance:.1f} meters")
print(f"  Planning time:   {planning_time:.1f} seconds")
if not no_collision:
    print()
    print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# Visualize the original dense path from A*, your sparse waypoints, and the
# smooth trajectory.
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import bagpy
from bagpy import bagreader
import cv2
import inspect
import os
from pathlib import Path
from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World


dir= "../bags"

c=0
for filename in os.listdir(dir):
    if filename.endswith(".bag"):
        file_path = os.path.join(dir, filename)
        bag_name = os.path.splitext(filename)[0]  # Extract bag file name without extension
        output_dir = os.path.join(dir, bag_name)
        os.makedirs(output_dir, exist_ok=True)  # Create directory for this bag file

        b = bagreader(file_path)
        csvfiles = []     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
        for t in b.topics:
            data = b.message_by_topic(t)
            csvfiles.append(data)


        odom_csv = None
        for csvfile in os.listdir(output_dir):  # Look for CSVs in the same output directory
            print(csvfile)
            if csvfile.endswith("dom.csv"):
                odom_csv = os.path.join(output_dir, csvfile)

                break  # Stop searching once found
        state = pd.read_csv(odom_csv)   # The topic "odom" contains all the state information we need

        vicon_time = state['Time'] - b.start_time   # Here we are extracting time and subtracting the start time of the .bag file
        # print(state)
        # state.to_csv("state_data.csv", index=False)
        #
        # Position
        x = state['pose.pose.position.x']
        y = state['pose.pose.position.y']
        z = state['pose.pose.position.z']

        # Velocity
        xdot = state['twist.twist.linear.x']
        ydot = state['twist.twist.linear.y']
        zdot = state['twist.twist.linear.z']

        # Angular Velocity (w.r.t. body frames x, y, and z)
        wx = state['twist.twist.angular.x']
        wy = state['twist.twist.angular.y']
        wz = state['twist.twist.angular.z']

        # Orientation (measured as a unit quaternion)
        qx = state['pose.pose.orientation.x']
        qy = state['pose.pose.orientation.y']
        qz = state['pose.pose.orientation.z']
        qw = state['pose.pose.orientation.w']

        # If you want to use Rotation, these lines might be useful
        q = np.vstack((qx,qy,qz,qw)).T      # Stack the quaternions, shape -> (N,4)
        rot = Rotation.from_quat(q[0,:])
        so3cmd_csv = None
        for csvfile in os.listdir(output_dir):  # Look for CSVs in the same output directory
            if csvfile.endswith("o3cmd_to_crazyflie-cmd_vel_fast.csv"):
                so3cmd_csv = os.path.join(output_dir, csvfile)

                break  # Stop searching once found

        control = pd.read_csv(so3cmd_csv)  # The topic "so3cmd_to_crazyflie/cmd_vel_fast" has our control inputs

        # Different topics publish at different rates and times... we need to make sure the times is synced up between topics
        control_time = control['Time'] - b.start_time

        # Coefficients below are used to convert thrust PWM (sent to Crazyflie) into Newtons (what your controller computes)
        c1 = -0.6709
        c2 = 0.1932
        c3 = 13.0652
        cmd_thrust = (((control['linear.z'] / 60000 - c1) / c2) ** 2 - c3) / 1000 * 9.81

        # Orientation is sent to the Crazyflie as Euler angles (pitch and roll, specifically)
        roll = control['linear.x']
        pitch = control['linear.y']
        yaw = np.zeros(pitch.shape)  # Here we assume 0 yaw.
        cmd_q = Rotation.from_euler('zyx', np.transpose([yaw, roll, pitch]),
                                    degrees=True).as_quat()  # Generate quaternions from Euler angles
        # # It's often useful to save the objects associated with a figure and its axes
        # (fig1, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time', figsize=(10, 6))
        #
        # ax = axes[0]    # Select the first plot

        # You can plot using multiple lines if you want it to be readable
        # ax.plot(vicon_time, x, 'r.', markersize=2)
        # ax.plot(vicon_time, y, 'g.', markersize=2)
        # ax.plot(vicon_time, z, 'b.', markersize=2)
        # ax.legend(('x', 'y', 'z'), loc='upper right')   # Set a legend
        # ax.set_ylabel('position, m')                    # Set a y label
        # ax.grid('major')                                # Put on a grid
        # ax.set_title('Position')                        # Plot title
        #
        # ax = axes[1]    # Select the second plot
        #
        # # Or to be more efficient you can plot everything with one line...
        # ax.plot(vicon_time, xdot, 'r.', vicon_time, ydot, 'g.', vicon_time, zdot, 'b.', markersize=2)
        # ax.legend(('x','y','z'), loc='upper right')
        # ax.set_ylabel('velocity, m/s')
        # ax.grid('major')
        # ax.set_title('Velocity')
        # ax.set_xlabel("time, s")
        # # fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')
        #
        # # Commands vs. Time
        # (fig2, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Commands vs Time', figsize=(10, 6))
        # ax = axes
        # ax.plot(control_time, cmd_thrust, 'k.-', markersize=5)
        # ax.set_ylabel('thrust, N')
        # ax.set_xlabel('time, s')
        # ax.grid('major')
        # ax.set_title('Commanded Thrust')
        # # fig2.savefig(os.path.join(output_dir, "Cmd_vs_time.png"), bbox_inches='tight')
        # fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')
        # plt.close(fig1)
        #
        # # Orientation and Angular Velocity vs. Time
        # (fig3, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time', figsize=(10, 6))
        #
        # ax = axes[0]
        # ax.plot(control_time, cmd_q[:,0], 'r', control_time, cmd_q[:,1], 'g',
        #         control_time, cmd_q[:,2], 'b', control_time, cmd_q[:,3], 'k')
        # ax.plot(vicon_time, q[:,0], 'r.',  vicon_time, q[:,1], 'g.',
        #         vicon_time, q[:,2], 'b.',  vicon_time, q[:,3],'k.', markersize=2)
        # ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
        # ax.set_ylabel('quaternion')
        # ax.grid('major')
        # ax.set_title('Orientation')
        #
        # ax = axes[1]
        # ax.plot(vicon_time, wx, 'r.', vicon_time, wy, 'g.', vicon_time, wz, 'b.',markersize=2)
        # ax.legend(('x', 'y', 'z'), loc='upper right')
        # ax.set_ylabel('angular velocity, rad/s')
        # ax.set_xlabel('time, s')
        # ax.grid('major')
        # ax.set_title('Body Rates')
        # fig3.savefig(os.path.join(output_dir, "Orientation_vs_Time.png"), bbox_inches='tight')
        # plt.close(fig3)
        # fig4 = plt.figure(figsize=(10, 10))
        # ax = fig4.add_subplot(projection='3d')
        #
        # ax.view_init(elev=30, azim=-45)
        # ax.set_xlabel('x position, m')
        # ax.set_ylabel('y position, m')
        # ax.set_zlabel('z position, m')
        # ax.plot3D(x,y,z,'k.-',markersize=5)
        # ax.scatter3D(x[0],y[0],z[0], marker='o', c='r', s=60)
        # ax.scatter3D(x.iloc[-1],y.iloc[-1],z.iloc[-1], marker='o', c='g', s=60)
        # ax.legend(('Trajectory','Start','Goal'))
        # ax.set_title("Crazyflie Trajectory")
        # fig4.savefig(os.path.join(output_dir, "3D_Path.png"), bbox_inches='tight')
        # plt.close(fig4)

        # fig3.savefig(os.path.join(output_dir,'box_1.pdf'),bbox_inches='tight')

        # Concatenate all the data into one array for control and one for states
        bag_states = np.vstack((vicon_time, x, y, z, xdot, ydot, zdot, wx, wy, wz, qw, qx, qy, qz)).T
        bag_control = np.vstack((control_time, cmd_thrust, cmd_q.T)).T


        filename = '../maze_2025_3.json'


        # Load the test example.
        file = Path(inspect.getsourcefile(lambda: 0)).parent.resolve() / filename
        world = World.from_file(file)  # World boundary and obstacles.
        start = world.world['start']  # Start point, shape=(3,)
        goal = world.world['goal']
        my_world_traj = WorldTraj(world, start, goal)



        fig = plt.figure('A* Path, Waypoints, and Trajectory')
        ax = Axes3Ds(fig)
        world.draw(ax)
        ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
        ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
        if hasattr(my_world_traj, 'path'):
            if my_world_traj.path is not None:
                world.draw_line(ax, my_world_traj.path, color='red', linewidth=2)
        else:
            print("Have you set \'self.path\' in WorldTraj.__init__?")
        if hasattr(my_world_traj, 'points'):
            if my_world_traj.points is not None:
                world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
        else:
            print("Have you set \'self.points\' in WorldTraj.__init__?")
        world.draw_line(ax, flat['x'], color='black', linewidth=2)
        ax.legend(handles=[
            Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
            Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
            Line2D([], [], color='black', linewidth=2, label='Trajectory'),
            Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
            loc = 'upper right')

        ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
        ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
        world.draw_line(ax, flat['x'], color='black', linewidth=2)
        world.draw_points(ax, bag_states[:, 1:4], color='blue', markersize=4)
        if collision_pts.size > 0:
            ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
            Line2D([], [], color='black', linewidth=2, label='Trajectory'),

        plt.show()
        planned_pos = np.array(flat['x'])  # Planned trajectory (Nx3)
        actual_pos = np.array(bag_states[:, 1:4])  # Actual flight path from VICON (Nx3)
        expanded_actual_pos = np.zeros_like(planned_pos)

        # Determine how many planned timesteps per actual VICON step
        repeat_factor = len(planned_pos) / len(actual_pos)

        # Assign each planned timestep the nearest previous actual VICON value
        for i in range(len(planned_pos)):
            corresponding_index = int(i / repeat_factor)  # Find nearest previous VICON index
            expanded_actual_pos[i] = actual_pos[min(corresponding_index, len(actual_pos) - 1)]

        # np.set_printoptions(threshold=np.inf)
        #
        # # Ensure both trajectories have the same length


        # Compute tracking error separately for x, y, z
        tracking_error_x = np.abs(planned_pos[:, 0] - expanded_actual_pos[:, 0])
        tracking_error_y = np.abs(planned_pos[:, 1] - expanded_actual_pos[:, 1])
        tracking_error_z = np.abs(planned_pos[:, 2] - expanded_actual_pos[:, 2])

        plt.figure(figsize=(10, 4))
        # plt.plot(expanded_actual_pos[:, 0], label="X Error", alpha=0.7)
        # plt.plot(expanded_actual_pos[:, 1], label="Y Error", alpha=0.7)
        # plt.plot(expanded_actual_pos[:, 2], label="Z Error", alpha=0.7)
        # plt.plot(planned_pos[:, 0], label="X Error", alpha=0.7)
        # plt.plot(planned_pos[:, 1], label="Y Error", alpha=0.7)
        # plt.plot(planned_pos[:, 2], label="Z Error", alpha=0.7)
        plt.plot(tracking_error_x, label="X Error", alpha=0.7)
        plt.plot(tracking_error_y, label="Y Error", alpha=0.7)
        plt.plot(tracking_error_z, label="Z Error", alpha=0.7)

        plt.xlabel("Time Step")
        plt.ylabel("Position Error (m)")
        plt.title("Tracking Error Maze 3")
        plt.legend()
        plt.grid()
        plt.show()

        # #
        # # # Position and Velocity vs. Time
        # # (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
        # # x = state['x']
        # # x_des = flat['x']
        # # ax = axes[0]
        # # ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
        # # ax.plot(sim_time, x[:,0], 'r.',    sim_time, x[:,1], 'g.',    sim_time, x[:,2], 'b.')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('position, m')
        # # ax.grid('major')
        # # ax.set_title('Position')
        # # v = state['v']
        # # v_des = flat['x_dot']
        # # ax = axes[1]
        # # ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
        # # ax.plot(sim_time, v[:,0], 'r.',    sim_time, v[:,1], 'g.',    sim_time, v[:,2], 'b.')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('velocity, m/s')
        # # ax.set_xlabel('time, s')
        # # ax.grid('major')
        # #
        # # (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Velocity and Acceleration vs Time')
        # # x_des_dot = flat['x_dot']
        # # x_des_ddot = flat['x_ddot']
        # # ax = axes[0]
        # # ax.plot(sim_time, x_des_dot[:,0], 'r', sim_time, x_des_dot[:,1], 'g', sim_time, x_des_dot[:,2], 'b')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('velocity, m')
        # # ax.grid('major')
        # # ax.set_title('desired velocity and acceleration')
        # # ax = axes[1]
        # # ax.plot(sim_time, x_des_ddot[:,0], 'r', sim_time, x_des_ddot[:,1], 'g', sim_time, x_des_ddot[:,2], 'b')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('acceleration, m/s')
        # # ax.set_xlabel('time, s')
        # # ax.grid('major')
        # #
        # # # Orientation and Angular Velocity vs. Time
        # # (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
        # # q_des = control['cmd_q']
        # # q = state['q']
        # # ax = axes[0]
        # # ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
        # # ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
        # # ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
        # # ax.set_ylabel('quaternion')
        # # ax.set_xlabel('time, s')
        # # ax.grid('major')
        # # w = state['w']
        # # ax = axes[1]
        # # ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('angular velocity, rad/s')
        # # ax.set_xlabel('time, s')
        # # ax.grid('major')
        # #
        # # # Commands vs. Time
        # # (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
        # # s = control['cmd_motor_speeds']
        # # ax = axes[0]
        # # ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
        # # ax.legend(('1', '2', '3', '4'), loc='upper right')
        # # ax.set_ylabel('motor speeds, rad/s')
        # # ax.grid('major')
        # # ax.set_title('Commands')
        # # M = control['cmd_moment']
        # # ax = axes[1]
        # # ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
        # # ax.legend(('x', 'y', 'z'), loc='upper right')
        # # ax.set_ylabel('moment, N*m')
        # # ax.grid('major')
        # # T = control['cmd_thrust']
        # # ax = axes[2]
        # # ax.plot(sim_time, T, 'k.')
        # # ax.set_ylabel('thrust, N')
        # # ax.set_xlabel('time, s')
        # # ax.grid('major')
        # #
        # # # 3D Paths
        # # fig = plt.figure('3D Path')
        # # ax = Axes3Ds(fig)
        # # world.draw(ax)
        # # ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
        # # ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
        # # world.draw_line(ax, flat['x'], color='black', linewidth=2)
        # # world.draw_points(ax, state['x'], color='blue', markersize=4)
        # # if collision_pts.size > 0:
        # #     ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
        # # ax.legend(handles=[
        # #     Line2D([], [], color='black', linewidth=2, label='Trajectory'),
        # #     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
        # #     loc='upper right')
        # #
        # #
        # # Animation (Slow)
        # #
        # # Instead of viewing the animation live, you may provide a .mp4 filename to save.
        #
        # R = Rotation.from_quat(state['q']).as_matrix()
        # ani = animate(sim_time, state['x'], R, world=world, filename=None)
        #
        # plt.show()
