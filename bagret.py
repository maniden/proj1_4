import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import bagpy
from bagpy import bagreader
import cv2
import os
dir= "./bags"

# c=0
# for filename in os.listdir(dir):
#     if filename.endswith(".bag"):
#         file_path = os.path.join(dir, filename)
#         bag_name = os.path.splitext(filename)[0]  # Extract bag file name without extension
#         output_dir = os.path.join(dir, bag_name)
#         os.makedirs(output_dir, exist_ok=True)  # Create directory for this bag file
#
#         b = bagreader(file_path)
#         csvfiles = []     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
#         for t in b.topics:
#             data = b.message_by_topic(t)
#             csvfiles.append(data)
#
#
#         odom_csv = None
#         for csvfile in os.listdir(output_dir):  # Look for CSVs in the same output directory
#             print(csvfile)
#             if csvfile.endswith("dom.csv"):
#                 odom_csv = os.path.join(output_dir, csvfile)
#
#                 break  # Stop searching once found
#         state = pd.read_csv(odom_csv)   # The topic "odom" contains all the state information we need
#
#         vicon_time = state['Time'] - b.start_time   # Here we are extracting time and subtracting the start time of the .bag file
#         # print(state)
#         # state.to_csv("state_data.csv", index=False)
#         #
#         # Position
#         x = state['pose.pose.position.x']
#         y = state['pose.pose.position.y']
#         z = state['pose.pose.position.z']
#
#         # Velocity
#         xdot = state['twist.twist.linear.x']
#         ydot = state['twist.twist.linear.y']
#         zdot = state['twist.twist.linear.z']
#
#         # Angular Velocity (w.r.t. body frames x, y, and z)
#         wx = state['twist.twist.angular.x']
#         wy = state['twist.twist.angular.y']
#         wz = state['twist.twist.angular.z']
#
#         # Orientation (measured as a unit quaternion)
#         qx = state['pose.pose.orientation.x']
#         qy = state['pose.pose.orientation.y']
#         qz = state['pose.pose.orientation.z']
#         qw = state['pose.pose.orientation.w']
#
#         # If you want to use Rotation, these lines might be useful
#         q = np.vstack((qx,qy,qz,qw)).T      # Stack the quaternions, shape -> (N,4)
#         rot = Rotation.from_quat(q[0,:])
#         so3cmd_csv = None
#         for csvfile in os.listdir(output_dir):  # Look for CSVs in the same output directory
#             if csvfile.endswith("o3cmd_to_crazyflie-cmd_vel_fast.csv"):
#                 so3cmd_csv = os.path.join(output_dir, csvfile)
#
#                 break  # Stop searching once found
#
#         control = pd.read_csv(so3cmd_csv)  # The topic "so3cmd_to_crazyflie/cmd_vel_fast" has our control inputs
#
#         # Different topics publish at different rates and times... we need to make sure the times is synced up between topics
#         control_time = control['Time'] - b.start_time
#
#         # Coefficients below are used to convert thrust PWM (sent to Crazyflie) into Newtons (what your controller computes)
#         c1 = -0.6709
#         c2 = 0.1932
#         c3 = 13.0652
#         cmd_thrust = (((control['linear.z'] / 60000 - c1) / c2) ** 2 - c3) / 1000 * 9.81
#
#         # Orientation is sent to the Crazyflie as Euler angles (pitch and roll, specifically)
#         roll = control['linear.x']
#         pitch = control['linear.y']
#         yaw = np.zeros(pitch.shape)  # Here we assume 0 yaw.
#         cmd_q = Rotation.from_euler('zyx', np.transpose([yaw, roll, pitch]),
#                                     degrees=True).as_quat()  # Generate quaternions from Euler angles
#         # It's often useful to save the objects associated with a figure and its axes
#         (fig1, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time', figsize=(10, 6))
#
#         ax = axes[0]    # Select the first plot
#
#         # You can plot using multiple lines if you want it to be readable
#         ax.plot(vicon_time, x, 'r.', markersize=2)
#         ax.plot(vicon_time, y, 'g.', markersize=2)
#         ax.plot(vicon_time, z, 'b.', markersize=2)
#         ax.legend(('x', 'y', 'z'), loc='upper right')   # Set a legend
#         ax.set_ylabel('position, m')                    # Set a y label
#         ax.grid('major')                                # Put on a grid
#         ax.set_title('Position')                        # Plot title
#
#         ax = axes[1]    # Select the second plot
#
#         # Or to be more efficient you can plot everything with one line...
#         ax.plot(vicon_time, xdot, 'r.', vicon_time, ydot, 'g.', vicon_time, zdot, 'b.', markersize=2)
#         ax.legend(('x','y','z'), loc='upper right')
#         ax.set_ylabel('velocity, m/s')
#         ax.grid('major')
#         ax.set_title('Velocity')
#         ax.set_xlabel("time, s")
#         # fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')
#
#         # Commands vs. Time
#         (fig2, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Commands vs Time', figsize=(10, 6))
#         ax = axes
#         ax.plot(control_time, cmd_thrust, 'k.-', markersize=5)
#         ax.set_ylabel('thrust, N')
#         ax.set_xlabel('time, s')
#         ax.grid('major')
#         ax.set_title('Commanded Thrust')
#         # fig2.savefig(os.path.join(output_dir, "Cmd_vs_time.png"), bbox_inches='tight')
#         fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')
#         plt.close(fig1)
#
#         # Orientation and Angular Velocity vs. Time
#         (fig3, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time', figsize=(10, 6))
#
#         ax = axes[0]
#         ax.plot(control_time, cmd_q[:,0], 'r', control_time, cmd_q[:,1], 'g',
#                 control_time, cmd_q[:,2], 'b', control_time, cmd_q[:,3], 'k')
#         ax.plot(vicon_time, q[:,0], 'r.',  vicon_time, q[:,1], 'g.',
#                 vicon_time, q[:,2], 'b.',  vicon_time, q[:,3],'k.', markersize=2)
#         ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
#         ax.set_ylabel('quaternion')
#         ax.grid('major')
#         ax.set_title('Orientation')
#
#         ax = axes[1]
#         ax.plot(vicon_time, wx, 'r.', vicon_time, wy, 'g.', vicon_time, wz, 'b.',markersize=2)
#         ax.legend(('x', 'y', 'z'), loc='upper right')
#         ax.set_ylabel('angular velocity, rad/s')
#         ax.set_xlabel('time, s')
#         ax.grid('major')
#         ax.set_title('Body Rates')
#         fig3.savefig(os.path.join(output_dir, "Orientation_vs_Time.png"), bbox_inches='tight')
#         plt.close(fig3)
#         fig4 = plt.figure(figsize=(10, 10))
#         ax = fig4.add_subplot(projection='3d')
#
#         ax.view_init(elev=30, azim=-45)
#         ax.set_xlabel('x position, m')
#         ax.set_ylabel('y position, m')
#         ax.set_zlabel('z position, m')
#         ax.plot3D(x,y,z,'k.-',markersize=5)
#         ax.scatter3D(x[0],y[0],z[0], marker='o', c='r', s=60)
#         ax.scatter3D(x.iloc[-1],y.iloc[-1],z.iloc[-1], marker='o', c='g', s=60)
#         ax.legend(('Trajectory','Start','Goal'))
#         ax.set_title("Crazyflie Trajectory")
#         fig4.savefig(os.path.join(output_dir, "3D_Path.png"), bbox_inches='tight')
#         plt.close(fig4)
#
#         # fig3.savefig(os.path.join(output_dir,'box_1.pdf'),bbox_inches='tight')
#
#         # Concatenate all the data into one array for control and one for states
#         bag_states = np.vstack((vicon_time, x, y, z, xdot, ydot, zdot, wx, wy, wz, qw, qx, qy, qz)).T
#         bag_control = np.vstack((control_time, cmd_thrust, cmd_q.T)).T
#
#         # Save these arrays as .csv files
#         np.savetxt(os.path.join(output_dir, f"{bag_name}_states.csv"), bag_states, delimiter=",", fmt='%3.4f')
#         np.savetxt(os.path.join(output_dir, f"{bag_name}_control.csv"), bag_control, delimiter=",", fmt='%3.4f')
#
#         import cv2  # We can use cv2 to load a .png and matplotlib to display that image!
#
#         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
#         path = cv2.imread(os.path.join(output_dir,"3D_Path.png"))  # Loads the image file
#         axes.imshow(path)  # Displays the image
#         axes.axis('off')  # Turn off the axis
#         plt.savefig(os.path.join(output_dir, "3D_Path.png"), bbox_inches='tight', dpi=300)
#         plt.close(fig)
#
#
#         fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
#         pos = cv2.imread(os.path.join(output_dir,"Position_vs_Time.png"))
#         axes.imshow(pos)
#         axes.axis('off')
#         plt.savefig(os.path.join(output_dir, "Position_vs_Time_1.png"), bbox_inches='tight', dpi=300)
#         plt.close(fig)
#         print(c,"done")
#         c+=1

b = bagreader("./bags/_2025-03-18-15-43-10.bag")
bag_name="_2025-03-18-15-43-10"
output_dir = os.path.join(dir, bag_name)
csvfiles = []     # To avoid mixing up topics, we save each topic as an individual csv file, since some topics might have the same headers!
for t in b.topics:
    data = b.message_by_topic(t)
    csvfiles.append(data)
print(csvfiles)


state = pd.read_csv(csvfiles[1])   # The topic "odom" contains all the state information we need

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

control = pd.read_csv(csvfiles[2])  # The topic "so3cmd_to_crazyflie/cmd_vel_fast" has our control inputs

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
# It's often useful to save the objects associated with a figure and its axes
(fig1, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time', figsize=(10, 6))

ax = axes[0]    # Select the first plot

# You can plot using multiple lines if you want it to be readable
ax.plot(vicon_time, x, 'r.', markersize=2)
ax.plot(vicon_time, y, 'g.', markersize=2)
ax.plot(vicon_time, z, 'b.', markersize=2)
ax.legend(('x', 'y', 'z'), loc='upper right')   # Set a legend
ax.set_ylabel('position, m')                    # Set a y label
ax.grid('major')                                # Put on a grid
ax.set_title('Position')                        # Plot title

ax = axes[1]    # Select the second plot

# Or to be more efficient you can plot everything with one line...
ax.plot(vicon_time, xdot, 'r.', vicon_time, ydot, 'g.', vicon_time, zdot, 'b.', markersize=2)
ax.legend(('x','y','z'), loc='upper right')
ax.set_ylabel('velocity, m/s')
ax.grid('major')
ax.set_title('Velocity')
ax.set_xlabel("time, s")
# fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')

# Commands vs. Time
(fig2, axes) = plt.subplots(nrows=1, ncols=1, sharex=True, num='Commands vs Time', figsize=(10, 6))
ax = axes
ax.plot(control_time, cmd_thrust, 'k.-', markersize=5)
ax.set_ylabel('thrust, N')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Commanded Thrust')
# fig2.savefig(os.path.join(output_dir, "Cmd_vs_time.png"), bbox_inches='tight')
fig1.savefig(os.path.join(output_dir, "Position_vs_Time.png"), bbox_inches='tight')

# Orientation and Angular Velocity vs. Time
(fig3, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time', figsize=(10, 6))

ax = axes[0]
ax.plot(control_time, cmd_q[:,0], 'r', control_time, cmd_q[:,1], 'g',
        control_time, cmd_q[:,2], 'b', control_time, cmd_q[:,3], 'k')
ax.plot(vicon_time, q[:,0], 'r.',  vicon_time, q[:,1], 'g.',
        vicon_time, q[:,2], 'b.',  vicon_time, q[:,3],'k.', markersize=2)
ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
ax.set_ylabel('quaternion')
ax.grid('major')
ax.set_title('Orientation')

ax = axes[1]
ax.plot(vicon_time, wx, 'r.', vicon_time, wy, 'g.', vicon_time, wz, 'b.',markersize=2)
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('angular velocity, rad/s')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Body Rates')
fig3.savefig(os.path.join(output_dir, "Orientation_vs_Time.png"), bbox_inches='tight')

fig4 = plt.figure(figsize=(10, 10))
ax = fig4.add_subplot(projection='3d')

ax.view_init(elev=30, azim=-45)
ax.set_xlabel('x position, m')
ax.set_ylabel('y position, m')
ax.set_zlabel('z position, m')
ax.plot3D(x,y,z,'k.-',markersize=5)
ax.scatter3D(x[0],y[0],z[0], marker='o', c='r', s=60)
ax.scatter3D(x.iloc[-1],y.iloc[-1],z.iloc[-1], marker='o', c='g', s=60)
ax.legend(('Trajectory','Start','Goal'))
ax.set_title("Crazyflie Trajectory")
fig4.savefig(os.path.join(output_dir, "3D_Path.png"), bbox_inches='tight')

# fig3.savefig(os.path.join(output_dir,'box_1.pdf'),bbox_inches='tight')

# Concatenate all the data into one array for control and one for states
bag_states = np.vstack((vicon_time, x, y, z, xdot, ydot, zdot, wx, wy, wz, qw, qx, qy, qz)).T
bag_control = np.vstack((control_time, cmd_thrust, cmd_q.T)).T

# Save these arrays as .csv files
np.savetxt(os.path.join(output_dir, f"{bag_name}_states.csv"), bag_states, delimiter=",", fmt='%3.4f')
np.savetxt(os.path.join(output_dir, f"{bag_name}_control.csv"), bag_control, delimiter=",", fmt='%3.4f')

import cv2  # We can use cv2 to load a .png and matplotlib to display that image!

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
path = cv2.imread(os.path.join(output_dir,"3D_Path.png"))  # Loads the image file
axes.imshow(path)  # Displays the image
axes.axis('off')  # Turn off the axis
plt.savefig(os.path.join(output_dir, "3D_Path.png"), bbox_inches='tight', dpi=300)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
pos = cv2.imread(os.path.join(output_dir,"Position_vs_Time.png"))
axes.imshow(pos)
axes.axis('off')
plt.savefig(os.path.join(output_dir, "Position_vs_Time_1.png"), bbox_inches='tight', dpi=300)