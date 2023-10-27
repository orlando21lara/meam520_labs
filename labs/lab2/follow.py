import sys
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity import IK_velocity
from lib.calculateFK import FK

class JacobianDemo():
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function
    """
    active = False # When to stop commanding arm
    start_time = 0 # start time
    dt = 0.03 # constant for how to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher('/vis/trace', geometry_msgs.msg.PointStamped, queue_size=10)
    counter = 0
    x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
    last_iteration_time = None
    time_stamps = []
    des_pos_over_time = []
    des_vel_over_time = []
    real_pos_over_time = []
    commanded_vel_over_time = []

    ##################
    ## TRAJECTORIES ##
    ##################

    def eight(t,fx=1,fy=2,rx=.15,ry=.1):
        """
        Calculate the position and velocity of the figure 8 trajector

        Inputs:
        t - time in sec since start
        fx - frequecny in rad/s of the x portion
        fy - frequency in rad/s of the y portion
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        # Lissajous Curve
        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
        xdes = x0 + np.array([rx*sin(fx*t),ry*sin(fy*t),0])
        vdes = np.array([rx*fx*cos(fx*t),ry*fy*cos(fy*t),0])
        return xdes, vdes

    def ellipse(t,f=1,ry=.15,rz=.15):
        """
        Calculate the position and velocity of the figure ellipse trajector

        Inputs:
        t - time in sec since start
        f - frequecny in rad/s of the trajectory
        rx - radius in m of the x portion
        ry - radius in m of the y portion

        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """

        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position

        ## STUDENT CODE GOES HERE

        # TODO: replace these!
        xdes = JacobianDemo.x0
        vdes = np.array([0,0,0])

        ## END STUDENT CODE

        return xdes, vdes

    def line(t, f=1/5, L=0.2):
        """
        Calculate the position and velocity of the line trajector

        Inputs:
        t - time in sec since start
        f - frequecny in Hz of the line trajectory
        L - length of the line in meters
        
        Outputs:
        xdes = 0x3 np array of target end effector position in the world frame
        vdes = 0x3 np array of target end effector linear velocity in the world frame
        """
        # Create a triangle wave that spans the range [0, 1] and period 1/f
        sign = (-1) ** np.floor(2 * t * f)
        alpha = 2 * np.abs(t * f - np.floor(t * f + 0.5))

        x0 = np.array([0.307, 0, 0.487]) # corresponds to neutral position
        xdes = x0 + np.array([L * alpha, 0, 0])
        vdes = np.array([L * f* sign, 0, 0])

        return xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'endeffector'
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)

    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:
            try:
                t = time_in_seconds() - self.start_time

                # get desired trajectory position and velocity
                xdes, vdes = trajectory(t)

                # get current end effector position
                q = state['position']
                joints, T0e = self.fk.forward(q)
                x = (T0e[0:3,3])


                # First Order Integrator, Proportional Control with Feed Forward
                kp = 20
                v = vdes + kp * (xdes - x)

                # Store trajectory data
                self.time_stamps.append(t)
                self.real_pos_over_time.append(x)
                self.des_pos_over_time.append(xdes)
                self.des_vel_over_time.append(vdes)
                self.commanded_vel_over_time.append(v)

                # Velocity Inverse Kinematics
                dq = IK_velocity(q,v,np.array([np.nan,np.nan,np.nan]))

                # Get the correct timing to update with the robot
                if self.last_iteration_time == None:
                    self.last_iteration_time = time_in_seconds()
                
                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()
                
                new_q = q + self.dt * dq
                
                arm.safe_set_joint_positions_velocities(new_q, dq)
                
                # Downsample visualization to reduce rendering overhead
                self.counter = self.counter + 1
                if self.counter == 10:
                    self.show_ee_position()
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython follow.py line\n\tpython follow.py ellipse\n\tpython follow.py eight")
        exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    if sys.argv[1] == 'line':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.line)
    elif sys.argv[1] == 'ellipse':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.ellipse)
    elif sys.argv[1] == 'eight':
        callback = lambda state : JD.follow_trajectory(state, JacobianDemo.eight)
    else:
        print("invalid option")
        exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")
    JD.active = False
    
    # Plot the trajectory data
    smallest_len = min(len(JD.time_stamps), len(JD.real_pos_over_time), len(JD.des_pos_over_time), len(JD.des_vel_over_time), len(JD.commanded_vel_over_time))
    time_stamps = np.asarray(JD.time_stamps[:smallest_len])
    real_pos_over_time = np.asarray(JD.real_pos_over_time[:smallest_len])
    des_pos_over_time = np.asarray(JD.des_pos_over_time[:smallest_len])
    des_vel_over_time = np.asarray(JD.des_vel_over_time[:smallest_len])
    commanded_vel_over_time = np.asarray(JD.commanded_vel_over_time[:smallest_len])

    pos_error = des_pos_over_time - real_pos_over_time
    squared_error = pos_error ** 2
    mean_squared_error = np.mean(squared_error, axis=0)
    print("Mean Squared Error: ", mean_squared_error)

    fig, ax = plt.subplots()
    
    ax.plot(time_stamps, real_pos_over_time[:, 0], label='Real X-Position', linestyle='-')
    ax.plot(time_stamps, des_pos_over_time[:, 0], label='Desired X-Position', linestyle='--')
    ax.plot(time_stamps, des_vel_over_time[:, 0], label='Desired X-Velocity', linestyle='--')
    ax.plot(time_stamps, commanded_vel_over_time[:, 0], label='Commanded X-Velocity', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X-Position (m) and X-Velocity (m/s)')
    ax.legend()
    ax.grid(True)


    plt.show()
    
