# Python modules
import sys
import numpy as np
from copy import deepcopy
from math import pi

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# Personal modules
from lib.calculateFK import FK
from lib.IK_position_null import IK
from lib.IK_velocity_null import IK_velocity_null
from lib.calcAngDiff import calcAngDiff

# ROS modules
import rospy
import tf

tf_broad  = tf.TransformBroadcaster()
# Broadcasts a frame using the transform from given frame to world frame
def show_pose(H, child_frame, parent_frame="world"):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(H),
        tf.transformations.quaternion_from_matrix(H),
        rospy.Time.now(),
        child_frame,
        parent_frame
    )

def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis.
    k = rotvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R

class TrapezoidalVelocityProfile():
    def __init__(self, start_position, goal_position, v_max, a_max):
        self.v_max = v_max
        self.a_max = a_max
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance = np.linalg.norm(goal_position - start_position)
        self.direction = (goal_position - start_position) / self.distance

        self.t1 = self.v_max / self.a_max       # Time to reach max velocity
        self.t2 = self.distance / self.v_max    # Time to stop traveling at max velocity and start decelerating

        self.t_total = self.t1 + self.t2
    
    def getTrajectory(self, t):
        if t < self.t1:
            # Accelerating
            des_pos = self.start_position + 0.5 * self.a_max * t**2 * self.direction    # self.a_max = self.v_max / self.t1
            des_vel = self.a_max * t * self.direction
            return des_pos, des_vel
        elif t < self.t2:
            # Traveling at max velocity
            des_pos = self.start_position + self.v_max * (t - 0.5 * self.t1) * self.direction
            des_vel = self.v_max * self.direction
            return des_pos, des_vel
        elif t < self.t_total:
            # Decelerating
            des_pos = self.start_position + self.direction * (-self.v_max * 0.5 / (self.t_total - self.t2) * (self.t2**2 + t**2) \
                                                              + self.v_max / (self.t_total - self.t2) * self.t_total * t \
                                                              - self.v_max * 0.5 * self.t1)
            des_vel = self.direction * (self.v_max / (self.t_total - self.t2) * (self.t_total - t))
            return des_pos, des_vel
        else:
            # Done
            return self.goal_position, np.zeros(3)

class TriangularVelocityProfile():
    def __init__(self, start_position, goal_position, v_max, a_max):
        self.v_max = v_max
        self.a_max = a_max
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance = np.linalg.norm(goal_position - start_position)
        self.direction = (goal_position - start_position) / self.distance

        self.tp = np.sqrt(self.distance / self.a_max)   # Time to reach peak velocity
        self.t_total = 2 * self.tp
    
    def getTrajectory(self, t):
        v_top = self.a_max * self.tp

        if t < self.tp:
            # Accelerating
            des_pos = self.start_position + 0.5 * self.a_max * t**2 * self.direction    # self.a_max = self.v_top / self.tp
            des_vel = self.a_max * t * self.direction
            return des_pos, des_vel
        elif t < self.t_total:
            # Decelerating
            des_pos = self.start_position + self.direction * (v_top * 0.5 / (self.t_total - self.tp) * (self.tp**2 - t**2) \
                                                              + v_top / (self.t_total - self.tp) * self.t_total * (t - self.tp) \
                                                              + v_top * 0.5 * self.tp)
            des_vel = self.direction * (v_top / (self.t_total - self.tp) * (self.t_total - t))
            return des_pos, des_vel
        else:
            # Done
            return self.goal_position, np.zeros(3)

class VelocityController():
    def __init__(self, fk, start_pose, goal_pose):
        self.active = False

        self.fk = fk
        self.start_time = None
        self.last_iteration_time = None

        self.a_max = 1.0    # m/s^2
        self.v_max = 1.0    # m/s

        self.kp = 5.0       # Proportional gain for position
        self.kr = 5.0       # Proportional gain for rotation
        self.k0 = 1.0       # Proportional gain for secondary task

        # Determine which velocity profile to use
        self.start_pose = start_pose
        self.goal_pose = goal_pose
        start_position = start_pose[:3,3]
        goal_position = goal_pose[:3,3]
        self.distance = np.linalg.norm(goal_position - start_position)
        if self.distance < self.v_max**2 / self.a_max:
            self.trajectory = TriangularVelocityProfile(start_position, goal_position, self.v_max, self.a_max)
        else:
            self.trajectory = TrapezoidalVelocityProfile(start_position, goal_position, self.v_max, self.a_max)
        
        self.active = True

        self.time_stamps = []
        self.des_pos_over_time = []
        self.des_vel_over_time = []
        self.real_pos_over_time = []
        self.commanded_vel_over_time = []
        self.dq_over_time = []

    def getSecondaryTask(self, q):
        # This is an elbow up preference task
        lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        q_e = lower + (upper - lower) / 2

        return -self.k0 * (q - q_e)

    def followTrajectory(self, state, arm):
        if self.active:
            try:
                if self.start_time is None:
                    self.start_time = time_in_seconds()
                    self.last_iteration_time = self.start_time
                
                curr_time = time_in_seconds()

                t = curr_time - self.start_time

                # Get desired trajectory
                x_des, v_des = self.trajectory.getTrajectory(t)
                if t > self.trajectory.t_total:
                    # Done
                    self.active = False
                    return

                # Get current end effector position
                q = state['position']

                _, T0e = self.fk.forward(q)

                R_curr = (T0e[:3,:3])
                x_curr = (T0e[0:3,3])

                # Compute target velocities
                v_tar = v_des + self.kp * (x_des - x_curr)
                # w_tar = w_des + self.kr * calcAngDiff(R_des, R_curr).flatten()

                b = self.getSecondaryTask(q)

                # Velocity Inverse Kinematics
                dq = IK_velocity_null(q, v_tar, np.zeros(3), b).flatten()

                self.dt = curr_time - self.last_iteration_time
                self.last_iteration_time = curr_time

                new_q = q + self.dt * dq

                # Store trajectory data
                self.time_stamps.append(t)
                self.real_pos_over_time.append(x_curr)
                self.des_pos_over_time.append(x_des)
                self.des_vel_over_time.append(v_des)
                self.commanded_vel_over_time.append(v_tar)
                self.dq_over_time.append(dq)

                arm.set_joint_positions_velocities(new_q, dq, is_safe=True)

            except rospy.exceptions.ROSException:
                pass

class BlockStacker:
    def __init__(self, team):
        self.fk = FK()
        self.ik = IK()
        self.team = team

        self.detector = ObjectDetector()
        self.vel_controller = None

        callback = lambda state : self.stateCallback(state)
        self.arm = ArmController(on_state_callback=callback)

        self.q_static_table_view = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
        self.H_world_static_table_view = self.fk.forward(self.q_static_table_view)[1]

        self.stack_base_z = .225
        self.H_world_stack_base = deepcopy(self.H_world_static_table_view)
        self.H_world_stack_base[:3, 3] = np.array([0.562, -0.135, self.stack_base_z])
        self.seed = deepcopy(self.H_world_stack_base)
        self.seed[1, 3] = .135

        self.q_stack_base, _, _, _ = self.ik.inverse(self.H_world_stack_base, self.q_static_table_view, "J_pseudo", 0.5)
        self.q_seed, _, _, _ = self.ik.inverse(self.seed, self.q_static_table_view, "J_pseudo", 0.5)

        self.H_ee_camera = self.detector.get_H_ee_camera()
        self.H_world_ee = np.eye(4)
        self.static_blocks = []
        self.dynamic_blocks = []
    
    def stateCallback(self, state):
        if self.vel_controller is not None:
            self.vel_controller.followTrajectory(state, self.arm)

    def changeAxis(self, T):
        largest_mag = 0
        vertical_axis = None
        vertical_axis_idx = None

        # Identify the veritcal axis of the block
        for i in range(3):
            mag = np.dot(np.array([0,0,1]), T[:3,i])
            if np.abs(mag) > largest_mag:
                largest_mag = np.abs(mag)
                vertical_axis = deepcopy(T[:3,i])
                vertical_axis_idx = i
        
        # Create a new target pose such that the vertical axis is the z-axis
        H_target = deepcopy(T)
        if (vertical_axis_idx == 0):
            # If x-axis is vertical, rotate about y-axis by 90 degrees
            H_target[:3,:3] = H_target[:3, :3] @ np.array([[0,  0, 1],
                                                           [0,  1, 0],
                                                           [-1, 0, 0]])
        elif (vertical_axis_idx == 1):
            # If y-axis is vertical, rotate about x-axis by 90 degrees
            H_target[:3,:3] = H_target[:3, :3] @ np.array([[1,  0,  0],
                                                           [0,  0, -1],
                                                           [0,  1,  0]])
        else:
            H_target[:3,2] = vertical_axis

        # If the z-axis of the block is pointing up, flip it
        # Also flip another axis to make sure the block is still a right-handed coordinate system
        mag = np.dot(np.array([0,0,1]), H_target[:3,2])
        if (mag > 0):
            H_target[:3,2] = -H_target[:3,2]
            H_target[:3,1] = -H_target[:3,1]

        return H_target

    def moveToPosition(self, goal_pose):
        start_pose = self.fk.forward(self.arm.get_positions())[1]
        self.vel_controller = VelocityController(self.fk, start_pose, goal_pose)
        while(self.vel_controller.active):
            pass
        
        self.plotTrajectory(self.vel_controller)
        # Reset the velocity controller so that stateCallback does not try to follow the trajectory
        self.vel_controller = None
        
    def plotTrajectory(self, vel_controller):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        fig.suptitle('Trajectory Tracking')

        axs[0].plot(vel_controller.time_stamps, vel_controller.real_pos_over_time, label="Real")
        axs[0].plot(vel_controller.time_stamps, vel_controller.des_pos_over_time, label="Desired")
        axs[0].set_ylabel("Position (m)")
        axs[0].legend()

        axs[1].plot(vel_controller.time_stamps, vel_controller.des_vel_over_time, label="Desired")
        axs[1].plot(vel_controller.time_stamps, vel_controller.commanded_vel_over_time, label="Commanded")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()

        axs[2].plot(vel_controller.time_stamps, vel_controller.dq_over_time)
        axs[2].set_ylabel("Joint Velocity (rad/s)")
        axs[2].set_xlabel("Time (s)")

        plt.show()

    def moveToStaticTableView(self):
        self.arm.safe_move_to_position(self.q_static_table_view)
        self.H_world_ee = self.fk.forward(self.q_static_table_view)[1]
        self.arm.open_gripper()
    
    def detectStaticBlocks(self):
        for (name, H_camera_block) in self.detector.get_detections():
            H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
            block_changed = self.changeAxis(H_world_block)
            self.static_blocks.append((name, block_changed))
        
        self.static_blocks.sort(key=lambda x: x[1][1, 3], reverse=False)

    def stackStaticBlocks(self):
        H_current_stack = deepcopy(self.H_world_stack_base)
        
        for (name, H_block) in self.static_blocks:
            # Move above the block
            H_target = deepcopy(H_block)
            H_target[2, 3] += 0.1

            q_solution, _, success, _ = self.ik.inverse(H_target, self.q_seed, "J_pseudo", 0.5)
            if success:
                print("Attempting to move above block", name)
                # Attempting to move above the block
                self.arm.safe_move_to_position(q_solution)

                H_target = H_block      # New target is the pose of the block

                # Solving for the grasping position
                q_solution, _, success, _ = self.ik.inverse(H_target, self.q_seed, "J_pseudo", 0.5)

                if success:
                    # Attempting to move to grasping position
                    self.arm.safe_move_to_position(q_solution)
                    self.arm.close_gripper()

                    """ Now that the block is grasped we will move on to stacking it"""
                    # Move to position above the stack
                    H_current_stack[2, 3] += 0.05
                    q_stack_above, _, _, _ = self.ik.inverse(H_current_stack, self.q_stack_base, "J_pseudo", 0.5)
                    self.arm.safe_move_to_position(q_stack_above)

                    # Move down to drop the block
                    H_current_stack[2, 3] -= 0.04
                    q_stack, _, _, _ = self.ik.inverse(H_current_stack, self.q_stack_base, "J_pseudo", 0.5)
                    self.arm.safe_move_to_position(q_stack)
                    self.arm.open_gripper()

                    # Move back above the stack
                    self.arm.safe_move_to_position(q_stack_above)
                    
                    # Update the stack position
                    H_current_stack[2, 3] += 0.04
                else:
                    print("Failed to move to grasping position")
            else:
                print("Failed to move above block")


def main():
    np.set_printoptions(precision=4, suppress=True)
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    block_stacker = BlockStacker(team)

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    block_stacker.arm.safe_move_to_position(start_position) # on your mark!

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!


    """
    ###############################################
    Panda Block Stacking Challenge Code Begins Here
    ###############################################
    """

    block_stacker.moveToStaticTableView()
    print("Moved to static table view")
    # Print current position
    H_curr = block_stacker.fk.forward(block_stacker.arm.get_positions())[1]
    print("Current pose:", H_curr)

    H_target = deepcopy(H_curr)
    H_target[2, 3] -= 0.2

    print("Moving to target position:", H_target)
    block_stacker.moveToPosition(H_target)
    print("Done moving to target position")
    new_H_curr = block_stacker.fk.forward(block_stacker.arm.get_positions())[1]
    print("Current pose:", new_H_curr)
    # block_stacker.stackStaticBlocks()


if __name__ == "__main__":
    main()


