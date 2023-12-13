# Python modules
import os
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

class Stack:
    def __init__(self, x_pos, y_pos, q_goal_table_seed, team, name, ik, use_precomputed=False):
        self.stack_base = np.array([[1,  0,  0, x_pos],
                                    [0, -1,  0, y_pos],
                                    [0,  0, -1, 0.225], 
                                    [0,  0,  0,     1]])
        self.q_stack_base_seed = q_goal_table_seed
        self.team = team
        self.name = name
        self.ik = ik

        self.max_size = 6
        self.stack_size = 0

        self.preplacement_q = np.zeros((self.max_size, 7))
        self.placement_q = np.zeros((self.max_size, 7))
        self.steer_clear_q = np.zeros((self.max_size, 7))

        self.block_height = 0.05
        self.z_hover_offset = 0.08
        self.x_hover_offset = -0.10
        if team == 'red':
            # self.y_hover_offset = -0.05
            self.y_hover_offset = 0.0
        else:
            # self.y_hover_offset = 0.05
            self.y_hover_offset = 0.0
        self.z_drop_offset = 0.01
        self.z_stear_clear_offset = 0.05
        self.x_steer_clear_offset = -0.05
        if team == 'red':
            # self.y_steer_clear_offset = -0.05
            self.y_steer_clear_offset = 0.0
        else:
            # self.y_steer_clear_offset = 0.05
            self.y_steer_clear_offset = 0.0

        self.computations_are_valid = True

        if use_precomputed:
            self.loadPositions()
        else:
            print("Computing stack positions")
            self.computeStackPositions()
            if self.computations_are_valid:
                self.savePositions()
            else:
                print("Computations are invalid, not saving positions")

    def savePositions(self):
        print("Saving positions to file")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        np.save(dir_path + "/" + self.name + "_preplacement_q.npy", self.preplacement_q)
        np.save(dir_path + "/" + self.name + "_placement_q.npy", self.placement_q)
        np.save(dir_path + "/" + self.name + "_steer_clear_q.npy", self.steer_clear_q)
    
    def loadPositions(self):
        print("Loading positions from file")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.preplacement_q = np.load(dir_path + '/' + self.name + "_preplacement_q.npy")
        self.placement_q    = np.load(dir_path + '/' + self.name + "_placement_q.npy")
        self.steer_clear_q    = np.load(dir_path + '/' + self.name + "_steer_clear_q.npy")

    def computeStackPositions(self):
        H_curr = deepcopy(self.stack_base)
        q_curr = deepcopy(self.q_stack_base_seed)

        for i in range(self.max_size):
            # Compute configuration for preplacement, hovering above the stack
            H_curr[2, 3] += self.z_hover_offset
            H_curr[1, 3] += self.y_hover_offset
            H_curr[0, 3] += self.x_hover_offset
            q_curr, _, success, _ = self.ik.inverse(H_curr, q_curr, "J_pseudo", 0.5)
            if success:
                self.preplacement_q[i] = q_curr
            else:
                self.computations_are_valid = False
                break
                
            # Compute configuration for placement, dropping the block
            H_curr[2, 3] += self.z_drop_offset - self.z_hover_offset
            H_curr[1, 3] -= self.y_hover_offset
            H_curr[0, 3] -= self.x_hover_offset
            q_curr, _, success, _ = self.ik.inverse(H_curr, q_curr, "J_pseudo", 0.5)
            if success:
                self.placement_q[i] = q_curr
            else:
                self.computations_are_valid = False
                break
            
            # Compute configuration for steer clear, moving away from the stack
            H_curr[0, 3] += self.x_steer_clear_offset
            H_curr[1, 3] += self.y_steer_clear_offset
            H_curr[2, 3] += self.z_stear_clear_offset - self.z_drop_offset
            q_curr, _, success, _ = self.ik.inverse(H_curr, q_curr, "J_pseudo", 0.5)
            if success:
                self.steer_clear_q[i] = q_curr
            else:
                self.computations_are_valid = False
                break
            
            # Reset H_curr for next iteration
            H_curr[0, 3] -= self.x_steer_clear_offset
            H_curr[1, 3] -= self.y_steer_clear_offset
            H_curr[2, 3] += self.block_height - self.z_stear_clear_offset

class BlockStacker:
    def __init__(self, team):
        self.fk = FK()
        self.ik = IK()
        self.detector = ObjectDetector()
        self.arm = ArmController()
        self.team = team

        self.dynamic_blocks_height = 0.24
        """
        self.q_static_table_view: Configuration for viewing the static blocks
        self.q_static_table_seed: Seed configuration for solving IK for static blocks. This is centered and slightly above the table
        q_goal_table_seed: Seed configuration for solving IK for stacks we make on the goal table. This is centered and slightly above the goal table
        """
        if team == 'red':
            self.q_static_table_view = np.array([-0.1535, 0.2268, -0.1563, -1.0605, 0.0365, 1.2849, 0.4898])        # x, y, z = [0.562, -0.159, 0.6]    
            self.q_static_table_seed = np.array([-0.1045, 0.2287, -0.1776, -2.0549, 0.0528, 2.2796, 0.4735])        # x, y, z = [0.562, -0.159, 0.225]
            self.q_goal_table_seed = np.array([0.2518, 0.2253, 0.0248, -2.0552, -0.0074, 2.2805, 1.0661])                # x, y, z = [0.562,  0.159, 0.225]
            self.stack_1 = Stack(0.637, 0.135, deepcopy(self.q_goal_table_seed), team, "6_red_stack_1", self.ik, use_precomputed=True)
            self.stack_2 = Stack(0.487, 0.115, deepcopy(self.q_goal_table_seed), team, "6_red_stack_2", self.ik, use_precomputed=True)

            # Dynamic configuration
            # self.q_avoid_tower = [-0.1754, -0.693, 0.0953, -1.9921, 0.0631, 1.3019, 0.7002]   # x, y, z = [0.3,  0.0, 0.6]
            self.q_avoid_tower = [0.7886, 0.2287, 0.309, -1.0666, -0.072, 1.2855, 1.8551]   # x, y, z = [0.3,  0.5, 0.6]
            self.q_turntable_view = [pi/2,pi/8,0,-pi/2+pi/8,0,pi/2,pi/4]
            self.q_turntable_seed = [pi/4+0.2,pi/2,0,-0.1,pi/2-pi/10,pi/2,-pi/4-0.1]
            self.q_turntable_safepos = deepcopy(self.q_turntable_view)
            self.q_turntable_safepos[0] = pi/4

            _, H_turntable_pregrip = self.fk.forward(self.q_turntable_seed)
            H_turntable_pregrip[2,3] = self.dynamic_blocks_height
            self.q_turntable_pregrip, _, success, message = self.ik.inverse(H_turntable_pregrip, self.q_turntable_seed, "J_pseudo", 0.5)
            if success:
                self.attempt_dynamic_blocks = True
            else:
                self.attempt_dynamic_blocks = False

            self.q_turntable_grip = deepcopy(self.q_turntable_pregrip)
            self.q_turntable_grip[0] += 0.2
            self.q_turntable_grip[5] -= 0.2

            self.q_aftergrip = deepcopy(self.q_turntable_pregrip)
            self.q_aftergrip[4] += pi/6+pi/10
            # self.q_dynamic_drop = np.array([-0.1197, 0.1351, -0.1621, -1.9735, 0.0253, 2.1068, 0.4921])   # x, y, z = [0.562,  -0.159, 0.3]
            self.q_dynamic_drop = np.array([0.2927, 0.1159, 0.1445, -2.2093, -0.0229, 2.324, 1.2373])       # x, y, z = [0.492,  0.229, 0.225]
            self.q_dynamic_view = np.array([0.1844, -0.059, 0.2451, -1.9499, 0.015, 1.8926, 1.2097])        # x, y, z = [0.492,  0.229, 0.4]

        else:
            self.q_static_table_view = np.array([0.2401, 0.2248, 0.0455, -1.0606, -0.0106, 1.2853, 1.0668])         # x, y, z = [0.562,  0.159, 0.6]
            self.q_static_table_seed = np.array([0.2518, 0.2253, 0.0248, -2.0552, -0.0074, 2.2805, 1.0661])         # x, y, z = [0.562,  0.159, 0.225]
            self.q_goal_table_seed = np.array([-0.1045, 0.2287, -0.1776, -2.0549, 0.0528, 2.2796, 0.4735])               # x, y, z = [0.562, -0.159, 0.225]
            self.stack_1 = Stack(0.637, -0.135, deepcopy(self.q_goal_table_seed), team, "6_blue_stack_1", self.ik, use_precomputed=True)
            self.stack_2 = Stack(0.487, -0.135, deepcopy(self.q_goal_table_seed), team, "6_blue_stack_2", self.ik, use_precomputed=True)

            # Dynamic configuration
            # self.q_avoid_tower  = [-0.2401, 0.2249, 0.0455, -1.0605, -0.0106, 1.2852, 1.0669]
            self.q_avoid_tower = [-0.7062, 0.236, -0.4166, -1.0662, 0.0989, 1.284, -0.2991]   # x, y, z = [0.3,  -0.5, 0.6]
            self.q_turntable_view = [-pi/2,pi/8,0,-pi/2+pi/8,0,pi/2,pi/4]
            self.q_turntable_seed = [-2.8,pi/2,pi/2,-pi/2+pi/8,-pi/10,pi/2+pi/8,-pi/4]    # different for red
            self.q_turntable_safepos = deepcopy(self.q_turntable_view)
            self.q_turntable_safepos[0] = -2.3

            _, H_turntable_pregrip = self.fk.forward(self.q_turntable_seed)
            H_turntable_pregrip[2,3] = self.dynamic_blocks_height
            self.q_turntable_pregrip, _, success, message = self.ik.inverse(H_turntable_pregrip, self.q_turntable_seed, "J_pseudo", 0.5)
            if success:
                self.attempt_dynamic_blocks = True
            else:
                self.attempt_dynamic_blocks = False
            
            self.q_turntable_grip = deepcopy(self.q_turntable_pregrip)
            self.q_turntable_grip[0] += 0.2

            self.q_aftergrip = deepcopy(self.q_turntable_pregrip)
            self.q_aftergrip[3] += pi/6
            # self.q_dynamic_drop = np.array([ 0.2024,  0.1338,  0.0762, -1.9735, -0.0118,  2.1069,  1.0694])
            self.q_dynamic_drop = np.array([-0.2049, 0.1179, -0.2333, -2.2093, 0.0373, 2.3237, 0.3232])       # x, y, z = [0.492,  -0.229, 0.225]
            self.q_dynamic_view = np.array([-0.2218, -0.0585, -0.2087, -1.9499, -0.0127, 1.8927, 0.3592])        # x, y, z = [0.492,  -0.229, 0.4]


        self.H_ee_camera = self.detector.get_H_ee_camera()
        self.H_world_ee = np.eye(4)

        self.num_readings = 5       # Number of readings to take for each block for averaging
        self.static_blocks = []
        self.dynamic_blocks = []
    
    def openGripper(self):
        return self.arm.exec_gripper_cmd(0.08)

    def closeGripper(self):
        return self.arm.exec_gripper_cmd(0.03, 120)

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

    def moveToStaticTableView(self):
        self.arm.safe_move_to_position(self.q_static_table_view)
        self.H_world_ee = self.fk.forward(self.q_static_table_view)[1]
    
    def moveToTurnTableView(self):
        self.arm.open_gripper()

        print("Moving to avoid tower")
        self.arm.safe_move_to_position(self.q_avoid_tower)
        
        if self.team == 'blue':
            print("moving to safe position")
            self.arm.safe_move_to_position(self.q_turntable_safepos)

        print("STACK, moving to turntable pregrip")
        self.arm.safe_move_to_position(self.q_turntable_pregrip)

        print("STACK, moving to turntable grip")
        self.arm.safe_move_to_position(self.q_turntable_grip)

    def stackDynamicBlocks(self):
        rospy.sleep(15)
        self.closeGripper()

        if self.team == 'blue':
            print("STACK, moving to aftergrip")
            self.arm.safe_move_to_position(self.q_aftergrip)

        print("STACK, moving to q_avoid_tower")
        self.arm.safe_move_to_position(self.q_avoid_tower)
        
        print("STACK, moving to drop view")
        self.arm.safe_move_to_position(self.q_dynamic_drop)
        self.openGripper()

        # self.moveToStaticTableView()
        print("STACK, moving to dynamic view")
        self.arm.safe_move_to_position(self.q_dynamic_view)

        self.H_world_ee = self.fk.forward(self.q_dynamic_view)[1]

        self.detectSingleBlock()
        self.attemptStackDynamicBlock()
        
    def attemptStackDynamicBlock(self):
        self.openGripper()
        
        if len(self.dynamic_blocks) == 0:
            print("No blocks detected")
            return
        name, H_block = self.dynamic_blocks[0]
        # Move above the block and solve for hovering position
        H_target = deepcopy(H_block)
        H_target[2, 3] += 0.15
        q_solution, _, success, message = self.ik.inverse(H_target, self.q_static_table_seed, "J_pseudo", 0.5)
        if success:
            print("Attempting to move above block ", name)
            self.arm.safe_move_to_position(q_solution)
            
            # New target is the pose of the block. Then solve for grasping position
            H_target = H_block      
            q_solution, _, success, message = self.ik.inverse(H_target, self.q_static_table_seed, "J_pseudo", 0.5)
            if success:
                print("Attempting to move to grasping position for block ", name)
                self.arm.safe_move_to_position(q_solution)
                self.closeGripper()

                # Stack the grasped block
                self.stackBlock()

            else:
                print("Failed to move to grasping position for block", name)
        else:
            print("Failed to move above block ", name)
    
    def detectSingleBlock(self):
        self.dynamic_blocks = []    # Clear the list

        dynamic_blocks_readings = {}
        for i in range(self.num_readings):
            for (name, H_camera_block) in self.detector.get_detections():
                H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
                H_world_block[2, 3] = 0.225
                if self.team == 'red':
                    if H_world_block[1,3] > 0.159:
                        if name not in dynamic_blocks_readings:
                            # Create the empty list
                            dynamic_blocks_readings[name] = []
                        dynamic_blocks_readings[name].append(H_world_block)
                else:
                    if H_world_block[1,3] < -0.159:
                        if name not in dynamic_blocks_readings:
                            # Create the empty list
                            dynamic_blocks_readings[name] = []
                        dynamic_blocks_readings[name].append(H_world_block)
        
        for (name, block_list) in dynamic_blocks_readings.items():
            block_position = np.mean(np.asarray(block_list)[:,:3,3], axis=0)
            H_world_block = np.eye(4)
            H_world_block[:3,3] = block_position
            H_world_block[:3,:3] = block_list[-1][:3,:3]

            block_changed = self.changeAxis(H_world_block)
            show_pose(H_world_block, name, 'world')
            show_pose(block_changed, name+"_c", 'world')
            if np.abs(np.dot(np.array([0,0,1]), block_changed[:3,2])) > 0.86:
                self.dynamic_blocks.append((name, block_changed))

    def detectStaticBlocks(self):
        static_blocks_readings = {}
        for i in range(self.num_readings):
            for (name, H_camera_block) in self.detector.get_detections():
                if name not in static_blocks_readings:
                    # Create the empty list
                    static_blocks_readings[name] = []
                static_blocks_readings[name].append(H_camera_block)

        for (name, block_list) in static_blocks_readings.items():
            block_position = np.mean(np.asarray(block_list)[:,:3,3], axis=0)
            H_camera_block = np.eye(4)
            H_camera_block[:3,3] = block_position
            H_camera_block[:3,:3] = block_list[-1][:3,:3]

            H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
            H_world_block[2,3] = 0.225
            block_changed = self.changeAxis(H_world_block)
            show_pose(H_world_block, name, 'world')
            show_pose(block_changed, name+"_c", 'world')
            if np.abs(np.dot(np.array([0,0,1]), block_changed[:3,2])) > 0.86:
                self.static_blocks.append((name, block_changed))

        if self.team == 'red':
            self.static_blocks.sort(key=lambda x: x[1][1, 3], reverse=True) 
        else:
            self.static_blocks.sort(key=lambda x: x[1][1, 3], reverse=False)

    def stackBlock(self):
        """
        This function assumes that the arm is already grasping a block
        """
        # First determine which stack to use
        if self.stack_1.stack_size == self.stack_1.max_size:
            if self.stack_2.stack_size == self.stack_2.max_size:
                print("Both stacks are full!")
                exit()
            stack = self.stack_2
        else:
            stack = self.stack_1
        stack_block_idx = stack.stack_size

        print("Stacking", stack_block_idx, "on ", stack.name)

        self.arm.safe_move_to_position(stack.preplacement_q[stack_block_idx])   # Move to preplacement position
        self.arm.safe_move_to_position(stack.placement_q[stack_block_idx])      # Move to placement position
        self.openGripper()                                                      # Drop the block
        self.arm.safe_move_to_position(stack.steer_clear_q[stack_block_idx])    # Move to steer clear position
        stack.stack_size += 1                                                   # Increment stack size
        print("Done stacking block idx ", stack_block_idx, "on ", stack.name)

    def stackStaticBlocks(self):
        self.openGripper()
        
        for (name, H_block) in self.static_blocks:
            # Move above the block and solve for hovering position
            H_target = deepcopy(H_block)
            H_target[2, 3] += 0.15
            q_solution, _, success, _ = self.ik.inverse(H_target, self.q_static_table_seed, "J_pseudo", 0.5)
            if success:
                print("Attempting to move above block ", name)
                self.arm.safe_move_to_position(q_solution)
                
                # New target is the pose of the block. Then solve for grasping position
                H_target = H_block      
                q_solution, _, success, _ = self.ik.inverse(H_target, self.q_static_table_seed, "J_pseudo", 0.5)
                if success:
                    print("Attempting to move to grasping position for block ", name)
                    self.arm.safe_move_to_position(q_solution)
                    self.closeGripper()

                    # Stack the grasped block
                    self.stackBlock()

                else:
                    print("Failed to move to grasping position for block", name)
            else:
                print("Failed to move above block ", name)

    """Helper Configuration Methods"""
    def positionFinder(self, target, seed):
        H_target = deepcopy(target)
        q_seed = deepcopy(seed)
        while True:
            x_offset = float(input("Enter x position: "))
            y_offset = float(input("Enter y position: "))
            z_offset = float(input("Enter z position: "))

            H_target[:3,3] = [x_offset, y_offset, z_offset]

            print("Trying to find solution for:\n", H_target)
            print("----------------------------------")

            q_solution, _, success, _ = self.ik.inverse(H_target, q_seed, "J_pseudo", 0.5)
            if success:
                print("SUCCESS!, Moving to pose:\n", self.fk.forward(q_solution)[1])
                print("With configuration:\n", q_solution)
                self.arm.safe_move_to_position(q_solution)
                q_seed = q_solution
            else:
                print("Failed to find inverse")

    def configurationFinder(self, seed):
        q_curr = deepcopy(seed)
        print("Current configuration:\n", q_curr)
        while True:
            q0_offset = float(input("Enter q0 offset: "))
            q1_offset = float(input("Enter q1 offset: "))
            q2_offset = float(input("Enter q2 offset: "))
            q3_offset = float(input("Enter q3 offset: "))
            q4_offset = float(input("Enter q4 offset: "))
            q5_offset = float(input("Enter q5 offset: "))
            q6_offset = float(input("Enter q6 offset: "))
            q_offset = np.array([q0_offset, q1_offset, q2_offset, q3_offset, q4_offset, q5_offset, q6_offset])
            q_curr += q_offset
            print("Current configuration:\n", q_curr)
            self.arm.safe_move_to_position(q_curr)

    def detect_dynamic_blocks_height(self):
        self.dynamic_blocks_height = 0
        self.arm.safe_move_to_position(self.q_turntable_view)
        print("start detection")
        for (name, H_camera_block) in self.detector.get_detections():
            H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
            show_pose(H_world_block, name, "world")
            self.dynamic_blocks_height += H_world_block[2,3]
            self.dynamic_blocks.append((name, H_world_block))
            # block_changed = self.change_axis(H_world_block)
        self.dynamic_blocks_height = (self.dynamic_blocks_height)/len(self.dynamic_blocks)
        print("dynamic height: ", self.dynamic_blocks_height)

    def testPositions(self):
        # Go to static table view
        input("Press ENTER to go to static table view")
        self.arm.safe_move_to_position(self.q_static_table_view)

        # Go to static table seed
        input("Press ENTER to go to static table seed")
        self.arm.safe_move_to_position(self.q_static_table_seed)

        # Go to goal table seed
        input("Press ENTER to go to goal table seed")
        self.arm.safe_move_to_position(self.q_goal_table_seed)

        # Go to stack 1 base
        input("Press ENTER to go to stack 1 base")
        self.arm.safe_move_to_position(self.stack_1.placement_q[0])

        # Go to stack 2 base
        input("Press ENTER to go to stack 2 base")
        self.arm.safe_move_to_position(self.stack_2.placement_q[0])

        print("##########################")
        input("Press ENTER to begin dynamic block detection")

        # self.detect_dynamic_blocks_height()
        self.arm.safe_move_to_position(self.q_turntable_view)

        print("moving to safe position")
        self.openGripper()
        self.arm.safe_move_to_position(self.q_turntable_safepos)

        print("STACK, moving to turntable pregrip")
        self.arm.safe_move_to_position(self.q_turntable_pregrip)

        print("STACK, moving to turntable grip")
        self.arm.safe_move_to_position(self.q_turntable_grip)

        input("Press ENTER to close gripper")
        self.closeGripper()

        print("STACK, moving to aftergrip")
        self.arm.safe_move_to_position(self.q_aftergrip)

        print("STACK, moving to q_avoid_tower")
        self.arm.safe_move_to_position(self.q_avoid_tower)
        
        print("STACK, moving to drop view")
        self.arm.safe_move_to_position(self.q_dynamic_drop)
        self.openGripper()

        print("#####################################")
        print("REMEMBER TO CHANGE THE DYNAMIC HEIGHT: ", self.dynamic_blocks_height)
        print("#####################################")


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

    # Attempt to stack static blocks
    block_stacker.moveToStaticTableView()
    block_stacker.detectStaticBlocks()
    block_stacker.stackStaticBlocks()
    
    # Now for the dynamic blocks
    if block_stacker.attempt_dynamic_blocks:
        while True:
            block_stacker.moveToTurnTableView()
            block_stacker.stackDynamicBlocks()

    """
    THIS IS DEBUGGING CODE BELOW
    """
    # block_stacker.testPositions()
    # block_stacker.positionFinder(block_stacker.H_world_ee, block_stacker.q_static_table_view)
    # block_stacker.configurationFinder(block_stacker.q_turntable_grip)


if __name__ == "__main__":
    main()


