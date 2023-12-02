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

class BlockStacker:
    def __init__(self, arm, detector, fk, ik):
        self.arm = arm
        self.detector = detector
        self.fk = fk
        self.ik = ik

        self.q_static_table_view = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
        self.q_stack_base = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])

        self.H_world_static_table_view = self.fk.forward(self.q_static_table_view)[1]

        self.H_ee_camera = self.detector.get_H_ee_camera()
        self.H_world_ee = np.eye(4)
        self.static_blocks = []
        self.dynamic_blocks = []
    
    def change_axis(self, T):
        # Identify the veritcal axis of the block
        largest_mag = 0
        vertical_axis = None
        vertical_axis_idx = None
        pointing_up = None
        switch_axis = T[:3,1]
        for i in range(3):
            mag = np.dot(np.array([0,0,1]), T[:3,i])
            if np.abs(mag) > largest_mag:
                largest_mag =np.abs(mag)
                vertical_axis = T[:3,i]
                vertical_axis_idx = i
                if mag > 0:
                    pointing_up = True
                else:
                    pointing_up = False
        
        print("Vertical axis: ", vertical_axis)
        print("Veritcal axis index: ", vertical_axis_idx)
        print("Pointing up: ", pointing_up)

        
        # if pointing_up:
        #     vertical_axis = -vertical_axis
        #     if (vertical_axis_idx == 2) :
        #         T[:3, 0] = - T[:3, 0]
        #     else:
        #         switch_axis = -switch_axis

        H_target = T
        
        if (vertical_axis_idx == 0):
            # target_orienntation[:3,1] = switch_axis
            # target_orienntation[:3,2] = vertical_axis
            H_target[:3,:3] = H_target[:3, :3] @ np.array([[0,0,1], [0,1,0], [-1,0,0]])
        elif (vertical_axis_idx == 1):
            # target_orienntation[:3,0] = switch_axis
            # target_orienntation[:3,2] = vertical_axis
            H_target[:3,:3] = H_target[:3, :3] @ np.array([[1,0,0], [0,0,-1], [0,1,0]])
        else:
            H_target[:3,2] = vertical_axis

        mag = np.dot(np.array([0,0,1]), H_target[:3,2])

        if (mag > 0):
            H_target[:3,2] = -H_target[:3,2]
            H_target[:3,1] = -H_target[:3,1]

        return H_target

    def moveToStaticTableView(self):
        self.arm.safe_move_to_position(self.q_static_table_view)
        self.H_world_ee = self.fk.forward(self.q_static_table_view)[1]
        self.arm.open_gripper()
    
    def detectStaticBlocks(self):
        for (name, H_camera_block) in self.detector.get_detections():
            H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
            show_pose(H_world_block, name, "world")

            block_changed = self.change_axis(H_world_block)

            self.static_blocks.append((name, block_changed))

    def stackStaticBlocks(self):
        x = deepcopy(self.H_world_static_table_view)
        x[:3, 3] = np.array([0.562, -0.259, 0.25])

        show_pose(x, "stacking_base", "world")

        self.q_stack_base, _, success, _ = self.ik.inverse(x, self.q_static_table_view, "J_pseudo", 0.5)
        if not success:
            print("AHHHHHHHHHHHH!!")
            return

        # Move to the first block
        H_target = deepcopy(self.static_blocks[0][1])    # Transform of first block (with modified orientation) wrt world frame
        H_target[2, 3] += 0.1

        q_solution, _, success, message = self.ik.inverse(H_target, self.q_static_table_view, "J_pseudo", 0.5)
        H_solution = self.fk.forward(q_solution)[1]
        if success:
            self.arm.safe_move_to_position(q_solution)

            H_target = self.static_blocks[0][1]    # Transform of first block (with modified orientation) wrt world frame

            q_solution, _, success, message = self.ik.inverse(H_target, q_solution, "J_pseudo", 0.5)
            H_solution = self.fk.forward(q_solution)[1]

            if success:
                print("Attempting to grasp {block}".format(block=self.static_blocks[0][0]))
                self.arm.safe_move_to_position(q_solution)
                self.arm.close_gripper()
                print("Closed Gripper")
                print("YAY :)")
                self.arm.safe_move_to_position(self.q_stack_base)
                self.arm.open_gripper()
                self.arm.safe_move_to_position(self.q_static_table_view)
            else:
                print("Failed to move to grasping position")
        else:
            print("Failed to move above block")

def main():
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

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
    fk = FK()
    ik = IK()

    block_stacker = BlockStacker(arm, detector, fk, ik)

    block_stacker.moveToStaticTableView()
    block_stacker.detectStaticBlocks()
    block_stacker.stackStaticBlocks()




if __name__ == "__main__":
    main()


