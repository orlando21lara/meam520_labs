# Python modules
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

class BlockStacker:
    def __init__(self, team):
        self.fk = FK()
        self.ik = IK()
        self.team = team

        self.detector = ObjectDetector()

        self.arm = ArmController()

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

        self.num_readings = 5
        self.static_blocks = []
        self.dynamic_blocks = []
    
    def openGripper(self):
        return self.arm.exec_gripper_cmd(0.08)

    def closeGripper(self, force=None):
        return self.arm.exec_gripper_cmd(0.03, force=force)

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
        dynamic_block_height = 0
        q_turntableview = np.array([pi/2,pi/8,0,-3*pi/8,0,pi/2,pi/4])
        self.arm.safe_move_to_position(q_turntableview)
        self.H_world_ee = self.fk.forward(q_turntableview)[1]
        # self.detector. get_detections()
        for (name, H_camera_block) in self.detector.get_detections():
            H_world_block = self.H_world_ee @ self.H_ee_camera @ H_camera_block
            show_pose(H_world_block, name, "world")
            dynamic_block_height += H_world_block[2,2]
            block_changed = self.changeAxis(H_world_block)
            self.dynamic_blocks.append((name, block_changed))
            # show_pose(block_changed, name, "dynamic")
        dynamic_block_height = dynamic_block_height/len(self.dynamic_blocks)
        for (name, H_block) in self.dynamic_blocks:
            x = H_block[0,2]
            y = H_block[1,2]
            break
        H_ee_dynamicgrip = np.array([[0,0,-1,x],[0,1,0,y],[1,0,0,dynamic_block_height],[0,0,0,1]])
        q_liedown, _, success, _ =self.ik.inverse(H_ee_dynamicgrip,self.q_static_table_view,"J_pseudo",0.5)
        print(success)
        print(q_liedown)
        # self.arm.safe_move_to_position(q_liedown)
        
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
            block_changed = self.changeAxis(H_world_block)
            # show_pose(H_world_block, name, 'world')
            # show_pose(block_changed, name+"_c", 'world')
            self.static_blocks.append((name, block_changed))

        if self.team == 'red':
            self.static_blocks.sort(key=lambda x: x[1][1, 3], reverse=True) 
        else:
            self.static_blocks.sort(key=lambda x: x[1][1, 3], reverse=False)

    def stackStaticBlocks(self):
        H_current_stack = deepcopy(self.H_world_stack_base)
        self.openGripper()
        
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
                    self.closeGripper()

                    """ Now that the block is grasped we will move on to stacking it"""
                    # Move to position above the stack
                    H_current_stack[2, 3] += 0.05
                    q_stack_above, _, _, _ = self.ik.inverse(H_current_stack, self.q_stack_base, "J_pseudo", 0.5)
                    self.arm.safe_move_to_position(q_stack_above)

                    # Move down to drop the block
                    H_current_stack[2, 3] -= 0.04
                    q_stack, _, _, _ = self.ik.inverse(H_current_stack, self.q_stack_base, "J_pseudo", 0.5)
                    self.arm.safe_move_to_position(q_stack)
                    self.openGripper()

                    # Move back above the stack
                    self.arm.safe_move_to_position(q_stack_above)
                    
                    # Update the stack position
                    H_current_stack[2, 3] += 0.04
                else:
                    print("Failed to move to grasping position")
            else:
                print("Failed to move above block")

    def testGrasping(self):
        for (name, H_block) in self.static_blocks:
            # Move above the block
            H_target = deepcopy(H_block)
            H_target[2, 3] += 0.1

            q_solution, _, success, _ = self.ik.inverse(H_target, self.q_seed, "J_pseudo", 0.5)
            if success:
                print("Attempting to move above block", name)
                # Attempting to move above the block
                self.moveToPosition(q_goal=q_solution)
                self.openGripper()

                H_target = H_block      # New target is the pose of the block

                # Solving for the grasping position
                q_solution, _, success, _ = self.ik.inverse(H_target, self.q_seed, "J_pseudo", 0.5)

                if success:
                    # Attempting to move to grasping position
                    self.moveToPosition(q_goal=q_solution)
                    while True:
                        force = float(input("Enter force to apply: "))
                        success = self.closeGripper(force)
                        print("Gripper closed successfully:", success)

                        input("Press enter to open gripper")
                        success = self.openGripper()

                else:
                    print("Failed to find inverse when going down")
            else:
                print("Failed to find inverse going above")
            break

    def testGripper(self):
        input('Press enter to open')
        self.openGripper()
        input("Prese enter to close")
        self.closeGripper()


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
    block_stacker.detectStaticBlocks()
    block_stacker.stackStaticBlocks()


if __name__ == "__main__":
    main()


