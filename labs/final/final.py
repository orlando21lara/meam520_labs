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

def change_axis(T):
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

    print("Moving to static table view position")
    q_table = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4])
    arm.safe_move_to_position(q_table)
    arm.open_gripper()

    H_ee_camera = detector.get_H_ee_camera()    # Transformation of camera frame wrt end effector frame
    H_world_ee = fk.forward(q_table)[1]         # Transformation of end effector frame wrt world frame

    # Get detected blocks
    cubes_wrt_world = []
    for (name, H_camera_cube) in detector.get_detections():
        H_world_cube = H_world_ee @ H_ee_camera @ H_camera_cube
        show_pose(H_world_cube, name, "world")

        block_changed = change_axis(H_world_cube)
        show_pose(block_changed, name + "_changed")

        cubes_wrt_world.append((name, block_changed))


    # Move to the first block
    H_target = cubes_wrt_world[0][1]    # Transform of first block (with modified orientation) wrt world frame

    q_solution, _, success, message = ik.inverse(H_target, q_table, "J_pseudo", 0.5)
    H_solution = fk.forward(q_solution)[1]

    show_pose(H_target, "target", "world")
    show_pose(H_solution, "solution", "world")

    if success:
        print("Attempting to grasp {block}".format(block=cubes_wrt_world[0][0]))
        arm.safe_move_to_position(q_solution)
        arm.close_gripper()
        arm.safe_move_to_position(q_table)
    else:
        print("Failed to move to block")
        print("Target pose:\n", H_target)
        print("Solution:\n", H_solution)
        print(message)



if __name__ == "__main__":
    main()


