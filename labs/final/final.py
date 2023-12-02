import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# import modules from lib folder
from lib.calculateFK import FK
from lib.IK_position_null import IK

import tf

# Broadcasts a T0e as the transform from given frame to world frame
tf_broad  = tf.TransformBroadcaster()
def show_pose(T0e,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(T0e),
        tf.transformations.quaternion_from_matrix(T0e),
        rospy.Time.now(),
        frame,
        "world"
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

    target_orienntation = T
    
    if (vertical_axis_idx == 0):
        # target_orienntation[:3,1] = switch_axis
        # target_orienntation[:3,2] = vertical_axis
        target_orienntation[:3,:3] = target_orienntation[:3, :3] @ np.array([[0,0,1], [0,1,0], [-1,0,0]])
    elif (vertical_axis_idx == 1):
        # target_orienntation[:3,0] = switch_axis
        # target_orienntation[:3,2] = vertical_axis
        target_orienntation[:3,:3] = target_orienntation[:3, :3] @ np.array([[1,0,0], [0,0,-1], [0,1,0]])
    else:
        target_orienntation[:3,2] = vertical_axis

    mag = np.dot(np.array([0,0,1]), target_orienntation[:3,2])

    if (mag > 0):
        target_orienntation[:3,2] = -target_orienntation[:3,2]
        target_orienntation[:3,1] = -target_orienntation[:3,1]

    

    return target_orienntation

if __name__ == "__main__":
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

    # STUDENT CODE HERE

    fk = FK()
    q = np.array([ 0,    0,     0, -pi/2,     0, pi/2, pi/4 ])

    arm.safe_move_to_position(q)
    arm.open_gripper()

    # get the transform from camera to panda_end_effector
    H_ee_camera = detector.get_H_ee_camera()

    # Pose of end-effector in base frame
    _, T0e = fk.forward(q)

    block0 = None
    for (name, pose) in detector.get_detections():
        block0 = T0e @ H_ee_camera @ pose
        show_pose(block0, name)
        block_changed = change_axis(block0)
        print(name)
        print(pose)
        
        show_pose(block_changed, name + "_changed")
        

    
    print("T0e", T0e)

    # print("target_orienntation", target_orienntation)
    # print("det", np.linalg.det(target_orienntation[:3,:3]))

    




    # new_pose = deepcopy(T0e)
    # new_pose[:3,3] = T_block[:3,3]
    # new_pose[2,3] += 0.2

    # new_pose = np.eye(4)
    # new_pose[:3,3] = T_block[:3,3]
    # new_pose[2,3] += 0.2
    # new_pose[:3, 0] = T_block[:3, 0]
    # new_pose[:3, 1] = -T_block[:3, 2]
    # new_pose[:3, 2] = T_block[:3, 1]

    # T_0 = np.array([[0, 1, 0, 0],
    #                 [-1, 0, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0, 0, 0, 1]])
    
    # T_1 = np.array([[1, 0, 0, 0],
    #                 [0, 0, -1, 0],
    #                 [0, 1, 0, 0],
    #                 [0, 0, 0, 1]])

    # new_pose = T_block @ T_0 @ T_1 

    # show_pose(new_pose, "New Pose")

    # ik = IK()
    # q_f, _, _, _ = ik.inverse(new_pose, q, method='J_pseudo', alpha=.3)
    # arm.safe_move_to_position(q_f)

    # _, T0e = fk.forward(q_f)

    # H_ee_camera = detector.get_H_ee_camera()

    # blocks = []
    # for (name, pose) in detector.get_detections():
    #      print(name,'\n',pose)
    #      blocks.append(pose)

    # T_block_new = np.dot(T0e, np.dot(H_ee_camera, blocks[0]))

    # q_f, _, _, _ = ik.inverse(T_block_new, q_f, method='J_pseudo', alpha = .3)

    # arm.safe_move_to_position(q_f)

    # arm.close_gripper()

    # arm.safe_move_to_position(q)
    # q =
    # END STUDENT CODE
