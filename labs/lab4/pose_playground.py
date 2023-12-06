import sys, os
import rospy
import tf
import numpy as np
from math import pi, sin, cos
from time import perf_counter

from core.interfaces import ArmController
from lib.rrt import rrt

from lib.loadmap import loadmap
from copy import deepcopy
from lib.IK_position_null import IK
from lib.calculateFK import FK

def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array([
        [ 1, 0, 0, d[0] ],
        [ 0, 1, 0, d[1] ],
        [ 0, 0, 1, d[2] ],
        [ 0, 0, 0, 1    ],
    ])

def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array([
        [ 1,     0,       0,  0 ],
        [ 0, cos(a), -sin(a), 0 ],
        [ 0, sin(a),  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array([
        [ cos(a), 0, -sin(a), 0 ],
        [      0, 1,       0, 0 ],
        [ sin(a), 0,  cos(a), 0 ],
        [ 0,      0,       0, 1 ],
    ])

def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array([
        [ cos(a), -sin(a), 0, 0 ],
        [ sin(a),  cos(a), 0, 0 ],
        [      0,       0, 1, 0 ],
        [      0,       0, 0, 1 ],
    ])

def transform(d,rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])

def main():
    rospy.init_node('pose_playground')

    arm = ArmController()

    # start_joints
    # array([-0.24405413,  0.28047698, -0.4487313 , -2.02395409, -2.63632627,
    #         1.69257376,  1.46677655])
    # end_joints
    # array([ 1.11627952,  1.50089844, -1.04973314, -0.21116718, -1.2930511 ,
    #         1.09784537, -0.13778997])
    print("Moving to neutral position")
    q_neutral = arm.neutral_position()
    arm.safe_move_to_position(q_neutral)

    # input("Press enter to move to start position")
    # q_neutral[5] = pi
    # q_neutral[3] += pi/3
    # arm.safe_move_to_position(q_neutral)
    # q_neutral[1] += pi/5
    # arm.safe_move_to_position(q_neutral)
    # q_neutral[5] = 3*pi/4
    # arm.safe_move_to_position(q_neutral)
    # print("Q neutral:", q_neutral)

    fk = FK()
    neutral_arm_orientation = fk.forward(q_neutral)[1][0:3, 0:3]

    ik = IK()
    # Sample a random orientation
    roll_angle = np.random.uniform(-pi, pi)
    pitch_angle = np.random.uniform(-pi, pi)
    yaw_angle = np.random.uniform(-pi, pi)

    # start_position = np.array([0.6, 0.0, 0.8])
    end_position = np.array([0.45, 0.0, 0.15])

    # Compute the homogenous transform from the start position
    # start_pose = transform(start_position, [roll_angle, pitch_angle, yaw_angle])
    # # Compute the homogenous transform from the end position
    # end_pose = transform(end_position, [roll_angle, pitch_angle, yaw_angle])
    # start_pose = np.eye(4)
    # start_pose[0:3, 0:3] = neutral_arm_orientation
    # start_pose[0:3, 3] = start_position

    end_pose = np.eye(4)
    end_pose[0:3, 0:3] = neutral_arm_orientation
    end_pose[0:3, 3] = end_position

    # Compute the inverse kinematics for the start and end poses
    # start_joints, _, start_success, message = ik.inverse(start_pose, q_neutral, 'J_pseudo', 0.5)
    end_joints, _, end_success, message = ik.inverse(end_pose, q_neutral, 'J_pseudo', 0.5)

    # print("Start pose found:", start_success)
    # print("\t Start Joints:", start_joints)
    print("End pose found:", end_success)
    print("\t End Joints:", end_joints)


    # if start_success:
    #     input("Press enter to move to start position")
    #     arm.safe_move_to_position(start_joints)

    if end_success:
        input("Press enter to move to end position")
        arm.safe_move_to_position(end_joints)


if __name__ == "__main__":
    main()
