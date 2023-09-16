"""
Date: 08/26/2021

Purpose: This script creates an ArmController and uses it to command the arm's
joint positions and gripper.

Try changing the target position to see what the arm does!

"""

import sys
import rospy
import numpy as np
from math import pi

from core.interfaces import ArmController

def print_joint_values(arm, header_msg = "Current joint values:"):
    """Prints the current joint values of the arm"""
    print(header_msg)
    joint_values = arm.get_positions()
    for idx, joint in enumerate(joint_values):
        print("\tJoint {:.4f}: {:.4f}".format(idx, joint))

def main():
    rospy.init_node('demo')

    arm = ArmController()
    arm.set_arm_speed(0.2)

    print_joint_values(arm, "Initial joint values:")

    arm.close_gripper()
    q = arm.neutral_position()
    arm.safe_move_to_position(q)

    print_joint_values(arm, "Neutral joint values:")

    arm.open_gripper()
    q = np.array([0, -1, 0, -2, 0, 1, 1])
    arm.safe_move_to_position(q)
    arm.close_gripper()

    print_joint_values(arm, "Final joint values:")

if __name__ == "__main__":
    main()
