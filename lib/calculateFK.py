import numpy as np
from math import pi

class FK():
    def __init__(self):
        # The Frankas Emika Panda robot arm has 7 joints and 8 links. Each link has an associated
        # frame rigidly attached to it. If we consider the endeffector to be another link then we
        # have a total of 9 frames and thus need 8 relative transformations.
        self.num_rel_transforms = 8

        # These are the DH parameters in the order [a, alpha, d, theta]
        self.dh_params = np.array([
            [0,        -pi/2,   0.333,  0],
            [0,         pi/2,   0,      0],
            [0.0825,    pi/2,   0.316,  0],
            [0.0825,    pi/2,   0,      pi],
            [0,        -pi/2,   0.384,  0],
            [0.088,     pi/2,   0,      -pi],
            [0,         0,      0.051,  -pi/4],
            [0,         0,      0.159,  0]
        ])
        
        # The relative joints positions are fixed wrt the previous link's frame.
        # e.g joint position i (rel_joint_pos[i-1]) is fixed wrt link i-1's frame
        # Its confusing because are joints are numbered 1 t n but python is 0 indexed
        # Note: The last joint position is the end effector's position not a joint position
        self.rel_joint_pos = np.array([
            [0,         0,  0.141,  1],
            [0,         0,  0,      1],
            [0,         0,  0.195,  1],
            [0,         0,  0,      1],
            [0,         0,  0.125,  1],
            [0,         0, -0.015,  1],
            [0,         0,  0.051,  1],
            [0,         0,  0.159,  1]
        ])

    def getRelativeTransforms(self, q):
        """
        INPUT:
        - q
            1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        - relative_transforms
            8 x 4 x 4 numpy array, where each 4x4 matrix represents the
            homogeneous transformation from the current link's frame to the
            next link's frame. e.g relative_frame[0] is the homogeneous
            transformation from the base frame (link0's frame) to link1's frame.
            Also recall that joint i moves link i and thus frame i.
        """
        joint_angles = np.append(q, 0)  # Added a fake joint angle for the end effector frame

        relative_transforms = np.zeros((8,4,4))

        for idx in range(self.num_rel_transforms):
            a       = self.dh_params[idx, 0]
            alpha   = self.dh_params[idx, 1]
            d       = self.dh_params[idx, 2]
            theta   = self.dh_params[idx,3] + joint_angles[idx]

            relative_transforms[idx] = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
        
        return relative_transforms

    def forward(self, q):
        """
        INPUT:
        - q
            1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        - joint_pos
            8 x 3 matrix, where each row contains the [x,y,z] coordinates in the world frame of the
            respective joint's (or end effector's) center in meters. The base of the robot is
            located at [0,0,0].
        - T0e
            4 x 4 homogeneous transformation matrix, representing the end effector frame expressed
            in the world frame
        """
        relative_transforms = self.getRelativeTransforms(q)     # We have total of 8

        # The joint positions are 1-indexed, but python is 0-indexed so joint_pos[0] corresponds to joint 1
        # The last joint position is the end effector's position not a joint position
        # We add another dimension to the joint positions for homogenous transformation
        joint_pos = np.zeros((8, 4))
        T_0_curr_link = np.eye(4)  # Start off with the link0 (world frame's link) wrt itself

        for idx in range(self.num_rel_transforms):
            joint_pos[idx] = T_0_curr_link @ self.rel_joint_pos[idx]
            T_0_curr_link = T_0_curr_link @ relative_transforms[idx]

        return joint_pos[:,:3], T_0_curr_link

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()

    def unitTestFK(self):
        # First test case
        q = np.array([0, 0, 0, -pi/2, 0, pi/2, pi/4]) # Matches figure in the handout
        expected_end_effector_pos = np.array([0.5545, 0, 0.5215])
        joint_positions, T0e = self.forward(q)
        if(np.allclose(expected_end_effector_pos, T0e[:3,3])):
            print("Test 1 Passed")
        else:
            print("Test 1 Failed")
            print("Expected End Effector Position:\n",expected_end_effector_pos)
            print("Actual End Effector Position:\n",T0e[:3,3])

        # Second test case
        q = np.array([0, pi/2, 0, 0, 0, pi/2, pi/4]) # Matches figure in the handout
        expected_end_effector_pos = np.array([0.788, 0, 0.123])
        joint_positions, T0e = self.forward(q)
        if(np.allclose(expected_end_effector_pos, T0e[:3,3])):
            print("Test 2 Passed")
        else:
            print("Test 2 Failed")
            print("Expected End Effector Position:\n",expected_end_effector_pos)
            print("Actual End Effector Position:\n",T0e[:3,3])
        

if __name__ == "__main__":
    fk = FK()
    fk.unitTestFK()
