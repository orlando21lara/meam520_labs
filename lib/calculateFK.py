import numpy as np
from math import pi

class FK():

    def __init__(self):
        # These are the DH parameters for the Franka Emika Panda robot arm
        # They are in the order [a, alpha, d, theta]
        self.dh_params = np.array([
            [0,        -pi/2,   0.333,  0],
            [0,         pi/2,   0,      0],
            [0.0825,    pi/2,   0.316,  0],
            [0.0825,    pi/2,   0,      pi],
            [0,        -pi/2,   0.384,  0],
            [0.088,     pi/2,   0,      -pi],
            [0,         0,      0.051,   -pi/4],
            [0,         0,      0.159,  0]
        ])
        
        # The relative joints positions are relative to the previous link frame
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
            homogeneous transformation between a link's frame and the
            previous link's frame. The eith matrix corresponds to the end
            effector frame
        """
        joint_angles = np.append(q, 0)  # Added a fake joint angle for the end effector

        relative_transforms = np.zeros((8,4,4))

        for idx in range(self.dh_params.shape[0]):
            a = self.dh_params[idx,0]
            alpha = self.dh_params[idx,1]
            d = self.dh_params[idx,2]
            theta = joint_angles[idx] + self.dh_params[idx,3]

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
        relative_transforms = self.getRelativeTransforms(q)

        joint_pos = np.zeros((8,4))
        T_0_curr_joint = np.eye(4)

        for idx in range(self.rel_joint_pos.shape[0]):
            joint_pos[idx] = T_0_curr_joint @ self.rel_joint_pos[idx]
            T_0_curr_joint = T_0_curr_joint @ relative_transforms[idx]

        return joint_pos[:,:3], T_0_curr_joint

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
    
if __name__ == "__main__":

    fk = FK()

    # Matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
