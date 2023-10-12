import numpy as np
from lib.calculateFK import FK

def skew(x):
    """
    input:  x - 3 x 1 numpy array
    output: 3 x 3 skew symmetric matrix corresponding to x
    """
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    arm_fk = FK()
    relative_transforms = arm_fk.getRelativeTransforms(q_in)
    end_effector_origin = arm_fk.forward(q_in)[1][:3, 3]

    num_joints = 7
    deg_of_freedom = 6
    J = np.zeros((deg_of_freedom, num_joints))

    T_0_curr_link = np.eye(4)  # Start off with the link0 (world frame's link) wrt itself (T_0_0)

    for idx in range(num_joints):
        # Get required geometric quantities
        prev_frame_z_axis = T_0_curr_link[:3, 2]
        prev_frame_origin = T_0_curr_link[:3, 3]
        skew_prev_frame_z_axis = skew(prev_frame_z_axis)

        # Populate the linear velocity component of the Jacobian
        J[:3, idx] = skew_prev_frame_z_axis @ (end_effector_origin - prev_frame_origin)
        # Populate the angular velocity component of the Jacobian
        J[3:, idx] = prev_frame_z_axis

        # Update the current transformation matrix
        T_0_curr_link = T_0_curr_link @ relative_transforms[idx]

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
