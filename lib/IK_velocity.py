import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """
    dq = np.zeros((1, 7))
    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    J = calcJacobian(q_in)
    Xi = np.concatenate((v_in, omega_in), axis=None)    # Will be a flat array

    processed_jacobian = J[~np.isnan(Xi)]
    processed_Xi = Xi[~np.isnan(Xi)].reshape(-1,1)
    dq = np.linalg.lstsq(processed_jacobian, processed_Xi)
    # dq = np.array(dq[0]).reshape((1,7))
    dq = np.array(dq[0])
    dq = dq.flatten()
    return dq

if __name__ == "__main__":
    result = IK_velocity(np.zeros((1,7)), np.array([1,2,3]), np.array([np.nan, 5, np.nan]))
    print("Result: ", result)
