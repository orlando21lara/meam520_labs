import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation 
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    r_err = R_curr.T @ R_des
    r_err_skew = 0.5 * (r_err - r_err.T)
    omega = np.array([r_err_skew[2, 1],
                      r_err_skew[0, 2],
                      r_err_skew[1, 0]])
    omega = R_curr @ omega

    return omega

def printOmega(omega):
    print("Omega: ", omega)
    magnitude = np.linalg.norm(omega)
    direction = omega / np.linalg.norm(omega)
    print("Magnitude: ", magnitude)
    print("Direction: ", direction)

    return magnitude, direction

def main():
    # 30 degrees about x
    deg1 = np.pi / 6
    deg2 = np.pi / 4
    r_curr = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(deg1), -np.sin(deg1)],
                       [0.0, np.sin(deg1), np.cos(deg1)]])
    # 45 degrees about x
    r_des = np.array([[1.0, 0.0, 0.0],
                      [0.0, np.cos(deg2), -np.sin(deg2)],
                      [0.0, np.sin(deg2), np.cos(deg2)]])
    omega = calcAngDiff(r_des, r_curr)
    mag, dir = printOmega(omega)

    expect_direction = np.array([1, 0, 0])
    expected_magnitude = np.sin(deg2 - deg1)
    assert(np.all(expect_direction == dir))
    assert(np.isclose(expected_magnitude, mag))

if __name__ == "__main__":
    main()