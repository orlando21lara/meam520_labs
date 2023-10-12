from lib.calculateFK import FK

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull

def getGradientColors(color1, color2, num_pts):
    """
    INPUTS:
    - color1: 3x1 numpy array
    - color2: 3x1 numpy array
    - num_pts: number of points to generate between the two colors

    OUTPUTS:
    - colors: num_pts x 3 numpy array
    """
    colors = np.zeros((num_pts, 3))
    for idx in range(num_pts):
        theta = idx / (num_pts - 1)
        colors[idx, :] = color1 + (color2 - color1) * theta
    return colors

def main():
    fk = FK()

    # the dictionary below contains the data returned by calling arm.joint_limits()
    limits = [
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -1.7628, 'upper': 1.7628},
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -3.0718, 'upper': -0.0698},
        {'lower': -2.8973, 'upper': 2.8973},
        {'lower': -0.0175, 'upper': 3.7525},
        {'lower': -2.8973, 'upper': 2.8973}
    ]

    default_config = [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4]

    num_values_per_joint = 15
    num_joints = 7
    joint_values = np.zeros((num_joints, num_values_per_joint))

    for idx, joint_limit in enumerate(limits):
        joint_values[idx, :] = np.linspace(joint_limit['lower'], joint_limit['upper'], num_values_per_joint)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim3d(-1.5, 1.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_zlim3d(-1, 1.5)


    # """ Random sample workspace """
    # num_samples = 100000
    # joint_pts = np.zeros((num_samples, 3))
    # for idx in range(num_samples):
    #     joint_1 = np.random.choice(joint_values[0, :])
    #     joint_2 = np.random.choice(joint_values[1, :])
    #     joint_3 = np.random.choice(joint_values[2, :])
    #     joint_4 = np.random.choice(joint_values[3, :])
    #     joint_5 = np.random.choice(joint_values[4, :])
    #     joint_6 = np.random.choice(joint_values[5, :])
    #     joint_7 = np.random.choice(joint_values[6, :])
    #     curr_config = np.array([joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7])
    #     _, T0e = fk.forward(curr_config)
    #     joint_pts[idx, :] = T0e[:3, 3]
    #     print("Completion percentage: ", idx/num_samples * 100, "%" )
    # np.save("random_pts_workspace.npy", joint_pts)


    """ Workspace for joint 1 """
    joint_pts = np.zeros((num_values_per_joint**2, 3))
    pt_colors = np.zeros((num_values_per_joint**2, 3))
    colors = getGradientColors(np.array([0, 0, 1]), np.array([1, 0, 0]), num_values_per_joint)
    counter = 0
    for joint_val in joint_values[1, :]:
        for joint_val_2 in joint_values[2, :]:
            curr_config = np.array([default_config[0],
                                    joint_val,
                                    joint_val_2,
                                    default_config[3],
                                    default_config[4],
                                    default_config[5],
                                    default_config[6]])
            _, T0e = fk.forward(curr_config)
            joint_pts[counter, :] = T0e[:3, 3]
            
            counter += 1
            print("Completion percentage: {:.3f}%".format(counter/(num_values_per_joint**2) * 100))
        pt_colors[counter - num_values_per_joint:counter, :] = colors[int(counter / num_values_per_joint) - 1, :]
    ax.scatter(joint_pts[:, 0], joint_pts[:, 1], joint_pts[:, 2], c=pt_colors, marker='o', s=1.0, alpha=1.0)
    plt.show()

if __name__ == "__main__":
    main()