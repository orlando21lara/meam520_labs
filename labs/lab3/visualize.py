import sys
from math import pi, sin, cos
import numpy as np
from time import perf_counter

import rospy
import roslib
import tf
import geometry_msgs.msg
import visualization_msgs
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController

from lib.IK_position_null import IK
from lib.calcManipulability import calcManipulability

rospy.init_node("visualizer")

# Using your solution code
ik = IK()

# Turn on/off Manipulability Ellipsoid
visulaize_mani_ellipsoid = False

#########################
##  RViz Communication ##
#########################

tf_broad  = tf.TransformBroadcaster()
ellipsoid_pub = rospy.Publisher('/vis/ellip', visualization_msgs.msg.Marker, queue_size=10)

# Broadcasts a frame using the transform from given frame to world frame
def show_pose(H,frame):
    tf_broad.sendTransform(
        tf.transformations.translation_from_matrix(H),
        tf.transformations.quaternion_from_matrix(H),
        rospy.Time.now(),
        frame,
        "world"
    )

def show_manipulability_ellipsoid(M):
    eigenvalues, eigenvectors = np.linalg.eig(M)

    marker = visualization_msgs.msg.Marker()
    marker.header.frame_id = "endeffector"
    marker.header.stamp = rospy.Time.now()
    marker.type = visualization_msgs.msg.Marker.SPHERE
    marker.action = visualization_msgs.msg.Marker.ADD

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    #axes_len = np.sqrt(eigenvalues)

    marker.scale.x = eigenvalues[0]
    marker.scale.y = eigenvalues[1]
    marker.scale.z = eigenvalues[2]

    R = np.vstack((np.hstack((eigenvectors, np.zeros((3,1)))), \
                    np.array([0.0, 0.0, 0.0, 1.0])))
    q = quaternion_from_matrix(R)
    q = q / np.linalg.norm(q)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

    marker.color.a = 0.5
    marker.color.r = 0.0
    marker.color.g = 0.0
    marker.color.b = 1.0

    ellipsoid_pub.publish(marker)

#############################
##  Transformation Helpers ##
#############################

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

#################
##  IK Targets ##
#################

# Note: below we are using some helper functions which make it easier to generate
# valid transformation matrices from a translation vector and Euler angles, or a
# sequence of successive rotations around z, y, and x. You are free to use these
# to generate your own tests, or directly write out transforms you wish to test.

"""targets = [
    transform( np.array([-.2, -.3, .5]), np.array([0,pi,pi])            ),
    transform( np.array([-.2, .3, .5]),  np.array([pi/6,5/6*pi,7/6*pi]) ),
    transform( np.array([.5, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.7, 0, .5]),    np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi])            ),
    transform( np.array([.2, .6, 0.5]),  np.array([0,pi,pi-pi/2])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.2, -.6, 0.5]), np.array([pi/4,pi-pi/2,pi])    ),
    transform( np.array([.5, 0, 0.2]),   np.array([0,pi-pi/2,pi])       ),
    transform( np.array([.4, 0, 0.2]),   np.array([pi/2,pi-pi/2,pi])    ),
    ]
    """
#New targets
targets = [
    transform( np.array([0.2, 0.3, 0.5]),  np.array([0, pi/2, pi]) ), #s
    # transform( np.array([0.2, 0.3, 0.5]),  np.array([pi, 0, pi]) ), #s
    transform( np.array([0.2, 0.3, -0.5]),  np.array([pi, 0, pi]) ), #s
    transform( np.array([0.2, -0.3, 0.3]),  np.array([pi, pi/2, pi]) ), #s

    # transform( np.array([0.3, 0.2, 0.5]),  np.array([0, 0, 0]) ), #s
    # transform( np.array([0.2, 0, 0.5]),  np.array([0, 0, -pi/2]) ), #f
    # transform( np.array([0.2, 0, 0.5]),  np.array([0, -pi/2, 0]) ), #f
    # transform( np.array([0.2, 0, 0.5]),  np.array([-pi/2, 0, 0]) ), #s
    # transform( np.array([0, 0, 0.9]),  np.array([0, -pi/3, -pi/2]) ), #s
    # transform( np.array([0.9, 0, 0]),  np.array([0, -pi/3, pi/2]) ), #i
    # transform( np.array([0, 0.9, 0]),  np.array([0, -pi/3, pi/2]) ), #i
    # transform( np.array([0.3, -0.3, 0.4]), np.array([-2.8973, 2.897, 2.897]) ), #s
    # transform( np.array([0.6, -0.5, 0.4]), np.array([-2.8973, 2.897, 2.897]) ),#s
    # transform( np.array([0.4, 0.3, 0.6]), np.array([-2.8973, -2.897, 2.897]) ),#s
    # transform( np.array([0.3, -0.3, 0.4]), np.array([pi/4, pi/3, pi/2]) ), # sucess
    # transform( np.array([0.3, 0.3, 0.6]),  np.array([-pi/4, -pi/3, -pi/2]) ),# failed
    
    # transform( np.array([0.1, -0.4, 0.7]), np.array([pi/6, pi/2, -pi/4]) ), #s
    # transform( np.array([-0.3, 0.2, 0.5]), np.array([pi/3, -pi/4, pi]) ), #f
    # transform( np.array([-0.2, -0.5, 0.3]), np.array([pi/2, pi/6, -pi/6]) ), #f
    # transform( np.array([0.4, 0.0, 0.8]),  np.array([-pi/3, pi/3, pi/4]) ),#s
    # transform( np.array([0.0, 0.5, 0.3]),  np.array([pi, -pi/2, pi/3]) ),#f
    # transform( np.array([-0.5, 0.1, 0.6]), np.array([-pi/6, -pi/4, pi/2]) ),#f
    # transform( np.array([0.2, -0.6, 0.4]), np.array([pi/3, pi, pi/6]) ), #i
    # transform( np.array([-0.1, 0.4, 0.7]), np.array([-pi/4, pi/2, -pi/3]) ), #f
]


####################
## Test Execution ##
####################

def testTargets(arm, method=''):
    assert method == 'J_pseudo' or method == 'J_trans', "Not a valid method for Numerical IK: 'J_pseudo' or 'J_trans'"

    solve_times = []
    iterations = []
    num_success = 0

    # Iterates through the given targets, using your IK solution
    # Try editing the targets list above to do more testing!
    for i, target in enumerate(targets):
        print("---------------------")
        print("Target " + str(i) + " located at:")
        print(target)
        print("Solving... ")
        show_pose(target,"target")

        seed = arm.neutral_position() # use neutral configuration as seed

        start = perf_counter()
        q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.5)  #try both methods
        stop = perf_counter()
        dt = stop - start
        
        # Store time and iteration data
        solve_times.append(dt)
        iterations.append(len(rollout))
        
        print("Solution found: ", 'True' if success else "False")
        print("\tMethod: ", method)
        print("\tTime (seconds): {time:2.2f}".format(time=dt))
        print("\tIterations: ", len(rollout))
        print("\tJoint Config: ", q)
        if success:
            arm.safe_move_to_position(q)
            num_success += 1

            # Visualize 
            if visulaize_mani_ellipsoid:
                mu, M = calcManipulability(q)
                show_manipulability_ellipsoid(M)
                print('Manipulability Index',mu)

        if i < len(targets) - 1:
            input("Press Enter to move to next target...")
    
    return solve_times, iterations, num_success
            

def computeStatistics(solve_times, iterations, num_success):
    mean_time = np.mean(solve_times)
    median_time = np.median(solve_times)
    max_time = max(solve_times)
    mean_iterations = np.mean(iterations)
    median_iterations = np.median(iterations)
    max_iterations = max(iterations)
    success_rate = num_success / len(targets)

    # Report statistics
    print("Elapsed Time (seconds):")
    print("\tMean: ", mean_time)
    print("\tMedian: ", median_time)
    print("\tMaximum: ", max_time)
    print("Iterations:")
    print("\tMean: ", mean_iterations)
    print("\tMedian: ", median_iterations)
    print("\tMaximum: ", max_iterations)
    print("Success rate: ", success_rate)


def main():
    np.set_printoptions(precision=3, suppress=True)
    arm = ArmController()
    seed = arm.neutral_position()
    arm.safe_move_to_position(seed)

    print("####################################")
    print("TEST FOR PSEUDO METHOD")
    print("####################################")
    pseudo_solve_times, pseudo_iterations, pseudo_num_success = testTargets(arm, "J_pseudo")

    print("####################################")
    print("STATISTICS FOR PSEUDO METHOD")
    print("####################################")
    computeStatistics(pseudo_solve_times, pseudo_iterations, pseudo_num_success)
    

    print("####################################")
    print("TEST FOR TRANSPOSE METHOD")
    print("####################################")
    trans_solve_times, trans_iterations, trans_num_success = testTargets(arm, "J_trans")

    print("####################################")
    print("STATISTICS FOR TRANS METHOD")
    print("####################################")
    computeStatistics(trans_solve_times, trans_iterations, trans_num_success)
    
    # Compute Statistics

if __name__ == "__main__":
    main()
