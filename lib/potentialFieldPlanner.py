import numpy as np
import math
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
import matplotlib.pyplot as plt

class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self, tol=1e-4, max_steps=2000, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        att_f = np.zeros((3, 1)) 
        zeta = 10

        if np.linalg.norm(current-target) > 1:
        # conic well potential
            if np.all(current - target)<1e-3:
                att_f = 0
            else:
                att_f = -(current - target)/np.linalg.norm(current - target)
        else:
            # parabolic well potential
            att_f = -zeta * (current - target)
        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 0x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        obstacle = obstacle
        current = current.reshape((3,1))
        unitvec = unitvec.reshape((3,1))
        
        rep_f = np.zeros((3, 1)) 
        rho0 = 0.1
        eta = 2
        dist,unit = PotentialFieldPlanner.dist_point2box(current.reshape((1,3)), obstacle)
        if dist < rho0:
            rep_f = eta * (1/dist - 1/rho0) * (1/(dist*dist)) * -unitvec
        if np.isnan(np.any(rep_f)):
            print('error')
        ## END STUDENT CODE
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        joint_forces = np.zeros((3, 7)) 
        a = obstacle.shape[0]
        repulsive_force = np.zeros((3,obstacle.shape[0]))    # calculate repulsive force against several obstacles
        repulsive_force_sum = np.zeros((3,7))
        for i in range(7):
            for j in range(obstacle.shape[0]):
                dist, unitvec = PotentialFieldPlanner.dist_point2box(current[:,i].reshape((1,3)),obstacle[j,:])
                unitvec = unitvec.reshape((3,1))
                repulsive_force[:,j] =  PotentialFieldPlanner.repulsive_force(obstacle[j,:],current[:,i],unitvec).flatten()
            # print('rep_F',repulsive_force)
            repulsive_force_sum[:,i] = sum(repulsive_force[:,j]for j in range(repulsive_force.shape[1]))
            
            joint_forces[:,i] = PotentialFieldPlanner.attractive_force(target[:,i],current[:,i]) + repulsive_force_sum[:,i]
            # joint_forces[:,i] = PotentialFieldPlanner.attractive_force(target[:,i],current[:,i]) 
            
        ## END STUDENT CODE
        # print('attr_F',joint_forces)
        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint 
        """

        ## STUDENT CODE STARTS HERE

        joint_torques = np.zeros((1, 7)) 
        q0 = np.hstack((np.array(q[:1]).reshape((1,1)), np.zeros((1,6)))).flatten()
        q1 = np.hstack((np.array(q[:2]).reshape((1,2)), np.zeros((1,5)))).flatten()
        q2 = np.hstack((np.array(q[:3]).reshape((1,3)), np.zeros((1,4)))).flatten()
        q3 = np.hstack((np.array(q[:4]).reshape((1,4)), np.zeros((1,3)))).flatten()
        q4 = np.hstack((np.array(q[:5]).reshape((1,5)), np.zeros((1,2)))).flatten()
        q5 = np.hstack((np.array(q[:6]).reshape((1,6)), np.zeros((1,1)))).flatten()
        q6 = np.array(q[:7]).reshape((1,7)).flatten()

        Jv1 = calcJacobian(q0)[:3,:]
        Jv2 = calcJacobian(q1)[:3,:]
        Jv3 = calcJacobian(q2)[:3,:]
        Jv4 = calcJacobian(q3)[:3,:]
        Jv5 = calcJacobian(q4)[:3,:]
        Jv6 = calcJacobian(q5)[:3,:]
        Jv7 = calcJacobian(q6)[:3,:]

        joint_torque_1 = np.transpose(Jv1) @ joint_forces[:,0]
        joint_torque_2 = np.transpose(Jv2) @ joint_forces[:,1]
        joint_torque_3 = np.transpose(Jv3) @ joint_forces[:,2]
        joint_torque_4 = np.transpose(Jv4) @ joint_forces[:,3]
        joint_torque_5 = np.transpose(Jv5) @ joint_forces[:,4]
        joint_torque_6 = np.transpose(Jv6) @ joint_forces[:,5]
        joint_torque_7 = np.transpose(Jv7) @ joint_forces[:,6]
        joint_torques = joint_torque_1+joint_torque_2+joint_torque_3+joint_torque_4+joint_torque_5+joint_torque_6+joint_torque_7
        ## END STUDENT CODES
        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE

        distance = 0
        distance = np.linalg.norm(target - current)
        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, goal, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        goal - 1x7 numpy array containing the desired joint angles configuration
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        jointPositions_c, T0e_c = PotentialFieldPlanner.fk.forward(q)
        jointPositions_t, T0e_t = PotentialFieldPlanner.fk.forward(goal)
        current = np.transpose(jointPositions_c[:7,:])    # 3x7 np array
        target = np.transpose(jointPositions_t[:7,:])   # 3x7 np array
        
        obstacle = np.array(map_struct.obstacles)
        joint_forces = PotentialFieldPlanner.compute_forces(target,obstacle,current)
        torque = PotentialFieldPlanner.compute_torques(joint_forces,q)
        # dq = torque/np.linalg.norm(torque)
        dq = torque
        ## END STUDENT CODE
        if np.isnan(np.any(dq)):
            print('dq error')
        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        q = np.array([]).reshape(0,7)
        start = np.array(start).reshape(1,7)      
        # alpha = 0.001+0.11*math.exp(-len(q_path)/100)
        alpha = 0.037
        q_path = np.concatenate((q_path,start), axis=0)
        q = np.concatenate((q,start), axis=0)
        count=0
        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            # print('planner',q_path[-1,:])
            dq = PotentialFieldPlanner.compute_gradient(q_path[-1,:],goal,map_struct)
            
            q_new = q_path[-1,:]+alpha*dq/np.linalg.norm(dq)
            q_path = np.concatenate((q_path,q_new.reshape((1,7))), axis=0)


            # Termination Conditions
            print(len(q_path))
            step_size = np.linalg.norm(dq)
            # print('step',step_size)
            dist2target = PotentialFieldPlanner.q_distance(goal,q_path[-1,:])
            if len(q_path)>=self.max_steps or dist2target < self.tol or step_size < self.min_step_size: # TODO: check termination conditions
                break # exit the while loop if conditions are met!
            # if np.any(q_path[-1,:]<PotentialFieldPlanner.lower) or np.any(q_path[-1,:]>PotentialFieldPlanner.upper):
            #     q_path = np.delete(q_path,-1,axis=0)
            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function 

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            if len(q_path)>3 and step_size < self.min_step_size:
                dq = 2*(np.random.rand(1,7)-np.ones((1,7)))
                dq = dq/np.linalg.norm(dq)
                q_new = q_path[-1,:]+alpha * dq
                q_path = np.concatenate((q_path,q_new), axis=0)
                count = count+1
            ## END STUDENT CODE
        print('count=',count)
        q_path = np.concatenate((q_path,start), axis=0)
        return q

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map2.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    # start = np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.0])
    goal =  np.array([-1.2, 1.57 , 1.57, -1.8, -1.57, 1.57, 0.0])
    # goal = np.array([0.2,0.3,1,1,-1,1,0.0])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
    plt.plot(q_path)
    plt.show()
