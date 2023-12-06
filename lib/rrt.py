import os
import numpy as np
import random
import lib.detectCollision as detectCollision
from lib.loadmap import loadmap
from copy import deepcopy

from lib.IK_position_null import IK
from lib.calculateFK import FK

# Implement RRT Connect for the panda robot arm

class Node:
    def __init__(self, q, ef_pose, joint_positions, parent, valid):
        self.q = q                              # Joint angles of the node (q1, q2, q3, q4, q5, q6, q7)
        self.ef_pose = ef_pose                  # End effector pose
        self.joint_positions = joint_positions  # Joint positions of the node (x, y, z) 8x3 matrix
        self.parent = parent                    # Parent node index
        self.is_valid = valid
    
    def __repr__(self):
        return "Node: " + str(id(self)) + \
               "\nq: " + str(self.q) + \
               "\nef pose:\n\t" + str(self.ef_pose) + \
               "\njoint Pos:\n\t" + str(self.joint_positions) + \
               "\nparent: " + ("None" if self.parent is None else str(id(self.parent))) + "\n"


class RRTConnect:
    def __init__(self, map, start, goal):
        self.lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
        self.upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

        self.fk = FK()
        self.ik = IK()

        self.map = map
        self.map_boundary = np.array([-1.0, -1.0, 0.05, 1.0, 1.0, 1.5])     # [xmin, ymin, zmin, xmax, ymax, zmax]

        self.start_tree = []
        self.start_node = self.createNode(q=start)
        self.start_tree.append(self.start_node)

        self.goal_tree = []
        self.goal_node = self.createNode(q=goal)
        self.goal_tree.append(self.goal_node)

        self.reached = False
        self.max_iter = 1000
        self.step_size = 0.5                # rad
        self.arm_sphere_radius = 0.05       # meters
        self.obs_sphere_radius = 0.15       # meters
    
    def createNode(self, position=None, q=None, parent=None):
        if q is not None:
            joint_positions, ef_pose = self.fk.forward(q)
            return Node(deepcopy(q), ef_pose, joint_positions, parent, True)
        elif position is not None:
            ef_pose = np.eye(4)
            ef_pose[:3,3] = position
            ef_pose[:3,:3] = self.goal_node.ef_pose[:3,:3]  # This is to ensure that the orientation of the end effector is the same as the goal orientation

            if parent is None:
                # If there is no parent node then use the start node as the parent
                q, _, success, message = self.ik.inverse(ef_pose, self.start_node.q, 'J_pseudo', 0.5)
            else:
                q, _, success, message = self.ik.inverse(ef_pose, parent.q, 'J_pseudo', 0.5)
            
            if success:
                joint_positions, ef_pose = self.fk.forward(q)
                return Node(q, ef_pose, joint_positions, parent, True)
            else:
                return Node(q, ef_pose, None, parent, False)

    def solve(self):
        idx = 0
        while idx < self.max_iter:
            sample_node = self.sampleNode()   # This is assumed to be a node that is sampled from the free task space
            
            # Get the node in the start tree that is closest to the sample node and try to connect them
            nearest_start_tree_node = self.nearestNodeInTree(sample_node, start_tree=True)
            connected_sample_to_start_tree, intermediate_nodes, start_sample_node = self.connectNodes(nearest_start_tree_node, sample_node)
            self.start_tree = self.start_tree + intermediate_nodes
            if connected_sample_to_start_tree:
                self.start_tree.append(start_sample_node)

            # Get the node in the goal tree that is closest to the sample node and try to connect them
            nearest_goal_tree_node = self.nearestNodeInTree(sample_node, start_tree=False)
            connected_sample_to_goal_tree, intermediate_nodes, goal_sample_node = self.connectNodes(nearest_goal_tree_node, sample_node)
            self.goal_tree = self.goal_tree + intermediate_nodes
            if connected_sample_to_goal_tree:
                self.goal_tree.append(goal_sample_node)
            
            if connected_sample_to_start_tree and connected_sample_to_goal_tree:
                self.reached = True
                idx += 1
                print("Iteration: {}\tStart tree size: {}\tGoal tree size: {}".format(idx, len(self.start_tree), len(self.goal_tree)))
                return self.getFinalPath(start_sample_node, goal_sample_node)

            idx += 1
            print("Iteration: {}\tStart tree size: {}\tGoal tree size: {}".format(idx, len(self.start_tree), len(self.goal_tree)))

        # Could not find a path
        return []
    
    def sampleNode(self):
        # Sample node within the jount limits
        sample_is_in_collision = True
        while(sample_is_in_collision):
            sample_q = np.random.uniform(self.lower, self.upper)
                    
            # Check if the arm is in collision
            sample_node = self.createNode(q=sample_q, parent=None)
            if sample_node.is_valid:
                if not self.armIsInCollision(sample_node):
                    sample_is_in_collision = False
            
        return sample_node
    
    def nearestNodeInTree(self, node, start_tree):
        if start_tree:
            tree = self.start_tree
        else:
            tree = self.goal_tree

        # Find the nearest node in the tree to the given node
        min_dist = float('inf')
        nearest_node = None
        for tree_node in tree:
            dist = np.linalg.norm(tree_node.q - node.q)
            if dist < min_dist:
                min_dist = dist
                nearest_node = tree_node
        
        return nearest_node

    def connectNodes(self, parent_node, child_node):
        """
        Check if there is any collision in the line segment between the two nodes
        create intermediate nodes and check if the arm is in collision with any of the obstacles
        return true if there was no collision along with the intermediate nodes else return false
        and an empty list
        """
        dist = np.linalg.norm(child_node.q - parent_node.q)
        direction = (child_node.q - parent_node.q) / dist

        if dist < self.step_size:
            child_node_copy = deepcopy(child_node)
            child_node_copy.parent = parent_node
            return True, [], child_node_copy

        q_curr = deepcopy(parent_node.q)
        curr_parent_node = parent_node
        intermediate_nodes = []
        while( np.linalg.norm(q_curr - child_node.q) > self.step_size):
            q_curr += direction * self.step_size
            curr_node = self.createNode(q=q_curr, parent=curr_parent_node)
            if not curr_node.is_valid:
                return False, intermediate_nodes, None
            elif self.armIsInCollision(curr_node):
                return False, intermediate_nodes, None
            else:
                curr_node.parent = curr_parent_node
                intermediate_nodes.append(curr_node)
                curr_parent_node = curr_node

        child_node_copy = deepcopy(child_node)
        child_node_copy.parent = curr_parent_node
        
        return True, intermediate_nodes, child_node_copy

    def armIsInCollision(self, node):
        # Check that joint angles are within limits
        if np.any(node.q < self.lower) or np.any(node.q > self.upper):
            return True

        # Check if the arm is in self collision by checking if any of the joints are in collision with each other
        num_joints = node.joint_positions.shape[0]
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                if detectCollision.sphereSphereCollision(sphere1_pos=node.joint_positions[i], sphere2_pos=node.joint_positions[j], sphere_r=self.arm_sphere_radius):
                    return True

        # Check if the arm is in collision with any of the obstacles
        for obs in self.map.obstacles:
            # Check if any of the joints are in collision with the obstacle
            for i in range(node.joint_positions.shape[0]):
                if detectCollision.sphereBoxCollision(node.joint_positions[i], self.obs_sphere_radius, obs):
                    return True
            # Check if the end effector is in collision with the obstacle
            if detectCollision.pointBoxCollision(node.ef_pose[:3,3], obs):
                return True
        
        # Check if the arm is in collision with the floor
        for i in range(1, node.joint_positions.shape[0]):
            if detectCollision.sphereBoxCollision(node.joint_positions[i], self.obs_sphere_radius , np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 0.00])):
                return True

    def getFinalPath(self, start_tree_final_node, goal_tree_final_node):
        # Get the path from the start node to the goal node
        path = []
        # Start from the start tree
        curr_node = start_tree_final_node
        while curr_node is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        path.reverse()

        # Now append the path from the goal tree
        curr_node = goal_tree_final_node.parent
        while curr_node is not None:
            path.append(curr_node.q)
            curr_node = curr_node.parent
        
        return deepcopy(path)

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    rrt_solver = RRTConnect(map, start, goal)
    path = rrt_solver.solve()

    return path

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Directory of this file: ", dir_path)

    map_struct = loadmap(dir_path + "/../maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal =  np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print("Path is:")
    for p in path:
        print(p)
