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
    def __init__(self, parent, joint_pos, ef_pos, q, idx):
        self.parent_idx = parent    # Parent node index
        self.idx = idx              # Index of the node   
        self.joint_pos = joint_pos  # Joint positions of the node (x, y, z) 8x3 matrix
        self.ef_pos = ef_pos        # End effector position of the node (x, y, z)
        self.q = q                  # Joint angles of the node (q1, q2, q3, q4, q5, q6, q7)

class Tree:
    def __init__(self):
        self.nodes = []
        self.node_count = 0

    def addNode(self, parent, joint_pos, ef_pos, q):
        current_idx = self.node_count
        self.nodes.append(Node(parent, joint_pos, ef_pos, q, current_idx))
        self.node_count += 1

    def getNode(self, index):
        return self.nodes[index]

    def size(self):
        return len(self.nodes)

    def get_path(self, index):
        path = []
        while index is not None:
            path.append(self.nodes[index].q)
            index = self.nodes[index].parent_idx
        return path[::-1]

class RRTConnect:
    def __init__(self, map, start, goal):
        self.fk = FK()
        self.ik = IK()

        self.map = map
        self.map.boundary = np.array([-1.0, -1.0, -0.5, 1.0, 1.0, 1.5])     # [xmin, ymin, zmin, xmax, ymax, zmax]

        self.startTree = Tree()
        self.start_joint_pos, self.start_ef_pose = self.fk.forward(start)
        self.startTree.addNode(None, self.start_joint_pos, self.start_ef_pose[:3,3], start)

        self.goalTree = Tree()
        self.goal_joint_pos, self.goal_ef_pose = self.fk.forward(goal)
        self.goalTree.addNode(None, self.goal_joint_pos, self.goal_ef_pose[:3,3], goal)

        self.reached = False
        self.max_iter = 1000
        self.step_size = 0.1            # meters
        self.sphere_radius = 0.2        # meters
    
    def solve(self):
        idx = 0
        while idx < self.max_iter:
            random_node = self.sampleNode()   # This is assumed to be a node that is sampled from the free task space

            # Extend the start tree towards the random node
            nearest_node = self.nearestNodeInTree(random_node, self.startTree)
            new_node = self.extendedNode(nearest_node, random_node)
            if new_node is not None:
                self.startTree.addNode(nearest_node.idx, new_node.joint_pos, new_node.ef_pose, new_node.q)

            # Extend the goal tree towards the random node
            nearest_node = self.nearestNodeInTree(random_node, self.goalTree)
            new_node = self.extendedNode(nearest_node, random_node)
            if new_node is not None:
                self.goalTree.addNode(nearest_node.idx, new_node.joint_pos, new_node.ef_pos, new_node.q)

            # Check if the trees have reached each other
            last_start_node = self.startTree.getNode(self.startTree.size() - 1)
            last_goal_node = self.goalTree.getNode(self.goalTree.size() - 1)

            trees_connected = connectTrees(last_start_node, last_goal_node)

            if trees_connected:
                self.reached = True
                return self.startTree.get_path(self.startTree.size() - 1) + self.goalTree.get_path(self.goalTree.size() - 1)[::-1]
        
        # Could not find a path
        return []
    
    def nearestNodeInTree(self, node, tree):
        # Find the nearest node in the tree to the given node
        min_dist = float('inf')
        nearest_node = None
        for tree_node in tree.nodes:
            dist = np.linalg.norm(tree_node.ef_pos - node.ef_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_node = tree_node
        
        return nearest_node

    def nodeLinkIsInCollision(self, node1, node2):
        """
        Check if there is any collision in the line segment between the two nodes
        """

        # Find the direction to extend the nearest node towards the random node
        dist = np.linalg.norm(node1.ef_pos - node2.ef_pos)
        direction = (node2.ef_pos - node1.ef_pos) / dist

        current_node = node1
        curr_ef_pos = node1.ef_pos
        curr_ef_pose = np.eye(4)
        curr_ef_pose[:3,3] = curr_ef_pos
        curr_ef_pose[:3,:3] = self.goal_ef_pose[:3,:3]
        distance_traveled = 0
        while distance_traveled < dist:
            if self.armIsInCollision(current_node):
                return True

            curr_ef_pos += direction * self.step_size
            curr_ef_pose[:3,3] = curr_ef_pos

            current_joint_pos, current_ef_pose = self.fk.forward(current_q)
            current_node = Node(None, current_joint_pos, current_ef_pose[:3,3], current_q, None)
            if self.armIsInCollision(current_node):
                return True
            distance_traveled += self.step_size

    def extendedNode(self, nearest_node, random_node):
        # Extend the nearest node towards the random node by the step size
        dist = np.linalg.norm(nearest_node.ef_pos - random_node.ef_pos)
        if dist < self.step_size:
            return random_node
        else:
            # Find the direction to extend the nearest node towards the random node
            direction = (random_node.ef_pos - nearest_node.ef_pos) / dist
            new_ef_pos = nearest_node.ef_pos + self.step_size * direction
            new_joint_pos = self.ik.inverse(new_ef_pos, nearest_node.joint_pos)
            new_q = self.fk.forward(new_joint_pos)
            new_node = Node(nearest_node.idx, new_joint_pos, new_ef_pos, new_q, self.startTree.size())
            if not self.armIsInCollision(new_node):
                return new_node
            else:
                return None

    def sampleNode(self):
        # Sample random point in free space (x, y, z) taking into account the boundary
        sample_is_in_free_space = False
        while(not sample_is_in_free_space):
            x = random.uniform(self.map.boundary[0], self.map.boundary[3])
            y = random.uniform(self.map.boundary[1], self.map.boundary[4])
            z = random.uniform(self.map.boundary[2], self.map.boundary[5])
            for obs in self.map.obstacles:
                if not detectCollision.pointBoxCollision([x, y, z], obs):
                    sample_is_in_free_space = True
                    break

        return np.array([x, y, z])
    
    def armIsInCollision(self, node):
        # Check if the arm is in self collision by checking if any of the joints are in collision with each other
        num_joints = node.joint_pos.shape[0]
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                if detectCollision.sphereSphereCollision(sphere1_pos=node.joint_pos[i], sphere2_pos=node.joint_pos[j], sphere_r=self.sphere_radius):
                    return True

        # Check if the arm is in collision with any of the obstacles
        for obs in self.map.obstacles:
            # Check if any of the joints are in collision with the obstacle
            for i in range(node.joint_pos.shape[0]):
                if detectCollision.sphereBoxCollision(node.joint_pos[i], self.sphere_radius, obs):
                    return True
            # Check if the end effector is in collision with the obstacle
            if detectCollision.pointBoxCollision(node.ef_pos, obs):
                return True


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    path.append(start)
    path.append(goal)

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    return np.array(path)

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Directory of this file: ", dir_path)

    map_struct = loadmap(dir_path + "/../maps/map2.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal =  np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print("Path is:")
    print(path)
