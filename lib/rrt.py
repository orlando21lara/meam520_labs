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
        self.fk = FK()
        self.ik = IK()

        self.map = map
        self.map_boundary = np.array([-1.0, -1.0, 0.1, 1.0, 1.0, 1.5])     # [xmin, ymin, zmin, xmax, ymax, zmax]

        self.startTree = []
        self.start_node = self.createNode(q=start)
        self.startTree.append(self.start_node)

        self.goalTree = []
        self.goal_node = self.createNode(q=goal)
        self.goalTree.append(self.goal_node)

        self.reached = False
        self.max_iter = 1000
        self.step_size = 0.1            # meters
        self.sphere_radius = 0.2        # meters
    
    def createNode(self, position=None, q=None, parent=None):
        if q is not None:
            joint_positions, ef_pose = self.fk.forward(q)
            return Node(q, ef_pose, joint_positions, parent, True)
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
            connected_sample_to_start_tree, intermediate_nodes = self.connectNodes(nearest_start_tree_node, sample_node)
            if connected_sample_to_start_tree:
                self.startTree + intermediate_nodes
                sample_node.parent = nearest_start_tree_node
                self.startTree.append(sample_node)

            # Get the node in the goal tree that is closest to the sample node and try to connect them
            nearest_goal_tree_node = self.nearestNodeInTree(sample_node, start_tree=False)
            connected_sample_to_goal_tree, intermediate_nodes = self.connectNodes(sample_node, nearest_goal_tree_node)
            if connected_sample_to_goal_tree:
                self.goalTree + intermediate_nodes
            
            if connected_sample_to_start_tree and connected_sample_to_goal_tree:
                self.reached = True
                return self.getFinalPath()

            idx += 1
            print("Iteration: ", idx)

        # Could not find a path
        return []
    
    def sampleNode(self):
        # Sample random point in free space (x, y, z) taking into account the boundary
        sample_is_in_collision = True
        while(sample_is_in_collision):
            x = random.uniform(self.map_boundary[0], self.map_boundary[3])
            y = random.uniform(self.map_boundary[1], self.map_boundary[4])
            z = random.uniform(self.map_boundary[2], self.map_boundary[5])
            for obs in self.map.obstacles:
                if detectCollision.pointBoxCollision([x, y, z], obs):
                    continue
                    
            # The point is not in collision with any of the obstacles now check if the arm is in collision
            sample_node = self.createNode(position=[x, y, z], parent=None)
            if sample_node.is_valid:
                if not self.armIsInCollision(sample_node):
                    sample_is_in_collision = False
            
        return sample_node
    
    def nearestNodeInTree(self, node, start_tree):
        if start_tree:
            tree = self.startTree
        else:
            tree = self.goalTree

        # Find the nearest node in the tree to the given node
        min_dist = float('inf')
        nearest_node = None
        for tree_node in tree:
            dist = np.linalg.norm(tree_node.ef_pose[:3,3] - node.ef_pose[:3,3])
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
        parent_position = parent_node.ef_pose[:3,3]
        child_position = child_node.ef_pose[:3,3]
        dist = np.linalg.norm(child_position - parent_position)
        direction = (child_position - parent_position) / dist

        dist_traveled = 0.0
        curr_position = parent_position
        curr_parent_node = parent_node
        intermediate_nodes = []
        while(dist_traveled < dist):
            curr_position += direction * self.step_size
            curr_node = self.createNode(position=curr_position, parent=curr_parent_node)
            if not curr_node.is_valid:
                return False, []
            elif self.armIsInCollision(curr_node):
                return False, []
            else:
                intermediate_nodes.append(curr_node)
                curr_parent_node = curr_node
                dist_traveled += self.step_size
        
        return True, intermediate_nodes

    def armIsInCollision(self, node):
        # Check if the arm is in self collision by checking if any of the joints are in collision with each other
        num_joints = node.joint_positions.shape[0]
        for i in range(num_joints):
            for j in range(i + 1, num_joints):
                if detectCollision.sphereSphereCollision(sphere1_pos=node.joint_positions[i], sphere2_pos=node.joint_positions[j], sphere_r=self.sphere_radius):
                    return True

        # Check if the arm is in collision with any of the obstacles
        for obs in self.map.obstacles:
            # Check if any of the joints are in collision with the obstacle
            for i in range(node.joint_pos.shape[0]):
                if detectCollision.sphereBoxCollision(node.joint_positions[i], self.sphere_radius, obs):
                    return True
            # Check if the end effector is in collision with the obstacle
            if detectCollision.pointBoxCollision(node.ef_pose[:3,3], obs):
                return True

    def getFinalPath(self):
        # Get the path from the start node to the goal node
        path = []
        curr_node = self.goal_node
        while curr_node is not None:
            path.append(curr_node)
            curr_node = curr_node.parent
        
        path.reverse()
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Directory of this file: ", dir_path)

    map_struct = loadmap(dir_path + "/../maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal =  np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print("Path is:")
    print(path)
