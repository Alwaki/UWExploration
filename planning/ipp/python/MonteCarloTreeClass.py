import numpy as np
from botorch.acquisition import qSimpleRegret, UpperConfidenceBound, qUpperConfidenceBound, qLowerBoundMaxValueEntropy, qKnowledgeGradient, qNoisyExpectedImprovement
import torch
from botorch.optim import optimize_acqf
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import rospy
import ipp_utils
import GaussianProcessClass
import pickle

class Node(object):
    """ 
    A node object for the search tree. The node represents
    a candidate solution for a location. Each node contains its own 
    GP, which child nodes can inherit to retrain.
    """
    
    def __init__(self, position, depth, id_nbr = 0, parent = None, gp = None) -> None:
        """ Constructor

        Args:
            position (list[double]): location in [x y] of node
            depth             (int): depth of current node in search tree
            id_nbr  (int, optional): ID number of node. Defaults to 0, which is root.
            parent (Node, optional): Parent node. Defaults to None.
            gp       (GP, optional): Gaussian process model of environment. Defaults to None.
        """
        self.position           = position
        self.depth              = depth
        self.parent             = parent
        self.children           = []
        self.reward             = -np.inf
        self.visit_count        = 0
        self.id                 = (depth ** 2) * id_nbr
        self.training           = False
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gp                 = gp #.to(self.device)
        self.map_frame          = rospy.get_param("~map_frame")
        self.simulated_points   = np.empty((0,3))
        
        # If not root, we generate and train on simulated points
        if id_nbr > 0:
            points = ipp_utils.generate_points(self.gp, self.parent.position, self.position)
            self.gp.simulated_beams = np.concatenate((points, self.gp.simulated_beams), axis=0)
        training_iteration = 0      
        
        # About 50 iterations allows the simulated points to affect the GP
        while training_iteration < 50:
            self.gp.train_simulated_and_real_iteration()
            training_iteration += 1
    
    
class MonteCarloTree(object):
    """
    Creates a tree of nodes with the goal of finding the best one. While
    method structure follows MCT ideas, it is not entirely a MCT method. 
    The rollout (in literature also called simulation, reward) is 
    different, and designed to take advantage of not needing to expand 
    the last depth layer of the tree (unlike with sequential methods).
    """
    
    def __init__(self, start_position, gp, beta, border_margin, horizon_distance, bounds) -> None:
        """ Constructor

        Args:
            start_position (list[double]): location in [x y] where optimizer searches for candidates from
            gp                       (GP): Gaussian process representation of environment
            beta                 (double): constant used for UCB acquisition function
            border_margin        (double): distance buffer from border where no candidates are searched for
            horizon_distance     (_type_): distance from current location where candidates are searched for
            bounds         (list[double]): [low_x, low_y, high_x, high_y]
        """
        self.root                       = Node(position=start_position, depth=0, gp=gp)
        self.beta                       = beta
        self.horizon_distance           = horizon_distance
        self.border_margin              = border_margin
        self.global_bounds              = bounds
        self.iteration                  = 0
        self.C                          = rospy.get_param("~MCTS_UCT_C")
        self.max_depth                  = rospy.get_param("~MCTS_max_depth")
        self.rollout_reward_distance    = rospy.get_param("~swath_width")/2.0
                
    def iterate(self):
        """
        Can be considered as a tick in the tree search, to determine executions.
        """
        # run iteration
        # 1. select a node
        node = self.select_node()
        # If node has not been visited, rollout to get reward
        if node.visit_count == 0 and node != self.root:
            value = self.rollout_node(node)
            
        # If node has been visited and there is reward, then expand children
        else:
            # If max depth not hit, and not still training, expand
            if node.depth < self.max_depth:
                self.expand_node(node)
                node = node.children[0]
            value = self.rollout_node(node)
        # backpropagate
        self.backpropagate(node, value)
        self.iteration += 1
    
    def get_best_solution(self):
        """ Returns the best current solution, as determined by the reward function.
            We only return a node (which is a candidate action/location to go to) 
            which is in the first depth layer. This is the action that will be 
            executed.

        Returns:
            Node: The node/action/location with the max reward
        """
        values = []
        for child in self.root.children:
            values.append(child.reward)
        max_ind = np.argmax(values)
        return self.root.children[max_ind]
    
    def select_node(self):
        """ Selects a node which is valuable for expansion. Selection
            is based on UCT reward, which is calculated for all nodes.

        Returns:
            Node: most promising node in search tree
        """

        node = self.root
        while node.children != [] and node.depth <= self.max_depth:
            children_uct = np.zeros(len(node.children))
            for id, child in enumerate(node.children):
                children_uct[id] = self.UCT(child)
            max_id = np.argmax(children_uct)
            node = node.children[max_id]
        return node
                
        
    def UCT(self, node):
        """ Calculates the Upper Tree Confidence value for a node.
            Note that if the node has not yet been visited, we set
            it as very valuable. This is later updated by subsequent 
            visits/rollouts.

        Args:
            node (Node): A node from the search tree

        Returns:
            double: the reward value
        """
        # Calculate tree based UCB
        if node.visit_count == 0:
            return np.inf
        return node.reward / node.visit_count + self.C * np.sqrt(np.log(self.root.visit_count)/node.visit_count)
    
    def expand_node(self, node, nbr_children = 3):
        """ Expands the tree in a specific branch. Uses parallel bayesian
            optimization to find candidates (valuable locations), which 
            are used as nodes in the tree.

        Args:
            node (Node): Node in search tree
            nbr_children (int, optional): How many candidates to return from BO. Defaults to 3.
        """
        
        #print("expanding, parent at node depth: " + str(node.depth))
        
        # Tried a few different acqusition functions
        #XY_acqf         = qSimpleRegret(model=node.gp.model)                                # Just pure exploration (but also gets stuck on maxima)
        XY_acqf         = qUpperConfidenceBound(model=node.gp.model, beta=self.beta)       # Issues with getting stuck in local maxima
        #XY_acqf         = qKnowledgeGradient(model=node.gp.model)                          # No, cant fantasize with variational GP
        
        
        local_bounds    = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.horizon_distance, self.border_margin)
        
        bounds_XY_torch = torch.tensor([[local_bounds[0], local_bounds[1]], [local_bounds[2], local_bounds[3]]]).to(torch.float)
        
        candidates, _   = optimize_acqf(acq_function=XY_acqf, bounds=bounds_XY_torch, q=nbr_children, num_restarts=5, raw_samples=100)
        
        
        ipp_utils.save_model(node.gp.model, "Parent_gp.pickle")
        t1 = time.time()
        
        for i in range(nbr_children):
            new_gp = GaussianProcessClass.frozen_SVGP()
            new_gp.model = ipp_utils.load_model(new_gp.model, "Parent_gp.pickle")
            new_gp.real_beams = node.gp.real_beams
            new_gp.simulated_beams = node.gp.simulated_beams
            n = Node(position=list(candidates[i,:].cpu().detach().numpy()), id_nbr=i+1,depth=node.depth + 1, parent=node, gp=new_gp)
            node.children.append(n)
        print("****** TIME TAKEN TO EXPAND NODES: " + str(time.time() - t1) + " ******")
        
    
    def rollout_node(self, node):
        """ Simulates the value of a location inherent in the node.
            To avoid having to go to terminal state (or in general,
            a deeper layer) we randomly sample values of GP around
            the location of the node. The value is determined by
            an acquisition function.

        Args:
            node (Node): A node in the search tree

        Returns:
            double: reward (mean of random samples from acq function)
        """
        
        local_bounds = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.rollout_reward_distance, self.border_margin)
        samples_np = np.random.uniform(low=[local_bounds[0], local_bounds[1]], high=[local_bounds[2], local_bounds[3]], size=[20, 2])
        samples_torch = (torch.from_numpy(samples_np).type(torch.FloatTensor)).unsqueeze(-2)
        
        acq_fun = UpperConfidenceBound(model=node.gp.model, beta=self.beta)
        ucb = acq_fun.forward(samples_torch)
        reward = ucb.mean().item() - node.depth
        return reward
    
    def backpropagate(self, node, value):
        """ Push (backpropagate) rewards up the tree after simulating

        Args:
            node (Node): Node in tree
            value (double): value/reward that should be backpropagated
        """
        
        current = node
        reward = max(current.reward, value)
        current.reward = reward
        current.visit_count += 1
        while current.depth > 0:
            current = current.parent
            if reward > current.reward:
                current.reward = reward
            current.visit_count += 1