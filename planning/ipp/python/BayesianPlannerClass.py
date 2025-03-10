# Torch libraries
import torch
import botorch

# Math libraries
import dubins
import numpy as np

# ROS imports
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import tf.transformations
import tf
import tf_conversions
import geometry_msgs

# Threading safety support
import filelock

# Python functionality
import filelock
import copy
import time
import pickle

# Custom imports
import PlannerTemplateClass
import MonteCarloTreeClass
import BayesianOptimizerClass
import GaussianProcessClass
import ipp_utils

class BOPlanner(PlannerTemplateClass.PlannerTemplate):
    """ Planner which uses Bayesian Optimization and MCTS.
        This class will publish an initial path when instanciated,
        using a deterministic or random path. It will then periodically
        check the status of waypoints, and when nearing a target, will
        plan a path. When approach is imminent, this path is published.
        The object will save the trained models every time it reaches a target.
    
        This class implements the following functions:
        
        `initial_deterministic_path`
        `initial_random_path`
        `update_wp_cb`
        `calculate_time2target`
        `periodic_call`
        `execute_planning`

    Args:
        PlannerTemplate (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, 
                 training_rate, wp_resolution, swath_width, path_nbr_samples, voxel_size, 
                 wp_sample_interval, horizon_distance, border_margin, beta):
        
        """ Constructor method

        Args:
            corner_topic        (string): publishing topic for corner waypoints
            path_topic          (string): publishing topic for planner waypoints
            planner_req_topic   (string): subscriber topic for callbacks to plan new paths
            odom_topic          (string): subscriber topic for callback to update vehicle odometry
            bounds        (list[double]): [low_x, low_y, high_x, high_y]
            turning_radius      (double): the radius on which the vehicle can turn on yaw axis
            training_rate          (int): rate at which GP is trained
            wp_resolution       (double): resolution at which waypoints along path are generated.
            swath_width         (double): Swath width of MBES sensor
            path_nbr_samples       (int): Number of (orthogonal to path) samples that emulate an MBES swath
            voxel_size          (double): Size of grids used to reduce number of overlapping samples
            wp_sample_interval  (double): Interval of sampling waypoint swaths orthogonally along path
            horizon_distance    (double): distance from current location where candidates are searched for
            border_margin       (double): distance buffer from border where no candidates are searched for
            beta                (double): constant used for UCB acquisition function
            
        """
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, training_rate) 
        
        # Logic checks
        assert border_margin > 0,       "Planner safety/border margin must be positive"
        assert horizon_distance > 0,    "Planner safety horizon must be positive"   
        assert voxel_size > 0,          "Planner voxel size must be positive" 
        
        # Planner variables
        self.planner_initial_pose   = copy.deepcopy(self.state)
        self.wp_list                = []
        self.currently_planning     = False
        self.nbr_wp                 = 0
        self.max_travel_distance    = rospy.get_param("~max_travel_distance")

        # Path publisher - publishes waypoints for AUV to follow
        self.path_pub               = rospy.Publisher(self.path_topic, Path, queue_size=100)
        rospy.sleep(1)  # Give time for topic to be registered
        
        # Filelocks for file mutexes
        self.gp_env_lock            = filelock.FileLock("GP_env.pickle.lock")
        
        # Setup GP for storage
        self.frozen_gp              = GaussianProcessClass.frozen_SVGP()
        self.beams                  = np.empty((0, 3))
        
        # Parameters for optimizer
        self.wp_resolution          = wp_resolution
        self.swath_width            = swath_width
        self.path_nbr_samples       = path_nbr_samples
        self.voxel_size             = voxel_size
        self.wp_sample_interval     = wp_sample_interval
        self.horizon_distance       = horizon_distance
        self.border_margin          = border_margin
        self.beta                   = beta
        
        # Publish an initial path
        initial_path                = self.initial_random_path(1)
        self.path_pub.publish(initial_path) 
        
        # Setup timers for callbacks
        self.finish_imminent        = False
        rospy.Timer(rospy.Duration(2), self.periodic_call)
        self.execute_planner_pub    = rospy.Publisher("execute_planning_topic_handle", std_msgs.msg.Bool, queue_size=1)
        self.execute_planner_sub    = rospy.Subscriber("execute_planning_topic_handle", std_msgs.msg.Bool, self.execute_planning)
        
        # Initiate training of GP
        self.begin_gp_train(rate=self.training_rate)
    
    def initial_deterministic_path(self):
        """ Generates a path to the center of the map, as defined by the boundaries.

        Returns:
            nav_msgs.msg.Path: ROS message with waypoint poses in list
        """
        
        while not self.odom_init and not rospy.is_shutdown():
            print("Planner is waiting for odometry before starting.")
            rospy.sleep(2)

        # Generate point in center of bounds
        x_pos = self.bounds[0] + (self.bounds[2] - self.bounds[0])/2
        y_pos = self.bounds[1] + (self.bounds[3] - self.bounds[1])/2
        samples = np.random.uniform(low=[x_pos - 1, y_pos - 1, -np.pi], high=[x_pos + 1, y_pos + 1, np.pi], size=[1, 3])
        h = std_msgs.msg.Header()
        h.frame_id = self.map_frame
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        for sample in samples:
            path = dubins.shortest_path(self.planner_initial_pose, [sample[0], sample[1], sample[2]], self.turning_radius)
            wp_poses, _ = path.sample_many(self.wp_resolution)

            skip = 1
            if len(wp_poses) == 1:
                skip = 0
            self.wp_list.extend(wp_poses[skip:])
            for pose in wp_poses[skip:]:
                wp = PoseStamped()
                wp.header = h
                wp.pose.position.x = pose[0]
                wp.pose.position.y = pose[1]
                wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
                sampling_path.poses.append(wp)
                self.nbr_wp += 1
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
        
    def initial_random_path(self, nbr_samples = 1):
        """ Generates a set of random waypoints for initial sampling of BO

        Args:
            nbr_samples (int): number of samples. Defaults to 1.

        Returns:
            nav_msgs.msg.Path: ROS message with waypoint poses in list
        """
        
        local_bounds = ipp_utils.generate_local_bounds(self.bounds, self.planner_initial_pose, self.horizon_distance, self.border_margin)
        
        samples = np.random.uniform(low=[local_bounds[0], local_bounds[1], -np.pi], high=[local_bounds[2], local_bounds[3], np.pi], size=[nbr_samples, 3])
        h = std_msgs.msg.Header()
        h.frame_id = self.map_frame
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        for sample in samples:
            path = dubins.shortest_path(self.planner_initial_pose, [sample[0], sample[1], sample[2]], self.turning_radius)
            wp_poses, _ = path.sample_many(self.wp_resolution)
            skip = 1
            if len(wp_poses) == 1:
                skip = 0
            self.wp_list.extend(wp_poses[skip:])
            for pose in wp_poses[skip:]:
                wp = PoseStamped()
                wp.header = h
                wp.pose.position.x = pose[0]
                wp.pose.position.y = pose[1]
                wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
                sampling_path.poses.append(wp)
                self.nbr_wp += 1
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
    
    def update_wp_cb(self, msg):
        """ When called, reduces number of waypoints tracked.
        
            NOTE: Had to be changed to work with real vehicle, so 
            slightly different than intended. To figure out when this
            callback is called, see the topic it is subscribed to. To work
            with a real vehicle, a separate queue is maintained. This is
            the queue that currently triggers planning, which is used
            in the 'periodic_call' function.

        Args:
            msg (bool): not currently used
        """
        self.wp_list.pop(0)
        self.nbr_wp -= 1
            
    def calculate_time2target(self):
        """ Calculates the time to target based on euclidean distance
            along the path. 
            
            NOTE: The Euclidean path is not a perfect measurement,
            but it is fast. This was not used in real experiments,
            as it was found to be unreliable (speed was not as expected).

        Returns:
            double: time to arrival at target location
        """
        speed = self.vehicle_velocity
        distance = 0
        current = copy.deepcopy(self.state)
        for pose in self.wp_list:
            distance += np.hypot(current[0] - pose[0], current[1] - pose[1])
            current = pose
        return distance/speed
            
    
    def periodic_call(self, msg):
        """ Timer based callback which initiates planner actions.

        Args:
            msg (bool): not used.
        """
        #time2target = self.calculate_time2target()
        if self.nbr_wp < 2 and self.currently_planning == False:
            self.currently_planning = True
            self.execute_planner_pub.publish(True)
            print("Beginning planning...")
        
        if self.nbr_wp < 1 and self.finish_imminent == False:
            self.finish_imminent = True
            print("Stopping planning... No more tree nodes will be expanded.")
            
    def execute_planning(self, msg):
        """ 
            This callback essentially runs the entire BO planning loop. To begin with,
            we freeze a copy of the current GP to use for BO. A myopic BO could potentially
            get away with using the current GP, but thread safety becomes an issue there as well.
            
            A tree search (with *sort of* MC based method) performs BO to find
            candidates (locations) in this frozen GP. For each candidate, we can then
            train a new, separate GP where we have simulated MBES measurements as
            if we actually travelled to that candidate location. The tree expands,
            and repeat until we run out of time.
            
            The location itself is the intensive part to find. After deciding that we are 
            running out of time, we take our currently best location, and decide the best
            angle of the connecting Dubins path. The optimal angle is found through
            secondary BO, with rewards from sampling MBES swaths along the candidate 
            paths (defined by the already determined location, and candidate angle).

        Args:
            msg (bool): dummy boolean
        """
        
        if self.distance_travelled < self.max_travel_distance:
        
        
            # Freeze a copy of current model for planning, to let real model keep training
            with self.gp.mutex:
                    torch.save({'model' : self.gp.model.state_dict()}, "GP_env.pickle")
                    print("Froze GP for planning")
                    
            with self.gp_env_lock:
                cp = torch.load("GP_env.pickle")
                self.frozen_gp.model.load_state_dict(cp['model'])
                nbr_beam_samples = min(self.gp.real_beams.shape[0]-1, 10000)
                idx = np.random.choice(self.gp.real_beams.shape[0]-1, nbr_beam_samples, replace=False)
                beams = self.gp.real_beams[idx,:]
                self.frozen_gp.real_beams = beams
                if len(self.wp_list) > 0:
                    points = ipp_utils.generate_points(self.frozen_gp, self.state[:2], self.wp_list[-1][:2])
                    self.frozen_gp.simulated_beams = np.concatenate((points, self.frozen_gp.simulated_beams), axis=0)
            
            # Signature in: Gaussian Process of terrain, xy bounds where we can find solution, current pose
            MCTS = MonteCarloTreeClass.MonteCarloTree(self.state[:2], self.frozen_gp, beta=self.beta, bounds=self.bounds,
                                horizon_distance=self.horizon_distance, border_margin=self.border_margin)
            
            t1 = time.time()
            while self.finish_imminent == False or time.time() - t1 < 75:
                MCTS.iterate()
            
            rush_order_activated = False
            
            try:
                candidates_XY = MCTS.get_best_solution().position
                candidates_XY = (torch.from_numpy(np.array([candidates_XY]))).type(torch.FloatTensor)
            
            except:
                # If MCTS fails then give a myopic quick candidate and tell angle optimization step to hurry up
                rospy.loginfo("MCTS failed to get candidate, fallback used.")
                rush_order_activated  = True
                local_bounds          = ipp_utils.generate_local_bounds(self.bounds, self.planner_initial_pose, self.horizon_distance, self.border_margin)
                self.bounds_XY_torch  = torch.tensor([[local_bounds[0], local_bounds[1]], [local_bounds[2], local_bounds[3]]]).to(torch.float)
                self.XY_acqf          = botorch.acquisition.UpperConfidenceBound(model=self.frozen_gp.model, beta=self.beta)
                candidates_XY, _      = botorch.optim.optimize_acqf(acq_function=self.XY_acqf, bounds=self.bounds_XY_torch, q=1, num_restarts=5, raw_samples=50)
            
            print("XY cand ", candidates_XY)
            
            print("Optimizing angle...")
            BO = BayesianOptimizerClass.BayesianOptimizer(self.state, self.frozen_gp.model, wp_resolution=self.wp_resolution,
                                                        turning_radius=self.turning_radius, swath_width=self.swath_width,
                                                        path_nbr_samples=self.path_nbr_samples, voxel_size=self.voxel_size,
                                                        wp_sample_interval=self.wp_sample_interval)
            
            angle_optim_max_iter = 5
            if rush_order_activated:
                print("Not enough time for full angle optimization, rush order requested")
                angle_optim_max_iter = 0

            candidates_theta, angle_gp  = BO.optimize_theta_with_grad(XY=candidates_XY, max_iter=angle_optim_max_iter, nbr_samples=15)
            
            candidate                   = torch.cat([candidates_XY, candidates_theta], 1).squeeze(0)
            
            with open("angle_gp" + str(self.distance_travelled) + ".pickle", 'wb') as handle:
                pickle.dump(angle_gp, handle)
            
            print("Optimal pose: ", candidate, ", publishing trajectory.")
            
            # Publish this trajectory as a set of waypoints
            h = std_msgs.msg.Header()
            h.stamp = rospy.Time.now()
            h.frame_id = self.map_frame
            sampling_path = Path()
            sampling_path.header = h
            location = candidate.numpy()
            path = dubins.shortest_path(self.planner_initial_pose, [location[0], location[1], location[2]], self.turning_radius)
            wp_poses, _ = path.sample_many(self.wp_resolution)
            skip = 1
            if len(wp_poses) == 1:
                skip = 0
            self.wp_list.extend(wp_poses[skip:])
            for pose in wp_poses[skip:]:       #removing first as hack to ensure AUV doesnt get stuck
                wp = PoseStamped()
                wp.header = h
                wp.pose.position.x = pose[0] 
                wp.pose.position.y = pose[1]  
                wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
                sampling_path.poses.append(wp)
                self.nbr_wp += 1
            self.planner_initial_pose = wp_poses[-1]
            self.path_pub.publish(sampling_path)
            self.currently_planning = False
            self.finish_imminent = False
            
            #torch.save({"model": angle_gp.state_dict()}, self.store_path  + "_GP_" + str(round(self.distance_travelled)) + "_angle.pickle")
            #torch.save({"model": self.frozen_gp.model.state_dict()}, self.store_path + "_GP_" + str(round(self.distance_travelled)) + "_env.pickle")
            #print("Decision models saved.")
            print("Current distance travelled: " + str(round(self.distance_travelled)) + " m.")
            
        else:
            print("******************************************")
            print("Finished simulation, max distance reached.")
            print("******************************************")
        
