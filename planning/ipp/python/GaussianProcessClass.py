#!/usr/bin/env python3

# Standard tools
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import torch

# GPyTorch imports
from gpytorch.priors import NormalPrior
from gpytorch.constraints import Interval
from gpytorch.models import VariationalGP, ExactGP
import gpytorch.variational
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, GaussianSymmetrizedKLKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.test.utils import least_used_cuda_device
import gpytorch.settings

# BOtorch
from botorch.fit import fit_gpytorch_mll
import botorch.models
from botorch.models.approximate_gp import (
    _SingleTaskVariationalGP,
    ApproximateGPyTorchModel,
    SingleTaskVariationalGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Log
from botorch.models.utils.inducing_point_allocators import (
    GreedyImprovementReduction,
    GreedyVarianceReduction,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

# ROS imports
import rospy
import actionlib
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Int32, Float32, Int32MultiArray
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_matrix

# Custom imports
from gp_mapping.convergence import ExpMAStoppingCriterion
from slam_msgs.msg import PlotPosteriorResult, PlotPosteriorAction
from slam_msgs.msg import ManipulatePosteriorAction, ManipulatePosteriorResult
from slam_msgs.msg import SamplePosteriorAction, SamplePosteriorResult
from slam_msgs.msg import MinibatchTrainingAction, MinibatchTrainingResult, MinibatchTrainingGoal
from slam_msgs.srv import Resample, ResampleResponse

# Python tools
import warnings
import os
import time
import tf
import pathlib
import ast
import copy
from collections import OrderedDict
from threading import Lock


class frozen_SVGP():
    
    def __init__(self):
        self.real_beams         = np.empty((0, 3))
        self.simulated_beams    = np.empty((0, 3))
        mb_gp_name = rospy.get_param("~minibatch_gp_server")
        self.ac_mb = actionlib.SimpleActionClient(mb_gp_name, MinibatchTrainingAction)
        self.prior_mean = rospy.get_param("~prior_mean")
        self.prior_vari = rospy.get_param("~prior_vari")
        self.num_inducing = rospy.get_param("~svgp_num_ind_points")
        self.mb_size = rospy.get_param("~svgp_minibatch_size")
        self.lr = rospy.get_param("~svgp_learning_rate")
        self.rtol = rospy.get_param("~svgp_rtol")
        self.n_window = rospy.get_param("~svgp_n_window")
        self.auto = rospy.get_param("~svgp_auto_stop")
        self.verbose = rospy.get_param("~svgp_verbose")
        assert isinstance(self.num_inducing, int)
        self.s = int(self.num_inducing)
        self.model = botorch.models.SingleTaskVariationalGP(
                train_X=torch.randn(self.s,2),
                num_outputs=1,
                inducing_points = torch.randn(self.s,2),
                variational_distribution=gpytorch.variational.CholeskyVariationalDistribution(self.s),
                likelihood=GaussianLikelihood(),
                learn_inducing_points=True,
                mean_module = ConstantMean(constant_prior=NormalPrior(self.prior_mean, self.prior_vari)),
                covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood = GaussianLikelihood()
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.model, self.mb_size, combine_terms=True)
        self.likelihood.to(self.device).float()
        self.model.to(self.device).float()
        self.opt = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=float(self.lr))
    
    def train_simulated_and_real_iteration(self):
        
        split_mb_size = int(self.mb_size/2)  
        idx         = np.random.choice(self.simulated_beams.shape[0]-1, split_mb_size, replace=False)
        beams1      = self.simulated_beams[idx,:]
        
        idx         = np.random.choice(self.real_beams.shape[0]-1, split_mb_size, replace=False)
        beams2      = self.real_beams[idx,:]
        beams       = np.concatenate((beams1, beams2), axis=0)
        
        input       = torch.from_numpy(beams[:,0:2]).to(self.device).float()
        target      = torch.from_numpy(beams[:,2]).to(self.device).float()
        
        # # compute loss, compute gradient, and update
        self.opt.zero_grad()
        loss = -self.mll(self.model(input), target)
        loss.backward()
        self.opt.step()

        del input
        del target

                         
class SVGP_map():
    """ Class which encapsulates the optimization tools for the SVGP map.
        Also provides ROS interface for subscription and publishing to topics,
        and an action server for accessing posterior.
    """

    def __init__(self, particle_id, corners = None):

        # Mutex to share with planner
        self.mutex = Lock()
        
        ## ROS INTERFACE
        self.particle_id = particle_id
        self.storage_path = rospy.get_param("~results_path")
        self.count_training = 0
        
        # AS for posterior region
        sample_gp_name = rospy.get_param("~sample_gp_server")
        self._as_posterior = actionlib.SimpleActionServer(sample_gp_name, SamplePosteriorAction,
                                                     execute_cb=self.full_posterior_cb, auto_start=False)
        self._as_posterior.start()
        
        # Beam storage
        self.real_beams         = np.empty((0, 3))
        self.simulated_beams    = np.empty((0, 3))

        # AS for expected meas
        manipulate_gp_name = rospy.get_param("~manipulate_gp_server")
        #self._as_manipulate = actionlib.SimpleActionServer("/particle_" + str(self.particle_id) + manipulate_gp_name, ManipulatePosteriorAction, 
        #                                        execute_cb=self.manipulate_posterior_cb, auto_start = False)
        self._as_manipulate = actionlib.SimpleActionServer(manipulate_gp_name, ManipulatePosteriorAction, 
                                               execute_cb=self.manipulate_posterior_cb, auto_start = False)
        self._as_manipulate.start()
        
        self.training = False
        self.plotting = False
        self.sampling = False
        self.resampling = False

        # AS for minibath training data from RBPF
        mb_gp_name = rospy.get_param("~minibatch_gp_server")
        self.ac_mb = actionlib.SimpleActionClient(mb_gp_name, MinibatchTrainingAction)
        while not self.ac_mb.wait_for_server(timeout=rospy.Duration(5)) and not rospy.is_shutdown():
            print("Waiting for MB AS ", particle_id)
        
         # Subscription to GP inducing points from RBPF
        ip_top = rospy.get_param("~inducing_points_top")
        rospy.Subscriber(ip_top, PointCloud2, self.ip_cb, queue_size=1)
        self.inducing_points_received = False

        # Subscription to particle resampling indexes from RBPF
        p_resampling_top = rospy.get_param("~gp_resampling_top")
        self.resample_srv = rospy.Service(str(p_resampling_top) + "/particle_" + str(self.particle_id), Resample,
                         self.resampling_cb)

        mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/expected_mbes')
        self.pcloud_pub = rospy.Publisher(mbes_pc_top, PointCloud2, queue_size=10)

        ## SVGP SETUP
        self.mb_size = rospy.get_param("~svgp_minibatch_size")
        self.lr = rospy.get_param("~svgp_learning_rate")
        self.rtol = rospy.get_param("~svgp_rtol")
        self.n_window = rospy.get_param("~svgp_n_window")
        self.auto = rospy.get_param("~svgp_auto_stop")
        self.verbose = rospy.get_param("~svgp_verbose")
        #n_beams_mbes = rospy.get_param("~n_beams_mbes", 1000)

        # Number of inducing points
        num_inducing = rospy.get_param("~svgp_num_ind_points")
        assert isinstance(num_inducing, int)
        self.s = int(num_inducing)
        
        self.prior_mean = rospy.get_param("~prior_mean")
        self.prior_vari = rospy.get_param("~prior_vari")

        # hardware allocation
        initial_x = torch.randn(self.s,2)
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(self.s)
        self.model = botorch.models.SingleTaskVariationalGP(
            train_X=initial_x,
            num_outputs=1,
            variational_distribution=var_dist,
            likelihood=GaussianLikelihood(),
            learn_inducing_points=True,
            mean_module = ConstantMean(constant_prior=NormalPrior(self.prior_mean, self.prior_vari)),
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, )))
        self.likelihood = GaussianLikelihood()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.model, self.mb_size, combine_terms=True)
        self.likelihood.to(self.device).float()
        self.model.to(self.device).float()
        
        self.opt = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=float(self.lr))

        # Convergence criterion
        self.criterion = ExpMAStoppingCriterion(rel_tol=float(self.rtol), 
                                                minimize=True, n_window=self.n_window)
        self.ready_for_LC = False # Set to true when ELBO converges
        enable_lc_top = rospy.get_param("~particle_enable_lc", '/enable_lc')
        self.enable_lc_pub = rospy.Publisher(enable_lc_top, Int32, queue_size=10)
        
        # Toggle training mode
        self.model.train()
        self.likelihood.train()
        self.loss = list()
        self.iterations = 0

        self.listener = tf.TransformListener()

        print("Particle ", self.particle_id, " set up")

        # Remove Qt out of main thread warning (use with caution)
        warnings.filterwarnings("ignore")

        self.n_plot = 0
        self.n_plot_loss = 0
        self.mission_finished = False

    def resampling_cb(self, req):

        self.resampling = True
        while not rospy.is_shutdown() and self.training:
            rospy.sleep(0.01)
            rospy.logdebug("GP %s waiting for training before resampling", self.particle_id)

        # If this particle has been resampled, save SVGP to disk
        # to share it with the rest
        response = ResampleResponse(True)
        if req.p_id == self.particle_id:
            self.save("/home/orin/catkin_ws/src/UWExploration/utils/uw_tests/rbpf" + "/svgp_" + str(req.p_id) + ".pth")
            print("Particle ", req.p_id, " saved to disk")
        
        # if 0 == self.particle_id:    
            # my_file = Path(self.storage_path + "svgp_" + str(req.p_id) + ".pth")
            # try:
            #     if my_file.is_file():
            #         self.load(str(my_file.as_posix()))
            # except FileNotFoundError:
            #     rospy.logerr("Particle failed to load SVGP")
            # response = ResampleResponse(False)

        # Else, load the SVGP from the disk with the particle ID received in the msg
        else:
            ## Loading from disk
            # print("Particle ", self.particle_id,
            #       "loading particle ", req.p_id)
            my_file = pathlib.Path("/home/orin/catkin_ws/src/UWExploration/utils/uw_tests/rbpf" + "/svgp_" + str(req.p_id) + ".pth")
            try:
                if my_file.is_file():
                    self.load(str(my_file.as_posix()))
            except FileNotFoundError:
                rospy.logerr("Particle failed to load SVGP")
            response = ResampleResponse(False)

        self.resampling = False

        return response
    
 
    def train_iteration(self):

        # Don't train until the inducing points from the RBPF node have been received
        if not self.inducing_points_received:
            rospy.loginfo_once("Waiting for inducing points")
            return

        if self.mission_finished:
            rospy.loginfo_once("GP finished %s", self.particle_id)
            return


        # Get beams for minibatch training as pcl
        goal = MinibatchTrainingGoal()
        goal.particle_id = self.particle_id
        goal.mb_size = self.mb_size
        self.ac_mb.send_goal(goal)
        self.ac_mb.wait_for_result()
        result = self.ac_mb.get_result()

        # If minibatch received from server
        try:  
              
            if result.success:
                # Store beams as array of 3D points
                beams = np.asarray(list(pc2.read_points(result.minibatch, 
                                        field_names = ("x", "y", "z"), skip_nans=True)))
                beams = np.reshape(beams, (-1,3))
                
                idx = np.random.choice(beams.shape[0]-1, 50, replace=False)
                self.real_beams = np.concatenate((self.real_beams, beams[idx,:]), axis=0)
                # Sample UIs covariances
                
                if not self.plotting and not self.sampling and not self.resampling:
                    with self.mutex:
                        self.training = True
                        
                        input = torch.from_numpy(beams[:, 0:2]).to(self.device).float()
                        target = torch.from_numpy(beams[:,2]).to(self.device).float()

                        # # compute loss, compute gradient, and update
                        self.opt.zero_grad()
                        loss = -self.mll(self.model(input), target)
                        loss.backward()
                        self.opt.step()

                        del input
                        del target
                        torch.cuda.empty_cache()
                        self.training = False
                        self.iterations += 1

                        # Check for ELBO convergence to signal this SVGP is ready
                        # to start LC prompting
                        loss_np = loss.detach().cpu().numpy()
                        # if not self.ready_for_LC:
                        #     # Delete self.criterion when ready for LCs. It consumes mem af
                        #     if self.criterion.evaluate(torch.from_numpy(loss_np)):
                        #         print("Particle ", self.particle_id, " ready for LCs ")
                        #         self.ready_for_LC = True
                        #         self.enable_lc_pub.publish(self.particle_id)
                        #         del self.criterion
                            
                        # Store loss for postprocessing
                        self.loss.append(loss_np)

                        if self.particle_id == 0:
                            if self.verbose == True:
                                print("Particle ", self.particle_id,
                                    "with iterations: ", self.iterations) #, "Training time ", time.time() - time_start)
                        # print("Training time ", time.time() - time_start)

                else:
                    rospy.logdebug("GP missed MB %s", self.particle_id)
                    rospy.sleep(0.1)
        
        except AttributeError:
            rospy.sleep(0.1)
            pass

        # print("Done with the training ", self.particle_id)

    def ip_cb(self, ip_cloud):
        print("Particle ", self.particle_id, " received inducing points")
        
        if not self.inducing_points_received:
            # Store beams as array of 3D points
            wp_locations = []
            for p_utm in pc2.read_points(ip_cloud, 
                                    field_names = ("x", "y", "z"), skip_nans=True):
                # p_map = np.dot(self.rot, p_utm)
                # p_map = np.add(p_map, -self.trans)
                p_map = p_utm
                wp_locations.append((p_map[0], p_map[1], p_map[2]))
                
            wp_locations = np.asarray(wp_locations)
            wp_locations = np.reshape(wp_locations, (-1,3))
            wp_locations[0, 2] += 1.
            wp_locations[-1, 2] += -1.

            # Distribute IPs evenly over irregular-shaped target area
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(wp_locations)
            tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(
                pcd)
            alpha = 1000000000.0
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha, tetra_mesh, pt_map)
            pcl = mesh.sample_points_poisson_disk(
                number_of_points=int(self.s))

            self.model.model.variational_strategy.inducing_points.data = torch.from_numpy(
                np.asarray(pcl.points)[:, 0:2]).to(self.device).float()

            self.inducing_points_received = True
            print("Particle ", self.particle_id, " starting training")


    ## AS for interfacing the sampling, plotting or saving to disk of the GP posterior
    def manipulate_posterior_cb(self, goal):

        beams = np.asarray(list(pc2.read_points(goal.pings, 
                                field_names = ("x", "y", "z"), skip_nans=True)))
        beams = np.reshape(beams, (-1,3)) 

        self.sampling = True
        while not rospy.is_shutdown() and self.training:
            rospy.sleep(0.01)
            rospy.logdebug(
                "GP %s waiting for training before sampling/saving", self.particle_id)

        if goal.sample:
            mu, sigma = self.sample(np.asarray(beams)[:, 0:2])
            self.sampling = False

            # Set action as success
            result = ManipulatePosteriorResult()
            result.p_id = self.particle_id
            result.mu = mu
            result.sigma = sigma
            self._as_manipulate.set_succeeded(result)
            # print("GP ", self.particle_id, " sampled")

            # Publish for testing
            # mbes_gp = np.concatenate((np.asarray(beams)[:, 0:2], np.reshape(mu, (-1,1))), axis=1)
            # mbes_pcloud = self.pack_cloud("map", mbes_gp)
            # self.pcloud_pub.publish(mbes_pcloud)
        #####
        else:
            # Flag to stop training
            self.plotting = True
            
            # Plot posterior and save it to image
            if goal.plot:

                # Plot the loss
                self.plot_loss(self.storage_path + '/particle_' + str(self.particle_id) 
                        + '_training_loss_' + str(self.n_plot_loss) + '.png' )
                self.n_plot_loss += 1

                # Plot the GP posterior
                self.plot(beams[:,0:2], beams[:,2], 
                            self.storage_path + '/particle_' + str(self.particle_id) 
                            + '_training_' + str(self.n_plot) + '.png',
                            n=50, n_contours=100 )
                self.n_plot += 1

            # Save to disk 
            else:
                track_position = np.asarray(list(pc2.read_points(goal.track_position, 
                                field_names = ("x", "y", "z"), skip_nans=True)))
                track_position = np.reshape(track_position, (-1,3)) 

                track_orientation = np.asarray(list(pc2.read_points(goal.track_orientation, 
                                field_names = ("x", "y", "z"), skip_nans=True)))
                track_orientation = np.reshape(track_orientation, (-1,3)) 

                # Save GP hyperparams
                self.save(self.storage_path + "/svgp_final_" +
                        str(self.particle_id) + ".pth")
                # Save particle's MBES map, trajectory and loss
                np.savez(self.storage_path + "/data_particle_" +
                        str(self.particle_id) + ".npz", beams=beams, loss=self.loss, 
                        track_position=track_position, track_orientation=track_orientation)
                self.plotting = False
                self.mission_finished = True

                del self.model
                del self.likelihood
                del self.opt
                torch.cuda.empty_cache()
                
                # Plot the loss
                # self.plot_loss(self.storage_path + '/particle_' + str(self.particle_id) 
                #         + '_training_loss_' + str(self.n_plot_loss) + '.png' )
                # self.n_plot_loss += 1

            # Set action as success
            result = ManipulatePosteriorResult()
            result.success = True
            self._as_manipulate.set_succeeded(result)
            self.plotting = False
            print("Saved GP ", self.particle_id, " after ", self.iterations, " iterations")


    def sample(self, x):

        '''
        Samples the posterior at x
        x: (n,2) numpy array
        returns:
            mu: (n,) numpy array of predictive mean at x
            sigma: (n,) numpy array of predictive variance at x
        '''
        
        self.likelihood.eval()
        self.model.eval()

        # sanity
        assert len(x.shape) == x.shape[1] == 2

        # sample posterior
        # TODO: fast_pred_var activates LOVE. Test performance on PF
        # https://towardsdatascience.com/gaussian-process-regression-using-gpytorch-2c174286f9cc
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # time_start = time.time()
            x = torch.from_numpy(x).to(self.device).float()
            dist = self.likelihood(self.model(x))
            
            # print("Sampling time ", time.time() - time_start)
        self.likelihood.train()
        self.model.train()
        return dist.mean.cpu().numpy(), dist.variance.cpu().numpy()
        


    def full_posterior_cb(self, goal):

        # toggle evaluation mode
        self.likelihood.eval()
        self.model.eval()

        # posterior sampling locations
        inputs = [
            np.linspace(goal.xmin, goal.xmax, goal.n),
            np.linspace(goal.ymin, goal.ymax, goal.n)
        ]
        inputs = np.meshgrid(*inputs)
        inputs = [_.flatten() for _ in inputs]
        inputs = np.vstack(inputs).transpose()

        # split the array into smaller ones for memory
        inputs = np.split(inputs, goal.subdivs, axis=0)

        # compute the posterior for each batch
        means, variances = list(), list()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i, input in enumerate(inputs):
                input = torch.from_numpy(input).to(self.device).float()
                dist = self.likelihood(self.model(input))
                means.append(dist.mean.cpu().numpy())
                variances.append(dist.variance.cpu().numpy())

        # assemble probabalistic pointcloud
        cloud = np.hstack((
            np.vstack(inputs),
            np.hstack(means).reshape(-1, 1),
            np.hstack(variances).reshape(-1, 1)
        ))

        self.likelihood.train()
        self.model.train()

        result = SamplePosteriorResult()
        result.posterior = cloud
        self._as_posterior.set_succeeded(result)



    def plot(self, inputs, targets, fname, n=80, n_contours=50):

        '''
        Plots:
            ax[0]: raw inputs and targets,
            ax[1]: posterior predictive mean,
            ax[2]: posterior predictive variance
        inputs: (n,2) numpy array of inputs
        output: (n,) numpy array of targets
        fname: path to save plot at (extension determines file type, e.g. .png or .pdf)
        n: determines n² number of sampling locations to plot GP posterior
        n_contours: number of contours to show output magnitude with
        '''

        # sanity
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == 2

        # toggle evaluation mode
        self.likelihood.eval()
        self.model.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputsg = [
            np.linspace(min(inputs[:,0]), max(inputs[:,0]), n),
            np.linspace(min(inputs[:,1]), max(inputs[:,1]), n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        inputst = torch.from_numpy(inputst).to(self.device).float()

        # sample
        with torch.no_grad():
            outputs = self.model(inputst)
            outputs = self.likelihood(outputs)
            mean = outputs.mean.cpu().numpy().reshape(s)
            variance = outputs.variance.cpu().numpy().reshape(s)

        # plot raw, mean, and variance
        levels = np.linspace(min(targets), max(targets), n_contours)
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        cr = ax[0].scatter(inputs[:,0], inputs[:,1], c=targets, cmap='viridis', s=0.4, edgecolors='none')
        # cm = ax[1].contourf(*inputsg, mean, levels=n_contours)
        cm = ax[1].contourf(*inputsg, mean, levels=levels)
        cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
        indpts = self.model.variational_strategy.inducing_points.data.cpu().numpy()
        ax[2].plot(indpts[:,0], indpts[:,1], 'ko', markersize=1, alpha=0.2)

        # colorbars
        fig.colorbar(cr, ax=ax[0])
        fig.colorbar(cm, ax=ax[1])
        fig.colorbar(cv, ax=ax[2])

        # formatting
        ax[0].set_aspect('equal')
        ax[0].set_title('Raw data')
        ax[0].set_ylabel('$y~[m]$')
        ax[1].set_aspect('equal')
        ax[1].set_title('Mean')
        ax[1].set_ylabel('$y~[m]$')
        ax[2].set_aspect('equal')
        ax[2].set_title('Variance')
        ax[2].set_xlabel('$x~[m]$')
        ax[2].set_ylabel('$y~[m]$')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

        # Free up GPU mem
        del inputst
        torch.cuda.empty_cache()

    def plot_loss(self, fname):

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(self.loss, 'k-')

        # format
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_yscale('log')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)
        
    def save(self, fname):
        time_start = time.time()
        torch.save({'model' : self.model.state_dict(),
                    'likelihood' : self.likelihood.state_dict(),
                    'mll' : self.mll.state_dict(),
                    'opt': self.opt.state_dict()}, fname)

    def load(self, fname):
        time_start = time.time()
        cp = torch.load(fname)
        self.model.load_state_dict(cp['model'])
        self.likelihood.load_state_dict(cp['likelihood'])
        self.mll.load_state_dict(cp['mll'])
        self.opt.load_state_dict(cp['opt'])
        
        # For localization testing
        # self.model.load_state_dict(torch.load(fname), strict=False)
        
        self.model.train()
        self.likelihood.train() 

    def pack_cloud(self, frame, mbes):
        mbes_pcloud = PointCloud2()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        mbes_pcloud = point_cloud2.create_cloud(header, fields, mbes)
        return mbes_pcloud 



if __name__ == '__main__':

    rospy.init_node('ipp_svgp' , disable_signals=False)
    node_name = rospy.get_name()
    namespace = rospy.get_namespace()
    node_name = node_name.replace(namespace, '')
    hdl_number = int(node_name.split('_')[2])
    particles_per_hdl = rospy.get_param("~num_particles_per_handler")

    try:
        particles_svgps = []
        # particles_ids = []
        # Create the SVGP maps for this handler
        for i in range(0, int(particles_per_hdl)):
            particles_svgps.append(SVGP_map(int(hdl_number)+i))
            # particles_ids.append(int(hdl_number)+i)

        # In each round, call one minibatch training iteration per SVGP
        r = rospy.Rate(30)
        while not rospy.is_shutdown():
            for i in range(0, int(particles_per_hdl)):
                particles_svgps[i].train_iteration()  
                r.sleep()

        # rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_svgp")
