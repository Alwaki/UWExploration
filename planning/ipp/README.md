# Informative Path Planning (IPP)

This package belongs to the AUV exploration collection. It was created as a master thesis in 2024 to explore using Bayesian Optimization (BO) together with the Gaussian Process bathymetric maps to plan efficient explorations paths.

## Dependencies
Requires the dependencies used by the overall simulation library (e.g. auvlib, python dependencies) but also GP dependencies used in mapping (pytorch, gpytorch, open3d). Installation instructions for these are in the UWExploration readme.

Besides those baseline libraries, this package also uses botorch for BO, and the dubins package for paths. These can be installed by

```
pip install botorch dubins 
```

## Code Structure
The structure of the project follows most other ROS packages. There is a launch file, which is dedicated to running the ROS master, along with any necessary nodes. The nodes in turn run python executables in dedicated threads.

As a starting point, one can look in `planner_node.py`. This node interfaces with ROS, collects ROS parameters from the parameter server as specified in the launch file, and sets a thread handle for our code.

Following this, a planner object is instantiated. This planner always inherits from the `PlannerTemplateClass.py`, but can take on two different types as specified by the user in the parameters. These two related objects are specified in the `LawnmowerPlannerClass.py` and the `BayesianPlannerClass.py`. While the lawnmower method is relatively self contained (besides relying on the `GaussianProcessClass.py` to create a GP map), the BO method creates several other objects as well. These are the `BayesianOptimierClass.py` to run BO in several layers, and the `MonteCarloTreeClass.py` in order to perform a search several steps ahead.

Other code is mainly utilities for plotting, visualizing, etc.

