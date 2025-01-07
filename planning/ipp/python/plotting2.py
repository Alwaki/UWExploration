import pickle
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from botorch.acquisition import UpperConfidenceBound
#from AcquisitionFunctionClass import UCB_path
import torch
import numpy as np

# Load MBES points
#MBES = pickle.load(open(r"/home/alex/.ros/Mon, 22 Apr 2024 15:24:15_iteration_2312_MBES.pickle", "rb"))
#print(MBES.shape)

name = "angle_gp78.00841861400856"

# Load second model
n = 100
device = torch.device("cpu")
model2 = pickle.load(open(r"/home/alex/.ros/old_data/angle_gps/" + name + ".pickle","rb"))
model2.eval()
model2.likelihood.eval()
likelihood2 = GaussianLikelihood()
likelihood2.to(device).float()
model2.to(device).float()

samples1D = np.linspace(-np.pi, np.pi, n)

# Outputs for GP 2
mean_list = []
var_list = []
with torch.no_grad():
    inputst_temp = torch.from_numpy(samples1D).to(device).float()
    outputs = model2(inputst_temp)
    outputs = likelihood2(outputs)
    mean_list.append(outputs.mean.cpu().numpy())
    var_list.append(outputs.variance.cpu().numpy())

mean2 = np.vstack(mean_list).squeeze(0)
variance2 = np.vstack(var_list).squeeze(0)
plt.plot(samples1D, mean2)
plt.fill_between(samples1D, mean2+np.sqrt(variance2), mean2-np.sqrt(variance2), facecolor='blue', alpha=0.5)
plt.ylabel("Value")
plt.xlabel("Heading [rad]")
plt.savefig("/home/alex/Pictures/BO_heading/" + "plot" + ".png")
plt.show()
# Plot particle trajectory
#ax[0].plot(track[:,0], track[:,1], "-r", linewidth=0.2)

# # save
#fig.savefig(fname, bbox_inches='tight', dpi=1000)