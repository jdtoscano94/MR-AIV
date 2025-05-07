# %%
import sys
import os

from instant_aiv.models.NNpp import *
from instant_aiv.models.metrics import *
from instant_aiv.manage.plots import *
from instant_aiv.manage.dataloader import *
from scipy.stats import wasserstein_distance
import pickle


# Libraries
import argparse
import numpy as np
import scipy.io
import time
import gc
import pandas as pd


#Plots
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm


import os
import numpy as np
import jax
from jax import jit, grad, vmap, value_and_grad,jvp

import jax.numpy as jnp
import optax

from typing import Tuple
import time


import h5py
from typing import List, Dict
from pyDOE import lhs
import tqdm
import scipy.integrate as integrate
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FormatStrFormatter


# %%
cmap = 'plasma'
num_colors=8
# Create a colormap
cmap = plt.get_cmap(cmap)
colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
colors=colors[:num_colors//4]+colors[3*num_colors//4:]
print(len(colors))
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 20

# %% [markdown]
# # Tunning Parameters

# %%
parser = argparse.ArgumentParser(description='Tunning_parameters')
parser.add_argument('--run_type'   , type=str  , default='Synthetic')
parser.add_argument('--mode'       , type=str  , default='PINN')
parser.add_argument('--Run_MODE'   , type=str  , default='Train')

parser.add_argument('--num_layer'  , type=int  , default=8)
parser.add_argument('--width_layer', type=int  , default=200)
parser.add_argument('--train_ratio' , type=float, default=0.5)
parser.add_argument('--Weight_Norm'    , type=int , default=1)
parser.add_argument('--Mod_MLP'    , type=int , default=0)
parser.add_argument('--Use_ResNet'    , type=int , default=1)
parser.add_argument('--Adaptive'    , type=int , default=1)
parser.add_argument('--Light'    , type=int , default=0)

parser.add_argument('--batch_size' , type=int  , default=10000)
parser.add_argument('--num_gsteps' , type=int  , default=400*(10**3))
parser.add_argument('--Name' , type=str  , default='Synthetic')
parser.add_argument('--offset_k' , type=float  , default=1.0)

parser.add_argument('--normalization' , type=str  , default='chebyshev5')
parser.add_argument('--Equal_Batch' , type=bool  , default=True)
parser.add_argument('--optimizer' , type=str  , default='AdamW')

parser.add_argument('--use_RBA'    , type=int , default=1)
parser.add_argument('--k_samp' , type=float  , default=2.0)
parser.add_argument('--c_samp' , type=float  , default=0.5)
parser.add_argument('--lr_lambdas_0' , type=float  , default=0.1)
parser.add_argument('--gamma' , type=float  , default=0.999)


parser.add_argument('--use_mass'    , type=int , default=1)


args, unknown = parser.parse_known_args()
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# %%
Run_type = args.run_type
Mode=args.mode
#ResNet Architecures
Use_ResNet=args.Use_ResNet 
Adaptive=args.Adaptive 
Light=args.Light 
num_layer=args.num_layer
use_mass=args.use_mass 
offset_k=args.offset_k 
args.Name=args.Name+str(offset_k)
if Use_ResNet:
    num_layer=int((num_layer+2)/2)

width_layer=args.width_layer
layers={
    'P' :[4]  + num_layer*[3*width_layer//4] + [1],
    'c' :[4]  + num_layer*[width_layer] + [1],
    'c_std' :[4]  + num_layer*[width_layer//3] + [1],
    'k' :[4]  + num_layer*[3*width_layer//4] + [1],
    'D' :[4]  + num_layer*[3*width_layer//4] + [1],
}
layers_ref=np.copy(layers['P'])
layers_1=np.copy(layers['P'])

train_ratio=args.train_ratio

Details=f'{args.Name}-Train:{train_ratio}_mass:{args.use_mass}'

Run_MODE=args.Run_MODE

print('Model Architecture:')
print(layers)

activation = 'sin'
initialization = 'xavier'
normalization  = args.normalization
loss_metric ='L2'
Weight_Norm =args.Weight_Norm#Weight Normalization
use_RBA=args.use_RBA
Mod_MLP=args.Mod_MLP  #Modified MLP
normalize_outputs=False
init_zero=True
Adaptive_AF=False

#Optimization
num_gsteps  = args.num_gsteps
batch_size = args.batch_size

#learning 
lam_min=0.0
lr0=2*10**(-3)
lrf=5*10**(-5)
lr_lambdas_0=args.lr_lambdas_0
gamma=args.gamma
max_lambda=lr_lambdas_0/(1-gamma)
print(f'Max lambdas will be:{max_lambda}')
T_e=num_gsteps
decay_rate=0.9
decay_step=0
seed_np=1234
key = jax.random.PRNGKey(seed_np)
key, subkey = jax.random.split(key)
# File Name
dataset_name = f'{Details}_RBA:{use_RBA}'

#Scaling factor
p_scale=10

#Sample Points
N_r=1000000
N_d=1000000

# %%
project_root = find_package_directory() # if error please define the root package directly (e.g. project_root='/users/.../' )

data_path=project_root+'/Data/Synthetic_Data/Realistic.csv'

out_BCs_path=project_root+"/Data/Synthetic_Data/BC_noflow.csv"

inlet1_path=project_root+"/Data/Synthetic_Data/BC_inlet.csv"

outlet1_path=project_root+"/Data/Synthetic_Data/BC_outlet2.csv"

outlet2_path=project_root+"/Data/Synthetic_Data/BC_outlet3.csv"

save_path=project_root+f'/Results/{Run_type}/'
big_data_path=save_path
print(f'Analizing:{data_path}')
print(f'Details:{Details}')
print(f'dataset_name:{dataset_name}')

# %%
use_RBA=int(use_RBA)
print(f'Using RBA:{use_RBA}')

# %% [markdown]
# ## Settings
# 

# %%
# Fetching activation function
activation_fn = ACTIVATION_FUNCTIONS.get(activation.lower())

# Fetching normalization functions and metrics
norm_settings = NORMALIZATION_FUNCTIONS.get(normalization.lower())
norm_fn = norm_settings['fn']
Norm_metric1 = norm_settings['metric1']
Norm_metric2 = norm_settings['metric2']

# Fetching error function
Error_fn = ERROR_FUNCTIONS.get(loss_metric.lower())

#Adjust layer structure to account for Chebyshev Feature Embbeding
for key in layers.keys():
    embed_size=layers[key][1]
    input_dim=layers[key][0]*5
    output_dim=layers[key][-1]
    layers[key]=[input_dim]+(num_layer)*[embed_size]+ [output_dim]
print(f'New architecture:{layers}')


# %%
np.random.seed(seed_np)
params= init_params_dict(layers, initialization,Use_ResNet)

# Model selection
pinn_fn =select_model(WN=Weight_Norm,Use_ResNet=Use_ResNet,Adaptive=Adaptive,Light=Light)
print(f'The model will be: {pinn_fn}')

# %% [markdown]
# ## Load Initialize Models

# %%
save_file_details=project_root+'/Results/Initialization_Details.pkl'

with open(save_file_details, 'rb') as f:
    file_details_loaded = pickle.load(f)
print(file_details_loaded.keys())

# %%
import numpy
print(numpy.__version__)

# %%
M1_all=file_details_loaded['M1']
M2_all=file_details_loaded['M2']
lb_all=file_details_loaded['lb_all']
ub_all=file_details_loaded['ub_all']
off_set=file_details_loaded['off_set']
layers_k=file_details_loaded['layers']
offset_k=file_details_loaded['offset_k']
k_params_file=project_root+file_details_loaded['path']
dataset_k=file_details_loaded['name']
p_scale=file_details_loaded['p_scale']
Norm_fn={}
Norm_fn['c']=norm_fn
Norm_fn['P']=norm_fn

Norm_fn['k']=Encode_Chebysev_5


k_char0=file_details_loaded['k_char0']
P_char=3.938900070900201
loaded_params_k = load_params_dict(k_params_file, dataset_k, layers_k, initialization,type='Final',Use_ResNet=Use_ResNet) 
print('Parameters loaded from file')
params['k']=loaded_params_k['k']
layers['k']=layers_k['k']
params['c']=loaded_params_k['c']
layers['c']=layers_k['c']
params['P']=loaded_params_k['P']
layers['P']=layers_k['P']
params['c_std']=loaded_params_k['c_std']
layers['c_std']=layers_k['c_std']

# %%
optimizer={}
opt_state={}
for key in layers.keys():
    print(f'Initializing Optimizer for:{key}')
    optimizer[key],decay_step= initialize_optimizer(lr0, decay_rate, lrf, decay_step, num_gsteps,optimizer_type=args.optimizer)
    opt_state[key]=optimizer[key].init(params[key])
M1= Norm_metric1
M2= Norm_metric2
print('Norm metrics:',M1,M2)


# %% [markdown]
# ## Create Folders

# %%
subdirs = ["General","Imgs", "Big_Data", "Data", "Flow_rate", "Errors"]
paths = create_and_return_directories(save_path, dataset_name, subdirs)
print(dataset_name)
result_path = paths["General"]
images_path = paths["Imgs"]
big_path = paths["Big_Data"]
Video_path = paths["Flow_rate"]
errors_path = paths["Errors"]


# %% [markdown]
# # Non-Dimensional values

# %%
k1=1*10**(-12)#mm2
D1=3.8*10**(-10)#mm2/s
k2=1*10**(-16)
D2=2.3*10**(-10)
nu_e=7*10**(-7)
rho=993
mu=0.6913e-3
h=0.00034 #Maximum length by 100
L_char=h
U_char=0.5/5#mm/min
D_char=2.4e-10#m2/s
U_char=U_char*(1/1000)*(1/60)#m/s
T_char=L_char/U_char
C_char=1
Pef=U_char*L_char/D_char
mms=10**6
L_char=L_char
k_char=k_char0
P_char=U_char*L_char*mu/k_char
print(T_char,L_char,U_char,C_char,D_char)
print(Pef)
print(k_char)
print(P_char)

T_SI=1
L_SI=1000
D_SI=1#10**6
U_SI=1#1000
P_SI=1
K_SI=1#10**6
SI_fact_steady=np.array([1/T_SI,1/L_SI,1/L_SI,1/L_SI,1/D_SI,1/U_SI,1/U_SI,1/U_SI,1/P_SI,1/K_SI])
SI_fact=np.array([1/T_SI,1/L_SI,1/L_SI,1/L_SI,1])
non_dim_fact_steady=np.array([1/T_char,1/L_char,1/L_char,1/L_char,1/D_char,1/U_char,1/U_char,1/U_char,1/P_char,1/k_char])
non_dim_fact=np.array([1/T_char,1/L_char,1/L_char,1/L_char,1])
print(non_dim_fact)

# %% [markdown]
# ## Load Data
# 

# %%
df = pd.read_csv(data_path)
all_data = df.to_numpy()


# %%
txyzDuvwpk=np.hstack((df['x(mm)'].to_numpy()[:,None],
                      df['x(mm)'].to_numpy()[:,None],
                      df['y(mm)'].to_numpy()[:,None],
                      df['z(mm)'].to_numpy()[:,None],
                      df['D(m^2/s)'].to_numpy()[:,None],
                      df['u(m/s)'].to_numpy()[:,None],
                      df['v(m/s)'].to_numpy()[:,None],
                      df['w(m/s)'].to_numpy()[:,None],
                      df['p(Pa)'].to_numpy()[:,None],
                      df['k(m^2)'].to_numpy()[:,None]))
txyzc=[]
for time_f in tqdm.tqdm(range(90)):
    x_frame=np.hstack((all_data[:,0:1]*0+time_f*60,all_data[:,:3],all_data[:,time_f+9:time_f+10]))
    txyzc.append(x_frame)
txyzc=np.vstack(txyzc)
print(txyzc.shape)
txyzDuvwuvwk=txyzDuvwpk

# %%
txyzc = txyzc[~np.isnan(txyzc).any(axis=1)]
c_min=np.min(txyzc[:,4:5])
c_max=np.max(txyzc[:,4:5])
c_norm=(txyzc[:,4:5]-c_min)/(c_max-c_min)
txyzc = np.hstack((txyzc[:,:4],c_norm))
print('Min and max for txyzc',txyzc.shape)
print(txyzc.min(0))
print(txyzc.max(0))
print('Min and max for txyzDuvwuvwk',txyzDuvwuvwk.shape)
print(txyzDuvwuvwk.min(0))
print(txyzDuvwuvwk.max(0))
txyzc=txyzc*SI_fact
txyzDuvwuvwk=txyzDuvwuvwk*SI_fact_steady
print('SI:Min and max for txyzc',txyzc.shape)
print(txyzc.min(0))
print(txyzc.max(0))
print('SI:Min and max for txyzDuvwuvwk',txyzDuvwuvwk.shape)
print(txyzDuvwuvwk.min(0))
print(txyzDuvwuvwk.max(0))
txyzc=txyzc*non_dim_fact
txyzDuvwuvwk=txyzDuvwuvwk*non_dim_fact_steady
print('ND:Min and max for txyzc',txyzc.shape)
print(txyzc.min(0))
print(txyzc.max(0))
print('ND:Min and max for txyzDuvwuvwk',txyzDuvwuvwk.shape)
print(txyzDuvwuvwk.min(0))
print(txyzDuvwuvwk.max(0))


# %%
X_plot = txyzDuvwuvwk
t,x, y, z,D, u_gt, v_gt, w_gt,p_gt,k = (X_plot[:,0],
                                            X_plot[:,1],
                                            X_plot[:,2],
                                            X_plot[:,3],
                                            X_plot[:,4],
                                            X_plot[:,5],
                                            X_plot[:,6],
                                            X_plot[:,7],
                                            X_plot[:,8],
                                            X_plot[:,9],)
# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.2  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [D,u_gt,v_gt,w_gt,p_gt,k]  # Use the scalar field for coloring
variable_names = ['D','u','v','w','p','k']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 14))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %%
X_init_clean=np.hstack((t[:,None],x[:,None],y[:,None],z[:,None],p_gt[:,None]))
X_init_clean = X_init_clean[~np.isnan(X_init_clean).any(axis=1)]
X_permeability=np.hstack((t[:,None],x[:,None],y[:,None],z[:,None],D[:,None],k[:,None]))
X_permeability = X_permeability[~np.isnan(X_permeability).any(axis=1)]

X_test2=np.hstack((t[:,None],x[:,None],y[:,None],z[:,None],u_gt[:,None],v_gt[:,None],w_gt[:,None],p_gt[:,None]))
X_test2 = X_test2[~np.isnan(X_test2).any(axis=1)]


# %%
Data=txyzc
all_times=np.unique(Data[:,0:1])
X_nnf=[]
for time_f in tqdm.tqdm(all_times):
    idx_time=np.argwhere(Data[:,0]==time_f).flatten()
    X_frame=Data[idx_time]
    X_nnf.append(X_frame)
X_nnf=np.array(X_nnf)

# %% [markdown]
# # Load Boundaries

# %%
df_bcs = pd.read_csv(outlet1_path)
X_out = df_bcs.to_numpy()
print(df_bcs.columns)
print(X_out.shape)
non_dim_fact_bcs=np.array(([[1/T_char,1/(1000*L_char),1/(1000*L_char),1/(1000*L_char),
                             1/(1000*L_char), 1/(1000*L_char),1/(1000*L_char),1/P_char
                             ,1/U_char,1/U_char,1/U_char]]))
print(non_dim_fact_bcs.shape)
df_bcs = pd.read_csv(out_BCs_path)
X_out = df_bcs.to_numpy()
print(X_out.shape)
X_out=np.hstack((X_out[:,0:1],X_out[:,:6],X_out[:,6:]))
X_out=X_out*non_dim_fact_bcs
df_bcs = pd.read_csv(outlet1_path)
X_o1 = df_bcs.to_numpy()
X_outlets=np.hstack((X_o1[:,0:1],X_o1[:,:6],X_o1[:,6:]))
X_outlets=X_outlets*non_dim_fact_bcs
df_bcs = pd.read_csv(outlet2_path)
X_o2 = df_bcs.to_numpy()
X_outlets2=np.hstack((X_o2[:,0:1],X_o2[:,:6],X_o2[:,6:]))
X_outlets2=X_outlets2*non_dim_fact_bcs
df_bcs = pd.read_csv(inlet1_path)
X_inlet = df_bcs.to_numpy()
X_inlet=np.hstack((X_inlet[:,0:1],X_inlet[:,:6],X_inlet[:,6:]))
X_inlet=X_inlet*non_dim_fact_bcs
X_in_out=np.vstack((X_inlet,X_outlets))
X_BCs=np.vstack((X_inlet,X_outlets,X_outlets2,X_out))
print(X_BCs.shape)

# %% [markdown]
# ## Remove the boundary information (Important)
# The PINN model learns using concentration gradient data. However, the numerical method explicitly defines zero normal concentration gradient at the boundaries. This creates an incompatibility, so it's crucial to discard the boundary data, since $\frac{\partial c}{\partial n}|_{boundary}=0$, in these regions.

# %%
indices_txyzc=remove_overlapping_points(X_frame,X_BCs,tolerance=1)
indices_xyzuvw=remove_overlapping_points(X_permeability,X_BCs,tolerance=1)
indices_xyzuvw_Crop=remove_overlapping_points(X_permeability,X_BCs,tolerance=1.5)

# %%
# Filter steady data
X_permeability=X_permeability[indices_xyzuvw]
X_test_crop=X_test2[indices_xyzuvw_Crop]
X_test2=X_test2[indices_xyzuvw]
# Filter unsteady data
X_nnf=X_nnf[:,indices_txyzc,:]
#Arrays
T_star=X_nnf[:,:,0]
X_star=X_nnf[:,:,1]
Y_star=X_nnf[:,:,2]
Z_star=X_nnf[:,:,3]
C_star=X_nnf[:,:,4]
Xnn=np.concatenate(X_nnf)
X_data_raw=Xnn
X_res=Xnn[:,0:4]
print(Pef)

# %%
frame=15
X_plot = X_nnf[frame]

t,x, y, z, c = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]
t2,x2, y2, z2, c2 = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]

# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [c]  # Use the scalar field for coloring
variable_names = ['c']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %%
X_plot = X_test2

t,x, y, z, u = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]
t2,x2, y2, z2, c2 = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]

# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [u]  # Use the scalar field for coloring
variable_names = ['u']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %%
X_plot = X_test_crop

t,x, y, z, u = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]
t2,x2, y2, z2, c2 = X_plot[:,0], X_plot[:,1], X_plot[:,2], X_plot[:,3], X_plot[:,4]

# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [u]  # Use the scalar field for coloring
variable_names = ['u']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Group Dataset
# 

# %%
avoid_from=5
avoid_from_org=np.copy(avoid_from)
X_c_H=[]
X_res_H=[]
C_max_H=[]
C_min_H=[]
C_mean_H=[]

X_c_L=[]
X_res_L=[]
C_max_L=[]
C_min_L=[]
C_mean_L=[]

diff_time=[]
all_frames=[]
tol=0.05
for it in tqdm.tqdm(range(avoid_from,len(X_nnf))):
    all_frames.append(it)

    X_frame_H=X_nnf[it]

    time_H=X_frame_H[0,0]

    C_max_H.append(np.max(X_frame_H[:,4:5]))
    C_min_H.append(np.min(X_frame_H[:,4:5]))
    C_mean_H.append(np.mean(X_frame_H[:,4:5]))


# Creating the figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(30, 4))

# Plotting Max Concentrations for High and Low
axs[0].plot(all_frames, C_max_H, label='Max Concentration High')
axs[0].set_yscale('log')
axs[0].set_title('Maximum Concentrations')
axs[1].set_xlabel('Time (min)')
axs[0].set_ylabel('Concentration')
axs[0].legend()

# Plotting Min Concentrations for High and Low
axs[1].plot(all_frames, C_min_H, label='Min Concentration High')
axs[1].set_yscale('log')
axs[1].set_title('Minimum Concentrations')
axs[1].set_xlabel('Time (min)')
axs[1].set_ylabel('Concentration')
axs[1].legend()

# Plotting Mean Concentrations for High and Low
axs[2].plot(all_frames, C_mean_H, label='Mean Concentration High')
axs[2].set_yscale('log')
axs[2].set_title('Mean Concentrations')
axs[1].set_xlabel('Time (min)')
axs[2].set_ylabel('Concentration')
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

min_H=np.min(C_min_H)
max_H=np.max(C_max_H)-min_H

max_all=max_H
print(max_all)
keep_c=max_H

X_c_H=[]
X_res_H=[]
C_max_H=[]
C_min_H=[]
C_mean_H=[]

X_c_L=[]
X_res_L=[]
C_max_L=[]
C_min_L=[]
C_mean_L=[]
X_nnf_bd=[]
all_frames=[]
for it in tqdm.tqdm(range(avoid_from,len(X_nnf))):
    all_frames.append(it)

    X_frame_H=X_nnf[it,:,:5]-min_H
    idx_keep0=np.argwhere(X_frame_H[:,4]<=keep_c).flatten()
    X_frame_H=X_frame_H[idx_keep0]*np.array([[1,1,1,1,100/keep_c]])

    C_max_H.append(np.max(X_frame_H[:,4:5]))
    C_min_H.append(np.min(X_frame_H[:,4:5]))
    C_mean_H.append(np.mean(X_frame_H[:,4:5]))

    idx_keep=np.argwhere(X_frame_H[:,4]>tol*np.mean(X_frame_H[:,4:5])).flatten()
    X_res_H.append(X_frame_H[idx_keep])
    X_c_H.append(X_frame_H)

    X_nnf_bd.append(X_frame_H)




# Creating the figure with 3 subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 4))

# Plotting Max Concentrations for High and Low
axs[0].plot(all_frames, C_max_H, label='Max Concentration High')
axs[0].set_yscale('log')
axs[0].set_title('Maximum Concentrations')
axs[1].set_xlabel('Time (min)')
axs[0].set_ylabel('Concentration')
axs[0].legend()

# Plotting Min Concentrations for High and Low
axs[1].plot(all_frames, C_min_H, label='Min Concentration High')
axs[1].set_yscale('log')
axs[1].set_title('Minimum Concentrations')
axs[1].set_xlabel('Time (min)')
axs[1].set_ylabel('Concentration')
axs[1].legend()

# Plotting Mean Concentrations for High and Low
axs[2].plot(all_frames, C_mean_H, label='Mean Concentration High')
axs[2].set_yscale('log')
axs[2].set_title('Mean Concentrations')
axs[1].set_xlabel('Time (min)')
axs[2].set_ylabel('Concentration')
axs[2].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()



# %%
Pe=Pef
def get_sytem_terms(params):
    # Neural Network
    fn={
        'c' :lambda x: pinn_fn(params['c'], x, M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'c_std' :lambda x: pinn_fn(params['c_std'], x, M1_all['c'], M2_all['c'],activation_fn,norm_fn),
    }
    #Pretrained Models
    #New Models
    c= lambda x: fn['c'](x)[0]
    c_std=lambda x: 10*jax.nn.sigmoid(fn['c_std'](x)[0])+sigma_0
   # derivatives
    def derivative_fn(f, x, idx):
        """Computes df/dx[idx] using forward mode AD."""
        perturbation = jnp.zeros_like(x)
        perturbation = perturbation.at[idx].set(1.0)
        primals, tangents = jvp(f, (x,), (perturbation,))
        return tangents


    c_t = lambda x: derivative_fn(c, x, 0)
    c_x = lambda x: derivative_fn(c, x, 1)
    c_y = lambda x: derivative_fn(c, x, 2)
    c_z = lambda x: derivative_fn(c, x, 3)
    c_xx = lambda x: derivative_fn(c_x, x, 1)
    c_yy = lambda x: derivative_fn(c_y, x, 2)
    c_zz = lambda x: derivative_fn(c_z, x, 3)

    # equations

    d_t = lambda x: c_t(x)- (1/Pe)*(c_xx(x) + c_yy(x) + c_zz(x))
    a_t = lambda x: c_x(x) 
    b_t = lambda x: c_y(x) 
    e_t = lambda x: c_z(x)  
    return d_t,a_t,b_t,e_t,c,c_std

    
def filter_lognormal_nan(data,key='a', k=3):
    data = np.array(data, dtype=np.float64)  # Ensure it's a NumPy array
    log_data = np.log(data+1e-12)  # Transform to log-space

    mu_log = np.nanmean(log_data)
    min_log = np.nanmin(log_data)
    sigma_log = np.nanstd(log_data)
    # Define filtering bounds
    if key=='c':
        lower_bound = 0.1
    else:
        lower_bound = mu_log - sigma_log

    upper_bound = mu_log + k * sigma_log

    # Identify outliers and replace with NaN
    if key=='c':
        mask = (data >= lower_bound) & (log_data <= upper_bound)
    else:
        mask = (log_data >= lower_bound) & (log_data <= upper_bound)
    
    filtered_data = np.where(mask, data, np.nan)
    return filtered_data

# =============================
# PDF & Plotting Functions
# =============================
def get_pdf(u_hist, bins=30):
    hist, bins = np.histogram(u_hist, bins=bins, density=True)
    return hist, bins

def get_colors_plot(cmap='Spectral', n_colors=5):
    n_colors = n_colors * 2
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    return colors[:n_colors // 4] + colors[3 * n_colors // 4:]

def plot_pdf_fields(all_pred, bins=100, time_img=0, figsize=(12, 12), x_lims=None, y_lims=None, mean_vals=None, save_path=None):
    keys = list(all_pred.keys())
    colors = get_colors_plot(n_colors=len(keys))
    
    fig, axs = plt.subplots(2, 3, figsize=figsize)
    axs = axs.flatten()
    
    for i, key in enumerate(keys):
        data = np.array(all_pred[key]).flatten()
        pdf, bins_arr = get_pdf(data[np.isfinite(data)], bins=bins)  # Ignore NaNs in PDF computation

        axs[i].plot(bins_arr[:-1], pdf, color=colors[i], label=f'{key}')
        axs[i].fill_between(bins_arr[:-1], pdf, color=colors[i], alpha=0.3)
        axs[i].set_xlabel(key)
        axs[i].set_ylabel('PDF')
        axs[i].set_yscale('log')  
        axs[i].set_xscale('log')  
        axs[i].set_title(f't={time_img:.2f} min')
        axs[i].legend(loc='upper left')

        # Add vertical dashed line for mean
        if mean_vals and key in mean_vals:
            axs[i].axvline(mean_vals[key], color='k', linestyle='dashed', linewidth=2, label='Mean')
            axs[i].legend(loc='upper left')

        # Apply consistent x-axis and y-axis limits
        if x_lims and key in x_lims:
            axs[i].set_xlim(x_lims[key])
        if y_lims:
            axs[i].set_ylim(y_lims)

    # Hide unused subplots if fewer than 6 fields
    for j in range(len(keys), len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


subdirs = ["General","Imgs", "Big_Data", "Data", "Flow_rate", "Errors",'System_video','System_video_abs','System_video_scaled','System_video_crop','PDF','PDF_filtered']
paths = create_and_return_directories(save_path, dataset_name, subdirs)

X_res_H = []
d_t, a_t, b_t, e_t, c_fx, c_std_fx = get_sytem_terms(params)

max_vals = {key: [] for key in ['a', 'b', 'c', 'd', 'e']}
min_vals = {key: [] for key in ['a', 'b', 'c', 'd', 'e']}
mean_vals = {key: [] for key in ['a', 'b', 'c', 'd', 'e']}
mean_vals_log = {key: [] for key in ['a', 'b', 'c', 'd', 'e']}

percentage_discarded = []
overall_max_list = []
keys_system = ['a', 'b','c', 'd', 'e']

thresh_keep = 0.005
plot_all_pdf = False
start = 0
stop = len(X_nnf)

# Storage for axis limits (determined in first snapshot)
x_lims = {}
y_lims = None

for it in tqdm.tqdm(range(avoid_from+start, stop)):
    all_frames.append(it)
    t__frame_plot = it
    snap = np.array([it])

    X_frame_H = X_nnf[it]

    t_test = X_frame_H[:, 0:1]
    x_test = X_frame_H[:, 1:2]
    y_test = X_frame_H[:, 2:3]
    z_test = X_frame_H[:, 3:4]
    c_test = X_frame_H[:, 4:5]

    X_test = np.hstack((t_test, x_test, y_test, z_test))
    
    # Compute raw (unfiltered) predictions
    all_pred_raw = {
        'a': np.abs(vmap(a_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'b': np.abs(vmap(b_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'c': np.abs(vmap(c_fx, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'd': np.abs(vmap(d_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'e': np.abs(vmap(e_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
    }

    # Apply lognormal filtering
    all_pred_filtered = {key: filter_lognormal_nan(all_pred_raw[key],key=key) for key in all_pred_raw}

    # Compute min, max, mean values for each field
    frame_mean_vals = {}
    frame_mean_vals_filter={}
    for key in keys_system:
        max_vals[key].append(jax.device_get(np.nanmax(all_pred_raw[key])).item())
        min_vals[key].append(jax.device_get(np.nanmin(all_pred_raw[key])).item())
        mean_vals[key].append(jax.device_get(np.nanmean(all_pred_raw[key])).item())
        mean_vals_log[key].append(jax.device_get(np.nanmean(all_pred_filtered[key])).item())
        frame_mean_vals[key] = mean_vals[key][-1] 
        frame_mean_vals_filter[key]={mean_vals_log[key][-1]}
    nan_masks = {key: np.isnan(all_pred_filtered[key]) for key in all_pred_filtered}    
    valid_mask = ~np.any(np.stack(list(nan_masks.values())), axis=0)
    intersection_indices_keep = np.where(valid_mask)[0]
    percentage_discarded.append(1-len(intersection_indices_keep)/len(X_frame_H))

    if plot_all_pdf:
        # Set consistent x-axis and y-axis limits based on the first snapshot
        if it == start:
            x_lims = {key: (np.nanmin([1e-7,1]), np.nanmax(all_pred_raw[key])) for key in all_pred_raw}
            y_lims = (1e-6, 1) 
        original_pdf_path = paths['PDF'] + f'{it / len(X_nnf):.3f}.png'
        plot_pdf_fields(all_pred_raw, bins=100, time_img=(T_char / 60) * t_test[0].item(), figsize=(12, 8), 
                        x_lims=x_lims, y_lims=y_lims, mean_vals=frame_mean_vals, save_path=original_pdf_path)

        # Save filtered PDF
        filtered_pdf_path = paths['PDF_filtered'] + f'{it / len(X_nnf):.3f}.png'
        plot_pdf_fields(all_pred_filtered, bins=100, time_img=(T_char / 60) * t_test[0].item(), figsize=(12, 8), 
                        x_lims=x_lims, y_lims=y_lims, mean_vals=frame_mean_vals, save_path=filtered_pdf_path)
if plot_all_pdf:
    video_name='Abs_vals_system_3D.mp4'
    make_video(paths['PDF'], video_name, 10) #
    video_name='Crop_vals_system_3D.mp4'
    make_video(paths['PDF_filtered'], video_name, 10) #


plt.plot(percentage_discarded)
plt.show()
# Extract data for plotting
components = ['a','b','d','e','c']
titles = [r'$c_x$', r'$c_y$', r'$c_z$', r'$c_t-(1/Pe)\nabla^2 c$',r'$c$']

metrics = ['max', 'min', 'mean']
metric_values = {'max': max_vals, 'min': min_vals, 'mean': mean_vals}

# Create a 5-row by 3-column plot (for all components)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 10), sharex=True)

for i, comp in enumerate(components): 
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        ax.plot(metric_values[metric][comp], linestyle='-')
        if j ==0:
            ax.set_ylabel(f'{titles[i]}')
        if i ==0:
            ax.set_title(f'{metric}')
        if i == len(components) - 1:
            ax.set_xlabel('t (min)')


plt.tight_layout()
plt.savefig(images_path+'original_dist.png')
plt.show()



X_res_H=[]

d_t,a_t,b_t,e_t,c_fx,c_std_fx=get_sytem_terms(params)
max_vals={
    'a':[],
    'b':[],
    'c':[],
    'd':[],
    'e':[],
}


min_vals={
    'a':[],
    'b':[],
    'c':[],
    'd':[],
    'e':[],
}


mean_vals={
    'a':[],
    'b':[],
    'c':[],
    'd':[],
    'e':[],
}
percentage_discarded=[]
overall_max_list=[]
keys_system=['a','b','c','d','e']
thresh_keep=0.005
plot_all=False
start=0
stop=avoid_from+start+1
stop=len(X_nnf)
for it in tqdm.tqdm(range(avoid_from+start,stop)):
    all_frames.append(it)
    t__frame_plot=it
    snap = np.array([it])

    X_frame_H=X_nnf[it]

    t_test=X_frame_H[:,0:1]
    x_test=X_frame_H[:,1:2]
    y_test=X_frame_H[:,2:3]
    z_test=X_frame_H[:,3:4]
    c_test=X_frame_H[:,4:5]

    X_test=np.hstack((t_test,x_test,y_test,z_test))
    all_pred_raw = {
        'a': np.abs(vmap(a_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'b': np.abs(vmap(b_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'c': np.abs(vmap(c_fx, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'd': np.abs(vmap(d_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
        'e': np.abs(vmap(e_t, (0))(X_test[:, 0:layers_1[0]])[:, None]),
    }
    # Apply lognormal filtering
    all_pred_filtered = {key: filter_lognormal_nan(all_pred_raw[key],key=key) for key in all_pred_raw}
    # Extract valid indices
    nan_masks = {key: np.isnan(all_pred_filtered[key].flatten()) for key in all_pred_filtered}
    score = 0
    for key in nan_masks:
        score = score + nan_masks[key].flatten()
    intersection_indices_keep = np.argwhere(score == 0).flatten()
    idx_delete={}
    idx_keep={}
    max_at_it={}
    mean_at_it={}
    all_pred_org={}
    all_pred={}
    for key in keys_system:
        all_pred_org[key]=jax.device_get(all_pred_filtered[key]).copy()
        all_pred[key] = jax.device_get(all_pred_filtered[key]).copy()
        if key=='c':
            max_at_c=jax.device_get(np.nanmax(all_pred[key])).item()
            all_pred[key] = all_pred[key]/max_at_c
        else:
            max_at_it[key]=jax.device_get(np.nanmax(all_pred[key])).item()
            all_pred[key] = all_pred[key]/max_at_it[key]
        mean_at_it[key]=jax.device_get(np.nanmean(all_pred[key])).item()

    # Extract valid indices
    nan_masks = {key: np.isnan(all_pred[key].flatten()) for key in all_pred}
    score = 0
    for key in nan_masks:
        score = score + nan_masks[key].flatten()
    intersection_indices_keep = np.argwhere(score == 0).flatten()
    intersection_indices = np.argwhere(score > 0).flatten()
    overal_max=min(max_at_it.values())
    overall_max_list.append(overal_max)
    for key in keys_system:
        all_pred[key] = jax.device_get(all_pred_org[key])/overal_max
        all_pred[key].flat[intersection_indices] = np.nan
        max_vals[key].append(jax.device_get(np.nanmax(all_pred[key])).item())
        min_vals[key].append(jax.device_get(np.nanmin(all_pred[key])).item())
        mean_vals[key].append(jax.device_get(np.nanmean(all_pred[key])).item())
    
    X_res_subsample=X_frame_H[intersection_indices_keep]
    X_res_H.append(np.hstack((X_res_subsample[:,:4],
                              X_res_subsample[:,0:1]*0+overal_max)))
    percentage_discarded.append(1-len(X_res_subsample[:,:4])/len(X_frame_H))
    if plot_all:
        a_pred  = all_pred['a']
        b_pred  = all_pred['b']
        c_pred  = all_pred_filtered['c']
        e_pred  = all_pred['e']
        d_pred  = all_pred['d']
        c_std_pred  = vmap(c_std_fx, (0))(X_test[:,0:layers_1[0]])[:,None]

        uvwpc_pred   = np.hstack((d_pred,a_pred,b_pred,e_pred,c_pred,c_std_pred))
        d_pred, a_pred, b_pred, e_pred, c_pred,c_std_pred = uvwpc_pred[:, 0:1], uvwpc_pred[:, 1:2], uvwpc_pred[:, 2:3], uvwpc_pred[:, 3:4] , uvwpc_pred[:, 4:5], uvwpc_pred[:, 5:6]
        data_frame = np.hstack((X_test, d_pred, a_pred, b_pred, e_pred, c_pred,c_std_pred))

        idx_plot = np.random.choice(len(y_test), int(len(y_test)), replace=False)
        t_test, x_test, y_test, z_test, d_pred, a_pred, b_pred, e_pred, c_pred,c_std_pred = [np.copy(data_frame[idx_plot, i:i+1]) for i in range(10)]
        c_test = np.copy(c_test[idx_plot])/C_char

        x_midplane = np.median(x_test)
        y_midplane = np.median(y_test)
        z_midplane = np.median(z_test)

        tolerance = 0.5
        x_midplane_filter = np.abs(x_test - x_midplane) <= tolerance
        y_midplane_filter = np.abs(y_test - y_midplane) <= tolerance
        z_midplane_filter = np.abs(z_test - z_midplane) <= tolerance
        variables = [c_test, c_pred,c_std_pred, d_pred, a_pred, b_pred, e_pred]
        variable_names = [r'$\hat{c}$', r'$c$',r'$\sigma_c$' ,r'$c_t-(1/Pe)\nabla^2 c$', r'$c_x$', r'$c_y$', r'$c_z$']
        name_image = System_video_scaled_path + f'time:{t__frame_plot / len(X_nnf):.4f}.png'

        fig = plt.figure(figsize=(15, 2 * len(variables)))

        for i, var in enumerate(variables):
            for j in range(3):
                ax = fig.add_subplot(len(variables), 3, i * 3 + j + 1)
                if j == 0:
                    filter = x_midplane_filter
                    x, y = z_test, y_test
                    xlabel, ylabel = 'Y', 'Z'
                elif j == 1:
                    filter = y_midplane_filter
                    x, y = x_test, z_test
                    xlabel, ylabel = 'X', 'Z'
                else:
                    filter = z_midplane_filter
                    x, y = x_test, y_test
                    xlabel, ylabel = 'X', 'Y'

                #vmin, vmax = global_min[i], global_max[i]  # Use the global min/max values for each variable
                scatter = ax.scatter(x[filter], y[filter], c=var[filter], cmap=cmap, s=5)#], vmin=vmin, vmax=vmax)
                ax.set_title(f'{variable_names[i]} at t={t__frame_plot}')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.axis('equal')
                ax.axis('off')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(scatter, cax=cax)

        plt.tight_layout()
        plt.savefig(name_image)
        if stop>avoid_from+start+1:
            plt.close()  
        else:
            plt.show()
if plot_all:
    video_name=dataset_name+'3D.mp4'
    make_video(System_video_scaled_path, video_name, 10) # 10 is the FPS (frames per second)

plt.plot(percentage_discarded)
plt.xlabel('t')
plt.ylabel('%Discarded data')
plt.show()
# Extract data for plotting
components = ['a','b','d','e','c']
titles = [r'$c_x$', r'$c_y$', r'$c_z$', r'$c_t-(1/Pe)\nabla^2 c$',r'$c$']

metrics = ['max', 'min', 'mean']
metric_values = {'max': max_vals, 'min': min_vals, 'mean': mean_vals}

# Create a 5-row by 3-column plot (for all components)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14, 10), sharex=True)

for i, comp in enumerate(components): 
    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        ax.plot(metric_values[metric][comp], linestyle='-')
        if j ==0:
            ax.set_ylabel(f'{titles[i]}')
        if i ==0:
            ax.set_title(f'{metric}')
        if i == len(components) - 1:
            ax.set_xlabel('t (min)')


plt.tight_layout()
plt.savefig(images_path+'New_dist.png')
plt.show()


# %%
frame=0
X_plot = np.vstack((X_c_H[frame]))

t,x, y, z, c = [X_plot[:, i] for i in range(5)]
t2,x2, y2, z2 = [X_out[:, i] for i in range(4)]

# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5 # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

x_midplane_filter2 = np.abs(x2 - x_midplane) <= tolerance
y_midplane_filter2 = np.abs(y2 - y_midplane) <= tolerance
z_midplane_filter2 = np.abs(z2 - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [c]  # Use the scalar field for coloring
variable_names = ['c']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            filter_mask2 = x_midplane_filter2
            scatter_x2, scatter_y2 = y2[filter_mask2], z2[filter_mask2]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            filter_mask2 = y_midplane_filter2
            scatter_x2, scatter_y2 = x2[filter_mask2], z2[filter_mask2]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            filter_mask2 = z_midplane_filter2
            scatter_x2, scatter_y2 = x2[filter_mask2], y2[filter_mask2]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x2, scatter_y2,c='green', s=0.1)
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=5)
    
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]} at t={frame}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %%
frame=0
X_plot = np.vstack((X_res_H[frame]))

t,x, y, z, c = [X_plot[:, i] for i in range(5)]
t2,x2, y2, z2 = [X_out[:, i] for i in range(4)]

# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5 # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

x_midplane_filter2 = np.abs(x2 - x_midplane) <= tolerance
y_midplane_filter2 = np.abs(y2 - y_midplane) <= tolerance
z_midplane_filter2 = np.abs(z2 - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [c]  # Use the scalar field for coloring
variable_names = ['c']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            filter_mask2 = x_midplane_filter2
            scatter_x2, scatter_y2 = y2[filter_mask2], z2[filter_mask2]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            filter_mask2 = y_midplane_filter2
            scatter_x2, scatter_y2 = x2[filter_mask2], z2[filter_mask2]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            filter_mask2 = z_midplane_filter2
            scatter_x2, scatter_y2 = x2[filter_mask2], y2[filter_mask2]
            xlabel, ylabel = 'X', 'Y'

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x2, scatter_y2,c='green', s=0.1)
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=5)
        ax.set_title(f'{variable_names[i]} Midplane {["X", "Y", "Z"][j]} at t={frame}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axis('equal')
        #ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.show()


# %%
frames=X_c_H
avoid_from=0
if train_ratio<=0.5:
    skip_frames = int(1 / train_ratio)
    idx_all_frames = np.arange(avoid_from, len(frames))
    idx_Train = np.arange(avoid_from+1, len(frames) - 1, skip_frames)
    idx_Train = np.append([avoid_from], idx_Train)  # Add the first frame
    idx_Train = np.append(idx_Train, [len(frames) - 1])  # Add the last frame
    idx_Test = np.setdiff1d(idx_all_frames, idx_Train)
    print(f'Skipping:{skip_frames}')
    print(f'Training on frames:{idx_Train}')
    print(f'Testing on frames:{idx_Test}')
    print(f'Training on {len(idx_Train)/len(frames)}')
else:
    idx_all_frames =np.arange(avoid_from,len(frames))
    np.random.seed(seed_np)
    idx_Train=np.random.choice(len(frames)-avoid_from,int((len(frames)-avoid_from)*train_ratio),replace=False)+avoid_from
    idx_Train = np.append([avoid_from], idx_Train)  # Add the first frame
    idx_Train = np.append(idx_Train, [len(frames) - 1])  # Add the last frame
    idx_Train = np.unique(idx_Train)
    idx_Test =np.setdiff1d(idx_all_frames,idx_Train)
    print(f'There are not enough frames to skip')
    print(f'Training on frames:{np.sort(idx_Train)}')
    print(f'Testing on frames:{idx_Test}')
    print(f'Training on {len(idx_Train)/len(frames)}')
X_train=np.zeros((1,5))
for idx in tqdm.tqdm(idx_Train):
    X_train=np.vstack((X_train,X_c_H[idx]))
X_train_H=np.copy(X_train[1:])
X_test=np.zeros((1,5))
for idx in tqdm.tqdm(idx_Test):
    X_test=np.vstack((X_test,X_c_H[idx]))
X_test_H=np.copy(X_test[1:])

# %%
X_res_H=np.vstack(X_res_H)

X_train_H=np.vstack(X_train_H)
X_test_H=np.vstack(X_test_H)

num_points=150000
if len(X_test_H)<num_points:
    X_GT_H=np.copy(X_test_H)
else:
    np.random.seed(seed_np)   
    idx_Gt=np.random.choice(len(X_test_H),num_points,replace=False)
    X_GT_H=X_test_H[idx_Gt]

# Printing the shapes of the datasets for verification
print(f"Training set shape high: {X_train_H.shape}")
print(f"Test set shape high: {X_train_H.shape}")
print(f"Test Eval set shape high: {X_GT_H.shape}")
print(f"Residual set shape high: {X_res_H.shape}")


# %%
N_r=1000000
N_d=1000000
np.random.seed(seed_np)
idx_res=np.random.choice(len(X_res_H),N_r,replace=False)
np.random.seed(seed_np)
idx_data=np.random.choice(len(X_train_H),N_d,replace=False)
X_res=X_res_H[idx_res]
X_train=X_train_H[idx_data]
X_GT=X_GT_H


# %%
def get_data_between(X_res_H,t_0):
    t_f=t_0+delta_t_2
    idx_0=np.argwhere(X_res_H[:,0]>t_0).flatten()
    idx_f=np.argwhere(X_res_H[:,0]<t_f).flatten()
    idx_keep=np.intersect1d(idx_0,idx_f)
    return X_res_H[idx_keep]

# %%
all_times_res=np.sort(np.unique(X_res[:,0]))
t_max=np.max(all_times_res)
t_min=np.min(all_times_res)
n_steps=3
delta_t=(t_max-t_min)/(n_steps)
delta_t_2=0.8*delta_t
print(delta_t,delta_t_2)

X_res_1=get_data_between(X_res_H,t_min)
X_res_2=get_data_between(X_res_H,t_min+delta_t)
X_res_3=get_data_between(X_res_H,t_min+2*delta_t)

# %% [markdown]
# ## Data Analytics

# %%
# %%
print(f'Bounds for Data Train with {X_train.shape} points:')
print(np.min(X_train,axis=0))
print(np.max(X_train,axis=0))
print(f'Bounds for front tracking Data with {X_init_clean.shape} points:')
print(np.min(X_init_clean,axis=0))
print(np.max(X_init_clean,axis=0))
print(f'Bounds for uvw Test with {X_test2.shape} points:')
print(np.min(X_test2,axis=0))
print(np.max(X_test2,axis=0))
print(f'Bounds for Data Eval with {X_GT.shape} points:')
print(np.min(X_GT,axis=0))
print(np.max(X_GT,axis=0))  
print(f'Bounds for Residual_1 Data with {X_res_1.shape} points:')
print(np.min(X_res_1,axis=0))
print(np.max(X_res_1,axis=0))
print(f'Bounds for Residual_2 Data with {X_res_2.shape} points:')
print(np.min(X_res_2,axis=0))
print(np.max(X_res_2,axis=0))
print(f'Bounds for Residual_3 Data with {X_res_3.shape} points:')
print(np.min(X_res_3,axis=0))
print(np.max(X_res_3,axis=0))
# %%
dataset = {
    'Data': X_train,
    'Test': X_test2 ,
    'Test_crop':X_test_crop,
    'Res_1': X_res_1,
    'Res_2': X_res_2,
    'Res_3': X_res_3,
    'GT': X_GT
}
lam = {
    'P': 1e-10,
    'Mass': 1e-16,
    'Data': 100,
}
neq = {
    'Data': 2,
    'Test': 1,
    'Test_crop': 1,
    'Res_1': 2,
    'Res_2': 2,
    'Res_3': 2,
    'GT': 1,
}


# %% [markdown]
# # RBA_Weights

# %%
lenghts=[]
for key in dataset.keys():
    lenghts.append(len(dataset[key]))
    print(f'Lenght of {key} dataset: {len(dataset[key])}')
N_train=int(np.max(lenghts))
if batch_size<0:
    batch_size=N_train
All_its=int(np.ceil(N_train/batch_size))
BATCH_SIZES = {}


for key in dataset.keys():
    if not(key == 'GT' or key == 'Test'):
        BATCH_SIZES[key] = int(np.ceil(len(dataset[key]) / All_its))
        if args.Equal_Batch:
            BATCH_SIZES[key]=batch_size
        print(f'Batch size for {key} dataset: {BATCH_SIZES[key]}')
    

lambdas={}
for key in dataset.keys():
    np.random.seed(seed_np)
    if not(key == 'GT' or key == 'Test'):
        lambdas[key] = np.random.uniform(0, 1, size=(len(dataset[key]), neq[key]))
        if init_zero:
            lambdas[key]= lambdas[key]*0+max_lambda/5
        print(f'Lambda shape for {key} dataset: {lambdas[key].shape}')


         



# %% [markdown]
# # Loss Functions

# %%

def Advec_Diff_Eq_full(params):
    # Neural Network
    fn={
        'P' :lambda x: pinn_fn(params['P'], jnp.array([0.0,x[1],x[2],x[3]]), M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'c' :lambda x: pinn_fn(params['c'], jnp.array([x[0],x[1],x[2],x[3]]), M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'c_std' :lambda x: pinn_fn(params['c_std'], jnp.array([x[0],x[1],x[2],x[3]]), M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'k' :lambda x: pinn_fn(params['k'], jnp.array([0.0,x[1],x[2],x[3]]), M1_all['k'], M2_all['k'],activation_fn,norm_fn),
        'D' :lambda x: pinn_fn(params['D'], jnp.array([0.0,x[1],x[2],x[3]]), M1_all['D'], M2_all['D'],activation_fn,norm_fn),
    }
    #Pretrained Models
    log_k= (
                 lambda x:-(lb_all['k']+ub_all['k']*jax.nn.sigmoid(fn['k'](x)[0])))
    k= (
                 lambda x: (jnp.exp(log_k(x)+off_set['k']))
                 )
    log_D= (
                 lambda x:-(lb_all['D']+ub_all['D']*jax.nn.sigmoid(fn['D'](x)[0]))
                 )
    D= (
                 lambda x: jnp.exp(log_D(x)+off_set['D'])
                 )

    #New Models
    p,c= (
                 lambda x: fn['P'](x)[0], 
                 lambda x: fn['c'](x)[0],
                 )
    scale_ad = lambda x: x[4]
   # derivatives
    def derivative_fn(f, x, idx):
        """Computes df/dx[idx] using forward mode AD."""
        perturbation = jnp.zeros_like(x)
        perturbation = perturbation.at[idx].set(1.0)
        primals, tangents = jvp(f, (x,), (perturbation,))
        return tangents

    p_x = lambda x: derivative_fn(p, x, 1)
    p_y = lambda x: derivative_fn(p, x, 2)
    p_z = lambda x: derivative_fn(p, x, 3)

    u = lambda x:-p_scale*k(x)*p_x(x)
    v = lambda x:-p_scale*k(x)*p_y(x)
    w = lambda x:-p_scale*k(x)*p_z(x)

    u_x = lambda x: derivative_fn(u, x, 1)
    v_y = lambda x: derivative_fn(v, x, 2)
    w_z = lambda x: derivative_fn(w, x, 3)

    c_t = lambda x: derivative_fn(c, x, 0)
    c_x = lambda x: derivative_fn(c, x, 1)
    c_y = lambda x: derivative_fn(c, x, 2)
    c_z = lambda x: derivative_fn(c, x, 3)
    c_xx = lambda x: derivative_fn(c_x, x, 1)
    c_yy = lambda x: derivative_fn(c_y, x, 2)
    c_zz = lambda x: derivative_fn(c_z, x, 3)

    D_x = lambda x: derivative_fn(D, x, 1)
    D_y = lambda x: derivative_fn(D, x, 2)
    D_z = lambda x: derivative_fn(D, x, 3)

     # equations
    eq0= (
        lambda x:(c_t(x) 
        + (u(x) * c_x(x) + v(x) * c_y(x) + w(x) * c_z(x)) 
        - (1/Pe)*(c_xx(x) + c_yy(x) + c_zz(x)))/scale_ad(x))

    eq6 = lambda x: (u_x(x) + v_y(x) + w_z(x))

    
    return eq0, eq6


# %%
sigma_0=0.01
k_loss=jnp.log(sigma_0**2)/2
def NLL(r,sigma,weight=1):# r is the residual r=(y-mu), sigma is the sandard deviation
    nnl=jnp.log(sigma**2)/2+(r)**2/(2*sigma**2)-k_loss
    return jnp.mean(weight*nnl)

# %%
def u_PINN(params):
    # Neural Network
    fn={
        'P' :lambda x: pinn_fn(params['P'], jnp.array([0.0,x[1],x[2],x[3]]),M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'c' :lambda x: pinn_fn(params['c'], jnp.array([x[0],x[1],x[2],x[3]]), M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'c_std' :lambda x: pinn_fn(params['c_std'], jnp.array([x[0],x[1],x[2],x[3]]), M1_all['c'], M2_all['c'],activation_fn,norm_fn),
        'k' :lambda x: pinn_fn(params['k'], jnp.array([0.0,x[1],x[2],x[3]]), M1_all['k'], M2_all['k'],activation_fn,norm_fn),
        'D' :lambda x: pinn_fn(params['D'], jnp.array([0.0,x[1],x[2],x[3]]), M1_all['D'], M2_all['D'],activation_fn,norm_fn),
    }
    #Pretrained Models
    log_k= (
                 lambda x:-(lb_all['k']+ub_all['k']*jax.nn.sigmoid(fn['k'](x)[0])))
    k= (
                 lambda x: (jnp.exp(log_k(x)+off_set['k']))
                 )
    log_D= (
                 lambda x:-(lb_all['D']+ub_all['D']*jax.nn.sigmoid(fn['D'](x)[0]))
                 )
    D= (
                 lambda x: jnp.exp(log_D(x)+off_set['D'])
                 )
    #New Models
    p,c,c_std= (
                 lambda x: fn['P'](x)[0], 
                 lambda x: fn['c'](x)[0],
                 lambda x: 10*jax.nn.sigmoid(fn['c_std'](x)[0])+sigma_0,
                 )
   # derivatives
    def derivative_fn(f, x, idx):
        """Computes df/dx[idx] using forward mode AD."""
        perturbation = jnp.zeros_like(x)
        perturbation = perturbation.at[idx].set(1.0)
        primals, tangents = jvp(f, (x,), (perturbation,))
        return tangents

    p_x = lambda x: derivative_fn(p, x, 1)
    p_y = lambda x: derivative_fn(p, x, 2)
    p_z = lambda x: derivative_fn(p, x, 3)

    u = lambda x:-p_scale*k(x)*p_x(x)
    v = lambda x:-p_scale*k(x)*p_y(x)
    w = lambda x:-p_scale*k(x)*p_z(x)
    return u,v,w,p,c,D,k,c_std

u_model=u_PINN


# %% [markdown]
# ## Training

# %%
def compute_prediction(X, funcs):
    result = jnp.zeros((X.shape[0], len(funcs)))
    for i, fn in enumerate(funcs):
        result = result.at[:, i].set(vmap(fn, (0))(X[:, :layers_ref[0]]).flatten())
    return result

# %%
key='Permeability'
Dataset2={}
Dataset2['Permeability']=X_permeability
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]*k_char
print('Bounds',K_pred.max(),K_pred.min())
max_log_10=np.max((K_pred))
K_pred=K_pred#/(max_log_10)
K_real  =Dataset2[key][:,5:6]*k_char
max_log_10=np.max((K_real))
K_real=K_real#/max_log_10
K_SI=K_pred*k_char
RL2_K = relative_error2(K_pred, K_real)
print(f'Relative L2 errors for K:{RL2_K:.4e}')
wasserstein_dist = wasserstein_distance(K_real.flatten(),K_pred.flatten())
print(f"Wasserstein distance between K_real and K_pred: {wasserstein_dist}")
wasserstein_dist_log10 = wasserstein_distance(np.log10(K_real).flatten(),np.log10(K_pred).flatten())
print(f"Wasserstein distance between K_real and K_pred: {wasserstein_dist_log10}")


# %%
mpl.rcParams['font.size'] = 20

t,x, y, z= Dataset2[key][:,0], Dataset2[key][:,1], Dataset2[key][:,2], Dataset2[key][:,3]
# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.3  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [np.log10(K_pred),np.log10(K_real),np.log10(np.abs(K_pred-K_real))/np.log10(K_real)]  # Use the scalar field for coloring
variable_names = ['K-Pred','K-Real','Relative Log Error','K-Real','K-Pred','K-Error']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y' 

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} -{["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.savefig(images_path+'Final_results.png')
plt.show()


# %%
key='Permeability'
Dataset2={}
Dataset2['Permeability']=X_permeability
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]*k_char
max_log_10=np.max((K_pred))
K_pred=K_pred/(max_log_10)
K_real  =Dataset2[key][:,5:6]*k_char
max_log_10=np.max((K_real))
K_real=K_real/max_log_10
K_SI=K_pred*k_char
RL2_K = relative_error2(K_pred, K_real)
max_log_10=np.min(np.log10((K_real)))
print(f'Relative L2 errors for K:{RL2_K:.4e}')
thresh=max_log_10/2
K_pred=np.array(K_pred)
idx_high=np.argwhere(K_pred>10**(thresh)).flatten()
print(len(idx_high))
idx_low=np.argwhere(K_pred<10**(thresh)).flatten()
print(len(idx_low))
#replace
K_pred[idx_high]=1
K_pred[idx_low]=0
idx_high=np.argwhere(K_real>10**(thresh)).flatten()
print(len(idx_high))
idx_low=np.argwhere(K_real<10**(thresh)).flatten()
print(len(idx_low))
#replace
K_real[idx_high]=1
K_real[idx_low]=0
RL2_K = relative_error2(K_pred, K_real)
print(f'Relative L2 errors for K:{RL2_K:.4e}')

# %%
mpl.rcParams['font.size'] = 20

t,x, y, z= Dataset2[key][:,0], Dataset2[key][:,1], Dataset2[key][:,2], Dataset2[key][:,3]
# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [K_pred,K_real,np.log10(np.abs(K_pred-K_real))]  # Use the scalar field for coloring
variable_names = ['K-Pred','K-Real','Relative Log Error','K-Real','K-Pred','K-Error']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y' 

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} -{["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.savefig(images_path+'Final_results.png')
plt.show()


# %%
K_pred = np.array(K_pred)
K_real = np.array(K_real)

# Compute the intersection and union
intersection = np.sum((K_pred == 1) & (K_real == 1))
union = np.sum((K_pred == 1) | (K_real == 1))

# Compute IoU
iou = intersection / union if union != 0 else 0

print(f'Intersection over Union (IoU) for K: {iou:.4f}')


# %%
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
u_pred  = vmap(u_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
v_pred  = vmap(v_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
w_pred  = vmap(w_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
p_pred  = vmap(p_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
c_pred  = vmap(c_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

RL2_c = relative_error2(c_pred, dataset['GT'][:,4:5])
print(f'Relative L2 errors for c:{RL2_c:.4e}')
u_max, u_min = np.max(u_pred), np.min(u_pred)
v_max, v_min = np.max(v_pred), np.min(v_pred)
w_max, w_min = np.max(w_pred), np.min(w_pred)
p_max, p_min = np.max(p_pred), np.min(p_pred)
c_max, c_min = np.max(c_pred), np.min(c_pred)
vmag_max, vmag_min= np.max(velocity_magnitude), np.min(velocity_magnitude)

dataset2={}
dataset2['Test_crop']=X_test2


u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
u_pred  = vmap(u_fx, (0))(dataset2['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
v_pred  = vmap(v_fx, (0))(dataset2['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
w_pred  = vmap(w_fx, (0))(dataset2['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
p_pred  = vmap(p_fx, (0))(dataset2['Test_crop'][:,0:layers_1[0]])[:,None]*p_scale*P_char
p_pred  =p_pred-p_pred.min()
c_pred  = vmap(c_fx, (0))(dataset2['Test_crop'][:,0:layers_1[0]])[:,None]

velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

RL2_u = relative_error2(u_pred, dataset2['Test_crop'][:,4:5]*U_char)
RL2_v = relative_error2(v_pred, dataset2['Test_crop'][:,5:6]*U_char)
RL2_w = relative_error2(w_pred, dataset2['Test_crop'][:,6:7]*U_char)

u_max, u_min = np.max(u_pred), np.min(u_pred)
v_max, v_min = np.max(v_pred), np.min(v_pred)
w_max, w_min = np.max(w_pred), np.min(w_pred)
p_max, p_min = np.max(p_pred), np.min(p_pred)
c_max, c_min = np.max(c_pred), np.min(c_pred)
vmag_max, vmag_min= np.max(velocity_magnitude), np.min(velocity_magnitude)
x_test = dataset2['Test_crop'][:,1:2]
y_test = dataset2['Test_crop'][:,2:3]
z_test = dataset2['Test_crop'][:,3:4]
u_test = dataset2['Test_crop'][:,4:5]*U_char  
v_test = dataset2['Test_crop'][:,5:6]*U_char  
w_test = dataset2['Test_crop'][:,6:7]*U_char 
p_test = dataset2['Test_crop'][:,7:8]*P_char
p_test  =p_test-p_test.min()
RL2_p = relative_error2(p_pred, p_test)
print(f'Relative L2 errors for u:{RL2_u:.4e} v:{RL2_v:.4e} w:{RL2_w:.4e},p:{RL2_p:.4e}')

V_test = np.sqrt(u_test**2 + v_test**2 + w_test**2)
V_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)



mpl.rcParams['font.size'] = 20
size_point=20

# Calculate midplane values
x_mid = np.median(x_test)
y_mid = np.median(y_test)
z_mid = np.median(z_test)

# Define tolerance for midplane filtering
tolerance = 0.5

# Create filters for midplane slices
x_filter = np.abs(x_test - x_mid) <= tolerance
y_filter = np.abs(y_test - y_mid) <= tolerance
z_filter = np.abs(z_test - z_mid) <= tolerance

# Prepare the data for plotting
data_pairs = [
    (u_test, u_pred, 'u FD', 'u Pred'),
    (v_test, v_pred, 'v FD', 'v Pred'),
    (w_test, w_pred, 'w FD', 'w Pred'),
    (p_test, p_pred, 'p FD', 'p Pred'),
    (V_test, V_pred, 'V FD', 'V Pred')
]

# Define colormap
cmap = plt.get_cmap('plasma')

# Start plotting
fig, axes = plt.subplots(nrows=15, ncols=3, figsize=(20, 40))#, constrained_layout=True)  # Adjusted for additional rows for error plots

for i, (test_data, pred_data, test_label, pred_label) in enumerate(data_pairs):
    # Compute global vmin and vmax for each pair from the test data
    vmin = np.min(test_data)
    vmax = np.max(test_data)
    error_data = np.abs(test_data - pred_data)  # Calculate absolute error

    for j in range(3):
        if j == 0:  # X Midplane: Plot Y vs Z
            x, y = z_test, y_test
            xlabel, ylabel = 'Z', 'Y'
            filter = x_filter
        elif j == 1:  # Y Midplane: Plot X vs Z
            x, y = x_test, z_test
            xlabel, ylabel = 'X', 'Z'
            filter = y_filter
        else:  # Z Midplane: Plot X vs Y
            x, y = x_test, y_test
            xlabel, ylabel = 'X', 'Y'
            filter = z_filter

        # Index calculation for subplot
        ax_test = axes[i*3, j]
        ax_pred = axes[i*3+1, j]
        ax_error = axes[i*3+2, j]  # Additional axis for error

        # Plot test data
        sc_test = ax_test.scatter(x[filter], y[filter], c=test_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_test.set_title(f'{test_label} - {xlabel}-{ylabel}')
        ax_test.set_xlabel(xlabel)
        ax_test.set_ylabel(ylabel)
        ax_test.axis('equal')
        ax_test.axis('off')
        divider = make_axes_locatable(ax_test)
        cax_test = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_test, cax=cax_test)

        # Plot predicted data
        sc_pred = ax_pred.scatter(x[filter], y[filter], c=pred_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_pred.set_title(f'{pred_label} - {xlabel}-{ylabel}')
        ax_pred.set_xlabel(xlabel)
        ax_pred.set_ylabel(ylabel)
        ax_pred.axis('equal')
        ax_pred.axis('off')
        divider = make_axes_locatable(ax_pred)
        cax_pred = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_pred, cax=cax_pred)

        # Plot error data
        sc_error = ax_error.scatter(x[filter], y[filter], c=error_data[filter], cmap=cmap, vmin=0,s=size_point)#, vmax=vmax-vmin)
        ax_error.set_title(f'Error - {xlabel}-{ylabel}')
        ax_error.set_xlabel(xlabel)
        ax_error.set_ylabel(ylabel)
        ax_error.axis('equal')
        ax_error.axis('off')
        divider = make_axes_locatable(ax_error)
        cax_error = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_error, cax=cax_error)
plt.tight_layout()
plt.savefig(images_path+'Pre_results.png')
plt.show()

# Save or show the figure


# %%
@jit
def update_stage_2(params, lambdas,lam,Pe, opt_state, dataset: Dict[str, np.ndarray], batch_indices: Dict[str, np.ndarray]):
    # Compute the predictions based on the given functions.
    def compute_prediction(X, funcs):
        result = jnp.zeros((X.shape[0], len(funcs)))
        for i, fn in enumerate(funcs):
            result = result.at[:, i].set(vmap(fn, (0))(X).flatten())
        return result
    def get_residuals_and_multipliers(key, dataset, functions, lambdas, target, it_details=None):
        batch_indices,gamma,lr_lambdas_0,use_RBA=it_details
        pred = compute_prediction(dataset[key], functions)
        residuals = jnp.abs(pred - target)
        lam_key = (gamma) * lambdas[key][batch_indices[key]] + lr_lambdas_0 * residuals / (jnp.max(residuals, axis=0) + 10**(-8))
        lambdas[key] = lambdas[key].at[batch_indices[key]].set(lam_key)
        lam_key = lam_key * use_RBA + 0.1
        return residuals,lam_key,lambdas
    it_details=(batch_indices,gamma,lr_lambdas_0,use_RBA)
    def loss_fn(params,lambdas):
        # eqn error
        eq0, eq6= Advec_Diff_Eq_full(params)
        u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
        # Initialize dictionaries
        residuals={}
        multipliers={}
        # Equations
        residuals['Res_1'],multipliers['Res_1'],lambdas = get_residuals_and_multipliers('Res_1', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_2'],multipliers['Res_2'],lambdas = get_residuals_and_multipliers('Res_2', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_3'],multipliers['Res_3'],lambdas = get_residuals_and_multipliers('Res_3', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Data'],multipliers['Data'],lambdas  = get_residuals_and_multipliers('Data', dataset, #Dataset to compute predictions
                                                             [c_fx,c_std_fx], #Functions to evaluate
                                                             lambdas,  # local multiplieers
                                                             dataset['Data'][:,4:6],
                                                             it_details)

        #Calculate Stds
        sigma_c=compute_prediction(dataset['Data'], [c_std_fx])

        
        loss_vals = {
            'AD': lam['P']*(Error_fn(residuals['Res_1'][:, 0:1], 0.0, weight=multipliers['Res_1'][:, 0:1])
                           +Error_fn(residuals['Res_2'][:, 0:1], 0.0, weight=multipliers['Res_2'][:, 0:1])
                           +Error_fn(residuals['Res_3'][:, 0:1], 0.0, weight=multipliers['Res_3'][:, 0:1])),
            'Mass':use_mass*lam['Mass']*(Error_fn(residuals['Res_1'][:, 1:2], 0.0, weight=multipliers['Res_1'][:, 1:2])
                                        +Error_fn(residuals['Res_2'][:, 1:2], 0.0, weight=multipliers['Res_2'][:, 1:2])
                                        +Error_fn(residuals['Res_3'][:, 1:2], 0.0, weight=multipliers['Res_3'][:, 1:2])),
            'c_std' :  lam['Data']*NLL(residuals['Data'][:, 0:1], sigma_c, weight=multipliers['Data'][:, 1:2]),
        }
        Error_vals = {
            'AD':  (jnp.mean(residuals['Res_1'][:, 0:1])
                   +jnp.mean(residuals['Res_2'][:, 0:1])
                   +jnp.mean(residuals['Res_3'][:, 0:1])),
            'Mass':(jnp.mean(residuals['Res_1'][:, 1:2])
                   +jnp.mean(residuals['Res_2'][:, 1:2])
                   +jnp.mean(residuals['Res_3'][:, 1:2])),
            'c' :  jnp.mean(residuals['Data'][:, 0:1]),
        }

        # Total Loss
        total_loss = sum(loss_vals.values())

        return total_loss, (Error_vals, lambdas)

    # Compute gradients of the loss function with respect to parameters.
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, loss_dict_and_lambdas), grads = grad_fn(params,lambdas)
    losses_dict = loss_dict_and_lambdas[0]
    updated_lambdas = loss_dict_and_lambdas[1]

    # Update model parameters based on the calculated gradients.
    updates={}
    updates['P'], opt_state['P'] = optimizer['P'].update(grads['P'], opt_state['P'], params['P'])
    params['P'] = optax.apply_updates(params['P'], updates['P'])
    

    return params, updated_lambdas, opt_state, losses_dict


# %%
@jit
def update_stage_3(params, lambdas,lam,Pe, opt_state, dataset: Dict[str, np.ndarray], batch_indices: Dict[str, np.ndarray]):
    # Compute the predictions based on the given functions.
    def compute_prediction(X, funcs):
        result = jnp.zeros((X.shape[0], len(funcs)))
        for i, fn in enumerate(funcs):
            result = result.at[:, i].set(vmap(fn, (0))(X).flatten())
        return result
    def get_residuals_and_multipliers(key, dataset, functions, lambdas, target, it_details=None):
        batch_indices,gamma,lr_lambdas_0,use_RBA=it_details
        pred = compute_prediction(dataset[key], functions)
        residuals = jnp.abs(pred - target)
        lam_key = (gamma) * lambdas[key][batch_indices[key]] + lr_lambdas_0 * residuals / (jnp.max(residuals, axis=0) + 10**(-8))
        lambdas[key] = lambdas[key].at[batch_indices[key]].set(lam_key)
        lam_key = lam_key * use_RBA + 0.1
        return residuals,lam_key,lambdas
    it_details=(batch_indices,gamma,lr_lambdas_0,use_RBA)
    def loss_fn(params,lambdas):
        # eqn error
        eq0, eq6= Advec_Diff_Eq_full(params)
        u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
        # Initialize dictionaries
        residuals={}
        multipliers={}
        # Equations
        residuals['Res_1'],multipliers['Res_1'],lambdas = get_residuals_and_multipliers('Res_1', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_2'],multipliers['Res_2'],lambdas = get_residuals_and_multipliers('Res_2', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_3'],multipliers['Res_3'],lambdas = get_residuals_and_multipliers('Res_3', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Data'],multipliers['Data'],lambdas  = get_residuals_and_multipliers('Data', dataset, #Dataset to compute predictions
                                                             [c_fx,c_std_fx], #Functions to evaluate
                                                             lambdas,  # local multiplieers
                                                             dataset['Data'][:,4:6],
                                                             it_details)

        #Calculate Stds
        sigma_c=compute_prediction(dataset['Data'], [c_std_fx])

        
        loss_vals = {
            'AD': lam['P']*(Error_fn(residuals['Res_1'][:, 0:1], 0.0, weight=multipliers['Res_1'][:, 0:1])
                           +Error_fn(residuals['Res_2'][:, 0:1], 0.0, weight=multipliers['Res_2'][:, 0:1])
                           +Error_fn(residuals['Res_3'][:, 0:1], 0.0, weight=multipliers['Res_3'][:, 0:1])),
            'Mass':use_mass*lam['Mass']*(Error_fn(residuals['Res_1'][:, 1:2], 0.0, weight=multipliers['Res_1'][:, 1:2])
                                        +Error_fn(residuals['Res_2'][:, 1:2], 0.0, weight=multipliers['Res_2'][:, 1:2])
                                        +Error_fn(residuals['Res_3'][:, 1:2], 0.0, weight=multipliers['Res_3'][:, 1:2])),
            'c' :  lam['Data']*Error_fn(residuals['Data'][:, 0:1], 0.0, weight=multipliers['Data'][:, 0:1]),
            #'c_std' :  lam['Data']*NLL(residuals['Data'][:, 0:1], sigma_c, weight=multipliers['Data'][:, 1:2]),
        }
        Error_vals = {
            'AD':  (jnp.mean(residuals['Res_1'][:, 0:1])
                   +jnp.mean(residuals['Res_2'][:, 0:1])
                   +jnp.mean(residuals['Res_3'][:, 0:1])),
            'Mass':(jnp.mean(residuals['Res_1'][:, 1:2])
                   +jnp.mean(residuals['Res_2'][:, 1:2])
                   +jnp.mean(residuals['Res_3'][:, 1:2])),
            'c' :  jnp.mean(residuals['Data'][:, 0:1]),
        }

        # Total Loss
        total_loss = sum(loss_vals.values())

        return total_loss, (Error_vals, lambdas)

    # Compute gradients of the loss function with respect to parameters.
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, loss_dict_and_lambdas), grads = grad_fn(params,lambdas)
    losses_dict = loss_dict_and_lambdas[0]
    updated_lambdas = loss_dict_and_lambdas[1]

    # Update model parameters based on the calculated gradients.
    updates={}
    updates['P'], opt_state['P'] = optimizer['P'].update(grads['P'], opt_state['P'], params['P'])
    params['P'] = optax.apply_updates(params['P'], updates['P'])
    updates['k'], opt_state['k'] = optimizer['k'].update(grads['k'], opt_state['k'], params['k'])
    params['k'] = optax.apply_updates(params['k'], updates['k'])
    return params, updated_lambdas, opt_state, losses_dict


# %%
@jit
def update_stage_4(params, lambdas,lam,Pe, opt_state, dataset: Dict[str, np.ndarray], batch_indices: Dict[str, np.ndarray]):
    # Compute the predictions based on the given functions.
    def compute_prediction(X, funcs):
        result = jnp.zeros((X.shape[0], len(funcs)))
        for i, fn in enumerate(funcs):
            result = result.at[:, i].set(vmap(fn, (0))(X).flatten())
        return result
    def get_residuals_and_multipliers(key, dataset, functions, lambdas, target, it_details=None):
        batch_indices,gamma,lr_lambdas_0,use_RBA=it_details
        pred = compute_prediction(dataset[key], functions)
        residuals = jnp.abs(pred - target)
        lam_key = (gamma) * lambdas[key][batch_indices[key]] + lr_lambdas_0 * residuals / (jnp.max(residuals, axis=0) + 10**(-8))
        lambdas[key] = lambdas[key].at[batch_indices[key]].set(lam_key)
        lam_key = lam_key * use_RBA + 0.1
        return residuals,lam_key,lambdas
    it_details=(batch_indices,gamma,lr_lambdas_0,use_RBA)
    def loss_fn(params,lambdas):
        # eqn error
        eq0, eq6= Advec_Diff_Eq_full(params)
        u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
        # Initialize dictionaries
        residuals={}
        multipliers={}
        # Equations
        residuals['Res_1'],multipliers['Res_1'],lambdas = get_residuals_and_multipliers('Res_1', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_2'],multipliers['Res_2'],lambdas = get_residuals_and_multipliers('Res_2', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_3'],multipliers['Res_3'],lambdas = get_residuals_and_multipliers('Res_3', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Data'],multipliers['Data'],lambdas  = get_residuals_and_multipliers('Data', dataset, #Dataset to compute predictions
                                                             [c_fx,c_std_fx], #Functions to evaluate
                                                             lambdas,  # local multiplieers
                                                             dataset['Data'][:,4:6],
                                                             it_details)

        #Calculate Stds
        sigma_c=compute_prediction(dataset['Data'], [c_std_fx])

        
        loss_vals = {
            'AD': lam['P']*(Error_fn(residuals['Res_1'][:, 0:1], 0.0, weight=multipliers['Res_1'][:, 0:1])
                           +Error_fn(residuals['Res_2'][:, 0:1], 0.0, weight=multipliers['Res_2'][:, 0:1])
                           +Error_fn(residuals['Res_3'][:, 0:1], 0.0, weight=multipliers['Res_3'][:, 0:1])),
            'Mass':use_mass*lam['Mass']*(Error_fn(residuals['Res_1'][:, 1:2], 0.0, weight=multipliers['Res_1'][:, 1:2])
                                        +Error_fn(residuals['Res_2'][:, 1:2], 0.0, weight=multipliers['Res_2'][:, 1:2])
                                        +Error_fn(residuals['Res_3'][:, 1:2], 0.0, weight=multipliers['Res_3'][:, 1:2])),
            'c' :  lam['Data']*Error_fn(residuals['Data'][:, 0:1], 0.0, weight=multipliers['Data'][:, 0:1]),
            #'c_std' :  lam['Data']*NLL(residuals['Data'][:, 0:1], sigma_c, weight=multipliers['Data'][:, 1:2]),
        }
        Error_vals = {
            'AD':  (jnp.mean(residuals['Res_1'][:, 0:1])
                   +jnp.mean(residuals['Res_2'][:, 0:1])
                   +jnp.mean(residuals['Res_3'][:, 0:1])),
            'Mass':(jnp.mean(residuals['Res_1'][:, 1:2])
                   +jnp.mean(residuals['Res_2'][:, 1:2])
                   +jnp.mean(residuals['Res_3'][:, 1:2])),
            'c' :  jnp.mean(residuals['Data'][:, 0:1]),
        }

        # Total Loss
        total_loss = sum(loss_vals.values())

        return total_loss, (Error_vals, lambdas)

    # Compute gradients of the loss function with respect to parameters.
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, loss_dict_and_lambdas), grads = grad_fn(params,lambdas)
    losses_dict = loss_dict_and_lambdas[0]
    updated_lambdas = loss_dict_and_lambdas[1]

    # Update model parameters based on the calculated gradients.
    updates={}
    updates['P'], opt_state['P'] = optimizer['P'].update(grads['P'], opt_state['P'], params['P'])
    params['P'] = optax.apply_updates(params['P'], updates['P'])
    updates['c'], opt_state['c'] = optimizer['c'].update(grads['c'], opt_state['c'], params['c'])
    params['c'] = optax.apply_updates(params['c'], updates['c'])
    updates['k'], opt_state['k'] = optimizer['k'].update(grads['k'], opt_state['k'], params['k'])
    params['k'] = optax.apply_updates(params['k'], updates['k'])
    

    return params, updated_lambdas, opt_state, losses_dict


# %%
@jit
def update_stage_5(params, lambdas,lam,Pe, opt_state, dataset: Dict[str, np.ndarray], batch_indices: Dict[str, np.ndarray]):
    # Compute the predictions based on the given functions.
    def compute_prediction(X, funcs):
        result = jnp.zeros((X.shape[0], len(funcs)))
        for i, fn in enumerate(funcs):
            result = result.at[:, i].set(vmap(fn, (0))(X).flatten())
        return result
    def get_residuals_and_multipliers(key, dataset, functions, lambdas, target, it_details=None):
        batch_indices,gamma,lr_lambdas_0,use_RBA=it_details
        pred = compute_prediction(dataset[key], functions)
        residuals = jnp.abs(pred - target)
        lam_key = (gamma) * lambdas[key][batch_indices[key]] + lr_lambdas_0 * residuals / (jnp.max(residuals, axis=0) + 10**(-8))
        lambdas[key] = lambdas[key].at[batch_indices[key]].set(lam_key)
        lam_key = lam_key * use_RBA + 0.1
        return residuals,lam_key,lambdas
    it_details=(batch_indices,gamma,lr_lambdas_0,use_RBA)
    def loss_fn(params,lambdas):
        # eqn error
        eq0, eq6= Advec_Diff_Eq_full(params)
        u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
        # Initialize dictionaries
        residuals={}
        multipliers={}
        # Equations
        residuals['Res_1'],multipliers['Res_1'],lambdas = get_residuals_and_multipliers('Res_1', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_2'],multipliers['Res_2'],lambdas = get_residuals_and_multipliers('Res_2', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Res_3'],multipliers['Res_3'],lambdas = get_residuals_and_multipliers('Res_3', dataset, #Dataset to compute predictions
                                                             [eq0, eq6 ], #Functions to evaluate
                                                             lambdas,  # local multipliers(NOTICE IT IS LAMBDA_0)
                                                             0.0,
                                                             it_details) 
        residuals['Data'],multipliers['Data'],lambdas  = get_residuals_and_multipliers('Data', dataset, #Dataset to compute predictions
                                                             [c_fx,c_std_fx], #Functions to evaluate
                                                             lambdas,  # local multiplieers
                                                             dataset['Data'][:,4:6],
                                                             it_details)
        #Calculate Stds
        sigma_c=compute_prediction(dataset['Data'], [c_std_fx])

        
        loss_vals = {
            'AD': lam['P']*(MAE(residuals['Res_1'][:, 0:1], 0.0, weight=multipliers['Res_1'][:, 0:1])
                           +MAE(residuals['Res_2'][:, 0:1], 0.0, weight=multipliers['Res_2'][:, 0:1])
                           +MAE(residuals['Res_3'][:, 0:1], 0.0, weight=multipliers['Res_3'][:, 0:1])),
            'Mass':use_mass*lam['Mass']*(MAE(residuals['Res_1'][:, 1:2], 0.0, weight=multipliers['Res_1'][:, 1:2])
                                        +MAE(residuals['Res_2'][:, 1:2], 0.0, weight=multipliers['Res_2'][:, 1:2])
                                        +MAE(residuals['Res_3'][:, 1:2], 0.0, weight=multipliers['Res_3'][:, 1:2])),
            'c' :  lam['Data']*Error_fn(residuals['Data'][:, 0:1], 0.0, weight=multipliers['Data'][:, 0:1]),
            #'c_std' :  lam['Data']*NLL(residuals['Data'][:, 0:1], sigma_c, weight=multipliers['Data'][:, 1:2]),
        }
        Error_vals = {
            'AD':  (jnp.mean(residuals['Res_1'][:, 0:1])
                   +jnp.mean(residuals['Res_2'][:, 0:1])
                   +jnp.mean(residuals['Res_3'][:, 0:1])),
            'Mass':(jnp.mean(residuals['Res_1'][:, 1:2])
                   +jnp.mean(residuals['Res_2'][:, 1:2])
                   +jnp.mean(residuals['Res_3'][:, 1:2])),
            'c' :  jnp.mean(residuals['Data'][:, 0:1]),
        }

        # Total Loss
        total_loss = sum(loss_vals.values())

        return total_loss, (Error_vals, lambdas)

    # Compute gradients of the loss function with respect to parameters.
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (_, loss_dict_and_lambdas), grads = grad_fn(params,lambdas)
    losses_dict = loss_dict_and_lambdas[0]
    updated_lambdas = loss_dict_and_lambdas[1]

    # Update model parameters based on the calculated gradients.
    updates={}
    updates['P'], opt_state['P'] = optimizer['P'].update(grads['P'], opt_state['P'], params['P'])
    params['P'] = optax.apply_updates(params['P'], updates['P'])
    updates['c'], opt_state['c'] = optimizer['c'].update(grads['c'], opt_state['c'], params['c'])
    params['c'] = optax.apply_updates(params['c'], updates['c'])
    updates['k'], opt_state['k'] = optimizer['k'].update(grads['k'], opt_state['k'], params['k'])
    params['k'] = optax.apply_updates(params['k'], updates['k'])    

    return params, updated_lambdas, opt_state, losses_dict


# %%
# %%
if normalization.lower()=='fourier_time':
    M1=generate_random_matrix(embed_size//(2*input_dim),input_dim)
    M2=10
elif normalization.lower()=='fourier':
    M1=generate_random_matrix(embed_size//(2*input_dim),input_dim)
    M2=10
else:
    M1= Norm_metric1(dataset['Data'][:, :layers_ref[0]])
    M2= Norm_metric2(dataset['Data'][:, :layers_ref[0]])

# %% [markdown]
# # Training
# 

# %%
save_params_dict(params,result_path,dataset_name,type='Test')
def compute_error_metrics(params,key='GT'):
    u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
    # Further computations for errors if any
    c_pred  = vmap(c_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    error_c = relative_error2(c_pred, dataset[key][:,4:5])
    return error_c

def compute_error_metrics_init(params,key='Test_crop'):
    u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
    # Further computations for errors if any
    u_pred  = vmap(u_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    v_pred  = vmap(v_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    w_pred  = vmap(w_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    p_pred  = vmap(p_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    error_u = relative_error2(u_pred, dataset[key][:,4:5])
    error_v = relative_error2(v_pred, dataset[key][:,5:6])
    error_w = relative_error2(w_pred, dataset[key][:,6:7])
    error_p = relative_error2(p_pred-p_pred.mean(), dataset[key][:,6:7]-(dataset[key][:,6:7]).mean())

    return error_u, error_v, error_w,error_p

def compute_error_metrics_Inlet(params,key='Test'):
    u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
    # Further computations for errors if any
    p_pred  = vmap(p_fx, (0))(dataset[key][:,0:layers_ref[0]])[:,None]
    error_p = relative_error2(p_pred, dataset[key][:,4:5])
    return error_p
def compute_error_permeability(params):
    key='Permeability'
    u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
    # Further computations for errors if any
    K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]
    K_real  =Dataset2[key][:,5:6]
    K_SI=K_pred*k_char
    RL2_K = relative_error2(K_pred*k_char, K_real*k_char)
    return RL2_K

# %%
initialization_step=1*num_gsteps//8 #Learn c
activate_physics=1*num_gsteps//8  #Learn c with std
full_model=4*num_gsteps//8  #Learn P+c
refine_it=7*num_gsteps//8 #Learn P+c L1
alpha_list=[]
beta_list=[]
lam = {
    'P': 10,
    'Mass': 0.01,
    'Data': 10,
}
if Run_MODE=='Train':
    epoch_losses = []
    log_loss     = []
    log_epoch    = []
    log_lambdas  = []
    log_error_u  = []
    log_error_v  = []
    start_time = time.time()
    running_time = 0
    it = 0
    it_opt=0
    Pe=Pef
    update=update_stage_2
    while it <=num_gsteps:
        #Sample Points
        batch_indices=sample_points_PDF(it,BATCH_SIZES,dataset,lambdas,k=args.k_samp,c=args.c_samp)

        batch_data = {key: dataset[key][batch_indices[key]] for key in BATCH_SIZES}
        
        #Update parameters
        params,lambdas, opt_state, batch_losses = update(
        params,lambdas,lam,Pe, opt_state, batch_data,batch_indices)
        
        epoch_losses.append(batch_losses)
        # Print and Store
        if it % 1000 == 0 or it==initialization_step or it==activate_physics  or it==full_model or it==refine_it:
            it_opt=it_opt+1000
            lr = lr0 * decay_rate ** (it_opt / decay_step)
           
            if it==activate_physics:
                print(f"NEW STAGE: Physics Constant D")
                lam = {
                    'P': 10,
                    'Mass': 0.01,
                    'Data': 100,
                }
                print(f"Reinitializing lambdas:")
                lambdas={}
                for key in dataset.keys():
                    np.random.seed(seed_np)
                    if not(key == 'GT' or key == 'Test'):
                        lambdas[key] = np.random.uniform(0, 1, size=(len(dataset[key]), neq[key]))
                        if init_zero:
                            lambdas[key]= lambdas[key]*0+max_lambda/5
                        print(f'Lambda shape for {key} dataset: {lambdas[key].shape}')
                print(f"Reinitializing Optimizer:")
                Te=num_gsteps-it
                it_opt=0
                lr0=lr
                optimizer={}
                opt_state={}
                for key in layers.keys():
                    print(f'Initializing Optimizer for:{key}')
                    optimizer[key],decay_step= initialize_optimizer(lr0, decay_rate, lrf, decay_step, num_gsteps,optimizer_type=args.optimizer)
                    opt_state[key]=optimizer[key].init(params[key])
                update=update_stage_3
                print(f"Setting update function(Fix c):{update}")
                #Save parameters:
                save_params_dict(params,result_path,dataset_name,type='Stage_2',Use_ResNet=Use_ResNet)
                save_list(log_loss,result_path,f'{dataset_name}-Stage_2_loss')
            elif it==full_model:
                print(f"NEW STAGE: Physics Constant D")
                lam = {
                    'P': 10,
                    'Mass': 0.01,
                    'Data': 100,
                }
                print(f"Reinitializing lambdas:")
                lambdas={}
                for key in dataset.keys():
                    np.random.seed(seed_np)
                    if not(key == 'GT' or key == 'Test'):
                        lambdas[key] = np.random.uniform(0, 1, size=(len(dataset[key]), neq[key]))
                        if init_zero:
                            lambdas[key]= lambdas[key]*0+max_lambda/5
                        print(f'Lambda shape for {key} dataset: {lambdas[key].shape}')
                print(f"Reinitializing Optimizer:")
                Te=num_gsteps-it
                it_opt=0
                lr0=lr
                optimizer={}
                opt_state={}
                for key in layers.keys():
                    print(f'Initializing Optimizer for:{key}')
                    optimizer[key],decay_step= initialize_optimizer(lr0, decay_rate, lrf, decay_step, num_gsteps,optimizer_type=args.optimizer)
                    opt_state[key]=optimizer[key].init(params[key])
                update=update_stage_4
                print(f"Setting update function(Fix c):{update}")
                #Save parameters:
                save_params_dict(params,result_path,dataset_name,type='Stage_3',Use_ResNet=Use_ResNet)
                save_list(log_loss,result_path,f'{dataset_name}-Stage_3_loss')
            elif it==refine_it:
                print(f"NEW STAGE: Refine it")
                lam = {
                    'P': 1,
                    'Mass': 0.01,
                    'Data': 10,
                }
                print(f"Reinitializing lambdas:")
                lambdas={}
                for key in dataset.keys():
                    np.random.seed(seed_np)
                    if not(key == 'GT' or key == 'Test'):
                        lambdas[key] = np.random.uniform(0, 1, size=(len(dataset[key]), neq[key]))
                        if init_zero:
                            lambdas[key]= lambdas[key]*0+max_lambda/5
                        print(f'Lambda shape for {key} dataset: {lambdas[key].shape}')
                print(f"Reinitializing Optimizer:")
                Te=num_gsteps-it
                it_opt=0
                lr0=lr
                optimizer={}
                opt_state={}
                for key in layers.keys():
                    print(f'Initializing Optimizer for:{key}')
                    optimizer[key],decay_step= initialize_optimizer(lr0, decay_rate, lrf, decay_step, num_gsteps,optimizer_type=args.optimizer)
                    opt_state[key]=optimizer[key].init(params[key])
                update=update_stage_5
                print(f"Setting update function:{update}")
                #Save parameters:
                save_params_dict(params,result_path,dataset_name,type='Stage_4',Use_ResNet=Use_ResNet)
                save_list(log_loss,result_path,f'{dataset_name}-Stage_4_loss')
            error_c= compute_error_metrics(params)
            error_u, error_v, error_w,error_p= compute_error_metrics_init(params)

            # Loss computation
            loss_avg_dict = {key: np.mean([loss[key] for loss in jax.device_get(epoch_losses)]) for key in epoch_losses[0].keys()}
            loss_avg_dict['RL2_u']=error_u
            loss_avg_dict['RL2_v']=error_v
            loss_avg_dict['RL2_w']=error_w
            loss_avg_dict['RL2_p']=error_p
            loss_avg_dict['RL2_c']=error_c
            loss_avg_dict['lr']=lr
            log_loss.append(loss_avg_dict)
            loss =sum(loss_avg_dict.values())
            epoch_losses = [] 

            # Lambda metrics
            lambda_res_metrics = {
                "Max": np.max(lambdas['Res_1']),
                "Min": np.min(lambdas['Res_1']),
                "Mean": np.mean(lambdas['Res_1'])
            }

            lambda_data_metrics = {
                "Max": np.max(lambdas['Data']),
                "Min": np.min(lambdas['Data']),
                "Mean": np.mean(lambdas['Data'])
            }
            # Compute learning rate (assuming you've defined lr0, decay_rate, decay_step earlier in the code)
            elapsed_time=time.time() - start_time
            running_time=running_time+elapsed_time
            

            print("-" * 100)
            print(args)
            print(f"It: {it:d}, Total Loss: {loss:.3e},lam Data:{lam['Data']:.3e},lam Physics:{lam['P']:.3e}")

            for key, value in loss_avg_dict.items():
                print(f"L1 {key.capitalize()}: {value:.3e}")

            print(f"Lambda PDE (Max): {lambda_res_metrics['Max']:.3e}, Lambda PDE (Min): {lambda_res_metrics['Min']:.3e}, Lambda PDE (Mean): {lambda_res_metrics['Mean']:.3e}")
            print(f"Lambda Data (Max): {lambda_data_metrics['Max']:.3e}, Lambda Data (Min): {lambda_data_metrics['Min']:.3e}, Lambda Data (Mean): {lambda_data_metrics['Mean']:.3e}")
            print(f'Iteration time: {elapsed_time:.2f}s, Total Time: {running_time/3600:.2f}h, Iteration/Total iterations: {it:.2f}/{num_gsteps:.2f}, Lr: {lr:.1e}')

            sys.stdout.flush()
            start_time = time.time()
            # Clean cache and run garbage collection
            update._clear_cache()
            gc.collect()
        if it % 50000 == 0 or it ==num_gsteps:
            save_list(log_loss,result_path,f'{dataset_name}-_loss')
            save_params_dict(params,result_path,dataset_name,type='Train',Use_ResNet=Use_ResNet)
        it += 1
for arg, value in vars(args).items():
    print(f'{arg}: {value}')

# %%
if Run_MODE=='Train':
    save_list(log_loss,result_path,f'{dataset_name}-_Final_loss')
    save_list(log_error_u,result_path,f'{dataset_name}-_Final_error_u')
    save_list(log_error_v,result_path,f'{dataset_name}-_Final_error_v')
    save_params_dict(params,result_path,dataset_name,type='Final',Use_ResNet=Use_ResNet)

else:
    loaded_params = load_params_dict(result_path, dataset_name, layers, initialization,type='Final',Use_ResNet=Use_ResNet) 
    print('Parameters loaded from file')

    loss_path=result_path+f'{dataset_name}-_loss.npy'
    losses=np.load(loss_path,allow_pickle=True)
    log_loss=losses
    params=loaded_params
    print('Parameters Loaded from file')

# %%
print(result_path)
print(dataset_name)
print(layers)
print(M1)
print(M2)

# %%
dataset['Test']=X_test2

# %% [markdown]
# ## Losses and Errors

# %%
# Call the function
plot_losses_grid(log_loss)
plt.savefig(result_path + f'Losses.png')
plt.show()


# %% [markdown]
# # Plots

# %%
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
u_pred  = vmap(u_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
v_pred  = vmap(v_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
w_pred  = vmap(w_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
p_pred  = vmap(p_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
c_pred  = vmap(c_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

RL2_c = relative_error2(c_pred, dataset['GT'][:,4:5])
print(f'Relative L2 errors for c:{RL2_c:.4e}')
u_max, u_min = np.max(u_pred), np.min(u_pred)
v_max, v_min = np.max(v_pred), np.min(v_pred)
w_max, w_min = np.max(w_pred), np.min(w_pred)
p_max, p_min = np.max(p_pred), np.min(p_pred)
c_max, c_min = np.max(c_pred), np.min(c_pred)
vmag_max, vmag_min= np.max(velocity_magnitude), np.min(velocity_magnitude)

# %%
#Arrays
T_star=X_nnf[:,:,0]
X_star=X_nnf[:,:,1]
Y_star=X_nnf[:,:,2]
Z_star=X_nnf[:,:,3]
C_star=X_nnf[:,:,4]

k1=1*10**(-12)#m2W
D1=3.8*10**(-10)#m2/s
k2=1*10**(-16)
D2=2.3*10**(-10)
nu_e=7*10**(-7)
rho=993
h=0.00034 #Maximum length by 100
L_char=h
U_char=0.1#mm/min
D_char=2.4e-10*5#m2/s
U_char=U_char*(1/1000)*(1/60)#m/s
T_char0=np.copy(T_char)
T_char=L_char/U_char
C_char=1/(100/max_all)
mms=10**3
L_char=L_char*mms
U_char=U_char*mms*60
print(T_char,L_char,U_char,C_char,D_char)
rec_fact=np.array([[T_char/60,L_char,L_char,L_char,U_char,U_char,U_char,P_char,C_char]])
print(rec_fact)

# %%
Errors_c=[]
min_vals=[]
max_vals=[]
average_std=0
for t__frame_plot in tqdm.tqdm(range(avoid_from_org,len(T_star))):
    snap = np.array([t__frame_plot])
    t_test = T_star[snap,:].T
    x_test = X_star[snap,:].T
    y_test = Y_star[snap,:].T
    z_test = Z_star[snap,:].T
    c_test = C_star[snap,:].T
    X_test=np.hstack((t_test,x_test,y_test,z_test))
    # Further computations for errors if any
    u_pred  = vmap(u_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]
    v_pred  = vmap(v_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]
    w_pred  = vmap(w_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]
    p_pred  = vmap(p_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]
    c_pred  = vmap(c_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]*C_char
    c_std_pred  = vmap(c_std_fx, (0))(X_test[:,0:layers_ref[0]])[:,None]*C_char
    average_std=c_std_pred+average_std
    velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
    uvwpc_pred   = np.hstack((u_pred,v_pred,w_pred,p_pred,c_pred))
    u_pred, v_pred, w_pred, p_pred,c_pred,  =uvwpc_pred[:,0:1],uvwpc_pred[:,1:2],uvwpc_pred[:,2:3],uvwpc_pred[:,3:4],uvwpc_pred[:,4:5]
    error_c = relative_error(c_pred, c_test)
    data_frame=np.hstack((X_test,u_pred, v_pred, w_pred, p_pred,c_pred))
    data_frame=data_frame*rec_fact
    Errors_c.append(error_c)
    min_vals.append(np.min(uvwpc_pred,axis=0))
    max_vals.append(np.max(uvwpc_pred,axis=0))


# %%
mpl.rcParams['font.size'] = 16
plt.figure(figsize=(9, 3))
plt.plot(5+np.arange(len(Errors_c)),jnp.array(Errors_c)*100, linestyle='-')
#plt.yscale('log')  # Set logarithmic scale for the y-axis
plt.xlabel('Time (min)')
plt.title(r'Concentration Relative $L^2$ Error (%)')
plt.grid(True)
plt.tight_layout()
plt.savefig('RL2_c.png',dpi=300)
plt.show()

# %%
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
c_pred  = np.abs(vmap(c_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None])*C_char+1e-16
c_test  =np.abs(dataset['GT'][:,4:5])*C_char+1e-16
RL2_c = relative_error2(c_pred, c_test)
print(f'Relative L2 errors for c:{RL2_c:.4e}')

mpl.rcParams['font.size'] = 16

# --- Define helper function for computing PDFs ---
def get_pdf(data, bins=300):
    hist, bins = np.histogram(data, bins=bins, density=True)
    return hist, bins

# Apply logarithmic transformation to concentration data
c_pred = np.log10(c_pred)
c_test = np.log10(c_test)

# --- Compute PDFs for c_pred and c_test ---
bins_pdf = 100  # number of bins to use for the histogram
pdf_pred, bins_pred = get_pdf(c_pred.flatten(), bins=bins_pdf)
pdf_test, bins_test = get_pdf(c_test.flatten(), bins=bins_pdf)

# --- Create a combined figure with two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# --- Left Panel: PDF Comparison Plot ---
ax1 = axes[1]
ax1.plot(bins_test[:-1], pdf_test, 'orange', label=rf'Reference', linewidth=2)
ax1.plot(bins_pred[:-1], pdf_pred, label=rf'Prediction', linewidth=2)
ax1.set_xlabel(rf'$\log_{{10}}|c|$', fontsize=14)
ax1.set_ylabel('Probability Density', fontsize=14)
#ax1.set_title('Concentration Distribution', fontsize=16)
ax1.legend(fontsize=12)
#ax1.set_yscale('log')  # Set y-axis to log scale
ax1.grid(True)
ax1.set_xlim(-4, 0)  # Limit the x-axis from -6 to 0

# --- Right Panel: Relative L2 Error Plot ---
# Assumes Errors_c is a list/array defined elsewhere
ax2 = axes[0]
# X-axis: starting at 5 minutes, one value per error in Errors_c
ax2.plot(5 + np.arange(len(Errors_c)), jnp.array(Errors_c) * 100, linestyle='-', linewidth=2)
ax2.set_xlabel('Time (min)', fontsize=14)
ax2.set_ylabel(rf'Relative $L^2$ Error (%)', fontsize=14)
#ax2.set_title(r'Concentration Relative $L^2$ Error (%)', fontsize=16)
ax2.grid(True)

plt.tight_layout()
plt.savefig('combined_concentration_plots.png', dpi=300)
plt.show()


# %%
import matplotlib.gridspec as gridspec

t_list = [5, 30, 80]
variables_all = []

# Collect data for each time step
for t__frame_plot in t_list:
    snap = np.array([t__frame_plot])
    t_test = T_star[snap,:].T
    x_test = X_star[snap,:].T
    y_test = Y_star[snap,:].T
    z_test = Z_star[snap,:].T
    c_test = C_star[snap,:].T
    X_test = np.hstack((t_test, x_test, y_test, z_test))

    u_pred = vmap(u_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    v_pred = vmap(v_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    w_pred = vmap(w_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    p_pred = vmap(p_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*P_char
    c_pred = vmap(c_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char
    c_std_pred = vmap(c_std_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char

    uvwpc_pred = np.hstack((u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred))
    u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred = [uvwpc_pred[:, i:i+1] for i in range(6)]
    data_frame = np.hstack((X_test, u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred))

    idx_plot = np.random.choice(len(y_test), int(len(y_test)), replace=False)
    t_test, x_test, y_test, z_test, u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred = [np.copy(data_frame[idx_plot, i:i+1]) for i in range(10)]
    c_test = np.copy(c_test[idx_plot])

    x_midplane = np.median(x_test)
    y_midplane = np.median(y_test)
    z_midplane = np.median(z_test)

    tolerance = 0.5
    y_midplane_filter = np.abs(y_test - y_midplane) <= tolerance

    # Save the Y–Z midplane data for plotting
    x, y = x_test, z_test
    filter = y_midplane_filter
    variables_all.append({
        'time': t__frame_plot,
        'data': [c_pred,np.abs(c_test-c_pred), c_std_pred],
        'labels': [r'$c$', r'$|\hat{c}-c|$',r'$\sigma_c$', ],
        'filter': filter,
        'x': x,
        'y': y
    })


# %%
fig = plt.figure(figsize=(12, 2.5 * len(t_list)))
gs = gridspec.GridSpec(nrows=len(t_list), ncols=3, width_ratios=[1, 1, 1])

for i, entry in enumerate(variables_all):
    time = entry['time']
    vars_t = entry['data']
    labels = entry['labels']
    x, y = entry['x'], entry['y']
    filter = entry['filter']

    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        sc = ax.scatter(x[filter], y[filter], c=vars_t[j][filter], cmap=cmap, s=3)
        ax.set_title(f'{labels[j]} at t={time} min', pad=6)
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # Independent colorbar for each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.02)
        plt.colorbar(sc, cax=cax)

plt.tight_layout()
plt.savefig("combined_timesteps.png", dpi=300)


# %%
import matplotlib.gridspec as gridspec

t_list = [5, 30, 80]
variables_all = []

# Collect data for each time step
for t__frame_plot in t_list:
    snap = np.array([t__frame_plot])
    t_test = T_star[snap,:].T
    x_test = X_star[snap,:].T
    y_test = Y_star[snap,:].T
    z_test = Z_star[snap,:].T
    c_test = C_star[snap,:].T
    X_test = np.hstack((t_test, x_test, y_test, z_test))

    u_pred = vmap(u_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    v_pred = vmap(v_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    w_pred = vmap(w_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    p_pred = vmap(p_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*P_char
    c_pred = vmap(c_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char*100
    c_std_pred = vmap(c_std_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char*100

    uvwpc_pred = np.hstack((u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred))
    u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred = [uvwpc_pred[:, i:i+1] for i in range(6)]
    data_frame = np.hstack((X_test, u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred))

    idx_plot = np.random.choice(len(y_test), int(len(y_test)), replace=False)
    t_test, x_test, y_test, z_test, u_pred, v_pred, w_pred, p_pred, c_pred, c_std_pred = [np.copy(data_frame[idx_plot, i:i+1]) for i in range(10)]
    c_test = np.copy(c_test[idx_plot])*100

    x_midplane = np.median(x_test)
    y_midplane = np.median(y_test)
    z_midplane = np.median(z_test)

    tolerance = 0.5
    y_midplane_filter = np.abs(y_test - y_midplane) <= tolerance
    z_midplane_filter = np.abs(z_test - z_midplane) <= tolerance

    # Save the Y–Z midplane data for plotting
    x, y = x_test, y_test
    filter = z_midplane_filter
    variables_all.append({
        'time': t__frame_plot,
        'data': [c_pred,np.abs(c_test-c_pred), c_std_pred,c_test],
        'labels': [r'$c$', r'$|\hat{c}-c|$',r'$\sigma_c$', ],
        'filter': filter,
        'x': x,
        'y': y,
    })


# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np # Make sure numpy is imported

# --- Assumed Prerequistes ---
# [Same as before - variables_all, t_list, L_char, cmap assumed defined]
# Example definitions if needed:
# L_char = 10.0
# cmap = 'viridis'
# t_list = [5, 30, 80]
# Assume variables_all is pre-populated correctly:
#   'data': [c_pred, np.abs(c_pred - c_test), c_std_pred, c_test]
# --- End Assumed Prerequistes ---

num_rows = 4 # Prediction, Reference, Error, Uncertainty
num_cols = len(t_list) # Number of time steps

# Adjust figsize: (width, height)
fig = plt.figure(figsize=(4.5 * num_cols, 2* num_rows)) # Keep adjusted height
gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols, figure=fig)

# Define the mapping from plot ROW index (i) to DATA index in entry['data']
# entry['data'] = [c_pred=0, error=1, c_std=2, c_test=3]
# Row 0 (Label hat(c)) -> data_idx 0
# Row 1 (Label c) -> data_idx 3
# Row 2 (Label error) -> data_idx 1
# Row 3 (Label sigma_c) -> data_idx 2
data_indices = [0, 3, 1, 2]

# Define the barscale length IN SCALED DATA UNITS
barscale_length = 3 # adjust to your desired length in scaled units
barscale_label = f'{barscale_length} mm'  # adjust label and units as needed

# Loop through columns (Time steps)
for j, entry in enumerate(variables_all):
    time = entry['time']
    # Apply scaling to coordinates for this time step
    x, y = entry['x'] * L_char, entry['y'] * L_char
    plot_filter = entry['filter']

    # --- Calculate common vmin/vmax for Row 0 (Pred) and Row 1 (True) ---
    data_pred_idx = data_indices[0] # Index for c_pred in entry['data']
    data_true_idx = data_indices[1] # Index for c_test in entry['data']

    data_pred = entry['data'][data_pred_idx]
    data_true = entry['data'][data_true_idx]

    # Apply filter
    filtered_pred = data_pred[plot_filter]
    filtered_true = data_true[plot_filter]

    # Concatenate filtered data to find overall min/max
    combined_data = np.concatenate(
        (filtered_pred.flatten(), filtered_true.flatten())
        )

    common_vmin, common_vmax = None, None
    if combined_data.size > 0: # Check if there's any data after filtering
        common_vmin = np.min(combined_data)
        common_vmax = np.max(combined_data)
    else:
        common_vmin, common_vmax = 0, 1 # Example default

    # Check if min and max are the same
    if common_vmin is not None and common_vmin == common_vmax:
         # Add tiny offset if min/max are same and not None
         common_vmax += 1e-6


    # Loop through rows (Quantities for the current time step j)
    for i in range(num_rows):
        ax = fig.add_subplot(gs[i, j]) # Get subplot at row i, column j

        # Select the correct data array for this plot row (i) using the mapping
        data_index = data_indices[i]
        data_to_plot = entry['data'][data_index]

        # --- Create the scatter plot ---
        # Apply common vmin/vmax only for the first two rows (i=0, i=1)
        if (i == 0 or i == 1) and common_vmin is not None:
             sc = ax.scatter(x[plot_filter], y[plot_filter],
                             c=data_to_plot[plot_filter],
                             cmap=cmap, s=3,
                             vmin=common_vmin, vmax=common_vmax) # Apply shared limits
        else:
             # Let other rows (error, uncertainty) scale automatically
             sc = ax.scatter(x[plot_filter], y[plot_filter],
                             c=data_to_plot[plot_filter],
                             cmap=cmap, s=3)

        # --- Axis Titles and Formatting ---
        # Set time title only for the top row (i=0)
        if i == 0:
            ax.set_title(f't={time} min', pad=6)

        # Removed the ax.text() call for left-side row labels

        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

        # --- Independent colorbar for each subplot ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.05) # Adjusted size
        plt.colorbar(sc, cax=cax)

        # --- Add barscale ONLY to the top-left subplot (i=0, j=0) ---
        if i == 0 and j == 0:
            # Get axis limits *after* plotting
            x_min_lim, x_max_lim = ax.get_xlim()
            y_min_lim, y_max_lim = ax.get_ylim()

            # Position bar near top-right corner
            bar_y = y_min_lim + 0.95 * (y_max_lim - y_min_lim) # 95% up from bottom
            bar_x_end = x_min_lim + 0.95 * (x_max_lim - x_min_lim) # 95% from left
            bar_x_start = bar_x_end - barscale_length

            # Position text below the bar
            text_y = bar_y - 0.05 * (y_max_lim - y_min_lim) # Adjust vertical spacing
            text_x = bar_x_start + barscale_length / 2 # Center text horizontally

            # Plot the bar
            ax.plot([bar_x_start, bar_x_end], [bar_y, bar_y], color='black', lw=4)
            # Add the text label
            ax.text(text_x, text_y, barscale_label,
                    ha='center', va='top', fontsize=14, color='black')


# Adjust layout to prevent overlap
plt.tight_layout() # Removed rect parameter

plt.savefig("combined_4rows_timesteps_scaled_shared_cbar.png", dpi=300)
plt.show() # Optional: display the plot

# %%
# Convert lists of min and max values to numpy arrays for easy indexing
min_vals_arr = np.array(min_vals)
min_vals_arr = np.hstack((min_vals_arr[:,4:5],min_vals_arr[:,4:5],min_vals_arr[:,:4]))
max_vals_arr = np.array(max_vals)
max_vals_arr = np.hstack((max_vals_arr[:,4:5],max_vals_arr[:,4:5],max_vals_arr[:,:4]))


# Compute global min and max for each variable across all frames
global_min = min_vals_arr.min(axis=0)
global_max = max_vals_arr.max(axis=0)
print(global_min)
print(global_max)
start=70
end=start+1
#end=len(T_star)

for t__frame_plot in tqdm.tqdm(range(start,end)):
    snap = np.array([t__frame_plot])
    t_test = T_star[snap,:].T
    x_test = X_star[snap,:].T
    y_test = Y_star[snap,:].T
    z_test = Z_star[snap,:].T
    c_test = C_star[snap,:].T
    X_test=np.hstack((t_test,x_test,y_test,z_test))
    # Further computations for errors if any
    u_pred  = vmap(u_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    v_pred  = vmap(v_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    w_pred  = vmap(w_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*U_char
    p_pred  = vmap(p_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*P_char
    c_pred  = vmap(c_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char
    c_std_pred  = vmap(c_std_fx, (0))(X_test[:,0:layers_1[0]])[:,None]*C_char

    velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
    uvwpc_pred   = np.hstack((u_pred,v_pred,w_pred,p_pred,c_pred,c_std_pred))
    u_pred, v_pred, w_pred, p_pred, c_pred,c_std_pred = uvwpc_pred[:, 0:1], uvwpc_pred[:, 1:2], uvwpc_pred[:, 2:3], uvwpc_pred[:, 3:4] , uvwpc_pred[:, 4:5], uvwpc_pred[:, 5:6]
    data_frame = np.hstack((X_test, u_pred, v_pred, w_pred, p_pred, c_pred,c_std_pred))

    idx_plot = np.random.choice(len(y_test), int(len(y_test)), replace=False)
    t_test, x_test, y_test, z_test, u_pred, v_pred, w_pred, p_pred, c_pred,c_std_pred = [np.copy(data_frame[idx_plot, i:i+1]) for i in range(10)]
    c_test = np.copy(c_test[idx_plot])

    x_midplane = np.median(x_test)
    y_midplane = np.median(y_test)
    z_midplane = np.median(z_test)

    tolerance = 0.5
    x_midplane_filter = np.abs(x_test - x_midplane) <= tolerance
    y_midplane_filter = np.abs(y_test - y_midplane) <= tolerance
    z_midplane_filter = np.abs(z_test - z_midplane) <= tolerance
    if end>start+1:
        variables = [c_test, c_pred]
        variable_names = ['c_real', 'c']
    variables = [c_test, c_pred,c_std_pred, u_pred, v_pred, w_pred, p_pred]
    variable_names = ['c_real', 'c','c_std' ,'u($mm/min$)', 'v($mm/min$)', 'w($mm/min$)', '$p(Pa)$']
    name_image = Video_path + f'time:{t__frame_plot / len(T_star):.4f}.png'

    fig = plt.figure(figsize=(14, 2 * len(variables)))

    for i, var in enumerate(variables):
        for j in range(3):
            ax = fig.add_subplot(len(variables), 3, i * 3 + j + 1)
            if j == 0:
                filter = x_midplane_filter
                x, y = z_test, y_test
                xlabel, ylabel = 'Y', 'Z'
            elif j == 1:
                filter = y_midplane_filter
                x, y = x_test, z_test
                xlabel, ylabel = 'X', 'Z'
            else:
                filter = z_midplane_filter
                x, y = x_test, y_test
                xlabel, ylabel = 'X', 'Y'

            #vmin, vmax = global_min[i], global_max[i]  # Use the global min/max values for each variable
            scatter = ax.scatter(x[filter], y[filter], c=var[filter], cmap=cmap, s=3)#], vmin=vmin, vmax=vmax)
            ax.set_title(f'{variable_names[i]} at t={t__frame_plot}')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis('equal')
            ax.axis('off')

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(scatter, cax=cax)

    plt.tight_layout()
    plt.savefig(name_image)
    if end>start+1:
        plt.close()


# %%
def relative_error2_log(pred,exact):
    return np.linalg.norm(np.log(exact+0.000000001)-np.log(pred+0.000000001),2)/np.linalg.norm(np.log(exact+0.000000001),2)

# %%
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
u_pred  = vmap(u_fx, (0))(dataset['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
v_pred  = vmap(v_fx, (0))(dataset['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
w_pred  = vmap(w_fx, (0))(dataset['Test_crop'][:,0:layers_1[0]])[:,None]*U_char
p_pred  = vmap(p_fx, (0))(dataset['Test_crop'][:,0:layers_1[0]])[:,None]*p_scale*P_char
p_pred  =p_pred-p_pred.min()
c_pred  = vmap(c_fx, (0))(dataset['Test_crop'][:,0:layers_1[0]])[:,None]

velocity_magnitude = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

RL2_u = relative_error2(u_pred, dataset['Test_crop'][:,4:5]*U_char)
RL2_v = relative_error2(v_pred, dataset['Test_crop'][:,5:6]*U_char)
RL2_w = relative_error2(w_pred, dataset['Test_crop'][:,6:7]*U_char)

u_max, u_min = np.max(u_pred), np.min(u_pred)
v_max, v_min = np.max(v_pred), np.min(v_pred)
w_max, w_min = np.max(w_pred), np.min(w_pred)
p_max, p_min = np.max(p_pred), np.min(p_pred)
c_max, c_min = np.max(c_pred), np.min(c_pred)
vmag_max, vmag_min= np.max(velocity_magnitude), np.min(velocity_magnitude)
x_test = dataset['Test_crop'][:,1:2]
y_test = dataset['Test_crop'][:,2:3]
z_test = dataset['Test_crop'][:,3:4]
u_test = dataset['Test_crop'][:,4:5]*U_char  
v_test = dataset['Test_crop'][:,5:6]*U_char  
w_test = dataset['Test_crop'][:,6:7]*U_char 
p_test = dataset['Test_crop'][:,7:8]*P_char
p_test  =p_test-p_test.min()
RL2_p = relative_error2(p_pred, p_test)
print(f'Relative L2 errors for u:{RL2_u:.4e} v:{RL2_v:.4e} w:{RL2_w:.4e}, p:{RL2_p:.4e}')

V_test = np.sqrt(u_test**2 + v_test**2 + w_test**2)
V_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

# %%
p_test.min()

# %%
p_test.max()

# %%
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
c_pred  = vmap(c_fx, (0))(dataset['GT'][:,0:layers_ref[0]])[:,None]
c_test =dataset['GT'][:,4:5]

RL2_c = relative_error2(c_pred, c_test)
print(f'Relative L2 errors for c:{RL2_c:.4e}')


# %%
name_file=args.Name+'.mat'
Data_save={
    'txyz_smm':dataset['Test_crop'][:,0:layers_1[0]]*rec_fact[:,:4],
    'u_mm_s':u_pred,
    'v_mm_s':v_pred,
    'w_mm_s':w_pred,
    'p_Pa':p_pred,
    'u_mm_s_true':u_test,
    'v_mm_s_true':v_test,
    'w_mm_s_true':w_test,
    'P_Pa_true':p_test,
    'c_pred':c_pred,
    'c_test':c_test,
    'Error_c':Errors_c,
    'percentage_discarded':percentage_discarded,
}
scipy.io.savemat(name_file,Data_save)

# %%
name_file

# %%

mpl.rcParams['font.size'] = 20
size_point=20

# Calculate midplane values
x_mid = np.median(x_test)
y_mid = np.median(y_test)
z_mid = np.median(z_test)

# Define tolerance for midplane filtering
tolerance = 0.5

# Create filters for midplane slices
x_filter = np.abs(x_test - x_mid) <= tolerance
y_filter = np.abs(y_test - y_mid) <= tolerance
z_filter = np.abs(z_test - z_mid) <= tolerance

# Prepare the data for plotting
data_pairs = [
    (u_test, u_pred, 'u FD', 'u Pred'),
    (v_test, v_pred, 'v FD', 'v Pred'),
    (w_test, w_pred, 'w FD', 'w Pred'),
    (p_test, p_pred, 'p FD', 'p Pred'),
    (V_test, V_pred, 'V FD', 'V Pred')
]

# Define colormap
cmap = plt.get_cmap('plasma')

# Start plotting
fig, axes = plt.subplots(nrows=15, ncols=3, figsize=(20, 40))#, constrained_layout=True)  # Adjusted for additional rows for error plots

for i, (test_data, pred_data, test_label, pred_label) in enumerate(data_pairs):
    # Compute global vmin and vmax for each pair from the test data
    vmin = np.min(test_data)
    vmax = np.max(test_data)
    error_data = np.abs(test_data - pred_data)  # Calculate absolute error

    for j in range(3):
        if j == 0:  # X Midplane: Plot Y vs Z
            x, y = z_test, y_test
            xlabel, ylabel = 'Z', 'Y'
            filter = x_filter
        elif j == 1:  # Y Midplane: Plot X vs Z
            x, y = x_test, z_test
            xlabel, ylabel = 'X', 'Z'
            filter = y_filter
        else:  # Z Midplane: Plot X vs Y
            x, y = x_test, y_test
            xlabel, ylabel = 'X', 'Y'
            filter = z_filter

        # Index calculation for subplot
        ax_test = axes[i*3, j]
        ax_pred = axes[i*3+1, j]
        ax_error = axes[i*3+2, j]  # Additional axis for error

        # Plot test data
        sc_test = ax_test.scatter(x[filter], y[filter], c=test_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_test.set_title(f'{test_label} - {xlabel}-{ylabel} Plane')
        ax_test.set_xlabel(xlabel)
        ax_test.set_ylabel(ylabel)
        ax_test.axis('equal')
        ax_test.axis('off')
        divider = make_axes_locatable(ax_test)
        cax_test = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_test, cax=cax_test)

        # Plot predicted data
        sc_pred = ax_pred.scatter(x[filter], y[filter], c=pred_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_pred.set_title(f'{pred_label} - {xlabel}-{ylabel} Plane')
        ax_pred.set_xlabel(xlabel)
        ax_pred.set_ylabel(ylabel)
        ax_pred.axis('equal')
        ax_pred.axis('off')
        divider = make_axes_locatable(ax_pred)
        cax_pred = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_pred, cax=cax_pred)

        # Plot error data
        sc_error = ax_error.scatter(x[filter], y[filter], c=error_data[filter], cmap=cmap, vmin=0,s=size_point)#, vmax=vmax-vmin)
        ax_error.set_title(f'Error - {xlabel}-{ylabel} Plane')
        ax_error.set_xlabel(xlabel)
        ax_error.set_ylabel(ylabel)
        ax_error.axis('equal')
        ax_error.axis('off')
        divider = make_axes_locatable(ax_error)
        cax_error = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_error, cax=cax_error)
plt.tight_layout()
plt.show()

# Save or show the figure


# %%

def get_pdf(u_hist, bins=30):
    hist, bins = np.histogram(u_hist, bins=bins, density=True)
    prob_density = hist
    return prob_density, bins
def get_colors_plot(cmap='RdBu', n_colors=2):
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    return colors
def plot_pdf_fields_with_error(datasets, field_names, colors, bins=100, figsize=(8, 8)):
    num_fields = len(datasets)
    fig, axs = plt.subplots(num_fields, 2, figsize=figsize)  # 3 columns for data, RBA, and error

    for i, ((data_piv, data_aiv), field_name) in enumerate(zip(datasets, field_names)):
        prob_density_piv, bins_piv = get_pdf(data_piv, bins=bins)
        prob_density_aiv, bins_aiv = get_pdf(data_aiv, bins=bins)
        error = data_piv - data_aiv
        prob_density_error, bins_error = get_pdf(error, bins=bins)

        # Plotting data
        axs[i, 0].plot(bins_piv[:-1], prob_density_piv, color=colors[0], label=f'FD Data')
        axs[i, 0].fill_between(bins_piv[:-1], prob_density_piv, color=colors[0], alpha=0.2)
        axs[i, 0].set_xlabel(f'{field_name}')
        axs[i, 0].plot(bins_aiv[:-1], prob_density_aiv, color=colors[-2], label='RBA')
        axs[i, 0].fill_between(bins_aiv[:-1], prob_density_aiv, color=colors[-2], alpha=0.2)
        axs[i, 0].set_xlabel(f'{field_name}')
        axs[i, 0].set_ylabel('Prob. Density')
        axs[i, 0].legend(loc='upper right')
        axs[i, 0].set_yscale('log')  # Apply log scale to the y-axis

        # Plotting error
        axs[i, 1].plot(bins_error[:-1], prob_density_error, color=colors[-2], label='Error FT Data')
        axs[i, 1].fill_between(bins_error[:-1], prob_density_error, color=colors[-2], alpha=0.2)
        axs[i, 1].set_xlabel(f'Error in {field_name}')
        axs[i, 1].legend(loc='upper right')
        axs[i, 1].set_yscale('log')  # Apply log scale to the y-axis

    plt.tight_layout()
    return fig


# Extracting and reshaping data
u_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,4:5])*U_char).reshape(-1)
u_aiv_hist = np.array(jax.device_get(u_pred)).reshape(-1)
v_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,5:6])*U_char).reshape(-1)
v_aiv_hist = np.array(jax.device_get(v_pred)).reshape(-1)
w_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,6:7])*U_char).reshape(-1)
w_aiv_hist = np.array(jax.device_get(w_pred)).reshape(-1)  # Assuming w_pred is defined

bins = 40

# Calculating probability densities
prob_density_u_piv, bins_u_piv = get_pdf(u_piv_hist, bins=bins)
prob_density_u_aiv, bins_u_aiv = get_pdf(u_aiv_hist, bins=bins)
prob_density_v_piv, bins_v_piv = get_pdf(v_piv_hist, bins=bins)
prob_density_v_aiv, bins_v_aiv = get_pdf(v_aiv_hist, bins=bins)
prob_density_w_piv, bins_w_piv = get_pdf(w_piv_hist, bins=bins)
prob_density_w_aiv, bins_w_aiv = get_pdf(w_aiv_hist, bins=bins)


colors = get_colors_plot(cmap='RdBu_r', n_colors=8)
datasets = [(u_piv_hist, u_aiv_hist), (v_piv_hist, v_aiv_hist), 
            (w_piv_hist, w_aiv_hist)]
field_names = ['x-velocity', 'y-velocity', 'z-velocity', 'T']
mpl.rcParams['font.size'] = 12

fig = plot_pdf_fields_with_error(datasets, field_names, colors)

plt.savefig(result_path + '/AIV-Prob_dist_epochs.png')
plt.show()



# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assuming the setup for x_test, y_test, z_test, u_test, v_test, w_test, u_pred, v_pred, w_pred, and the data_pairs list
mpl.rcParams['font.size'] = 16

# Define colormap
cmap = plt.get_cmap('plasma')

# Start plotting with adjusted figsize for just the last two rows
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))

# Assuming you are working with the last pair in data_pairs
test_data, pred_data, test_label, pred_label = data_pairs[-1]

for j in range(3):
    ax_test = axes[0, j]  # First row for test data
    ax_pred = axes[1, j]  # Second row for predicted data
    
    # Determine which midplane we are plotting
    if j == 0:  # X Midplane: Plot Y vs Z
        x, y = y_test, z_test
        xlabel, ylabel = 'Y', 'Z'
        filter = x_filter
        u, v = v_test, w_test
        u_pred_sub, v_pred_sub = v_pred, w_pred
    elif j == 1:  # Y Midplane: Plot X vs Z
        x, y = x_test, z_test
        xlabel, ylabel = 'X', 'Z'
        filter = y_filter
        u, v = u_test, w_test
        u_pred_sub, v_pred_sub = u_pred, w_pred
    else:  # Z Midplane: Plot X vs Y
        x, y = x_test, y_test
        xlabel, ylabel = 'X', 'Y'
        filter = z_filter
        u, v = u_test, v_test
        u_pred_sub, v_pred_sub = u_pred, v_pred

    # Sampling 10% of the data for vector plotting
    indices = np.random.choice(np.where(filter)[0], len(x[filter]) // 50, replace=False)

    # Plot test data and vectors
    sc_test = ax_test.scatter(x[filter], y[filter], c=test_data[filter], cmap=cmap,vmin=np.min(test_data[filter]),vmax=np.max(test_data[filter]),  s=10)
    ax_test.quiver(x[indices], y[indices], u[indices], v[indices], color='yellowgreen', scale=3)
    ax_test.set_title(f'{test_label} - {xlabel}-{ylabel} Plane')
    ax_test.set_xlabel(xlabel)
    ax_test.set_ylabel(ylabel)
    ax_test.axis('equal')

    # Plot predicted data and vectors
    sc_pred = ax_pred.scatter(x[filter], y[filter], c=pred_data[filter], cmap=cmap,vmin=np.min(test_data[filter]),vmax=np.max(test_data[filter]),  s=10)
    ax_pred.quiver(x[indices], y[indices], u_pred_sub[indices], v_pred_sub[indices], color='yellowgreen', scale=3)
    ax_pred.set_title(f'{pred_label} - {xlabel}-{ylabel} Plane')
    ax_pred.set_xlabel(xlabel)
    ax_pred.set_ylabel(ylabel)
    ax_pred.axis('equal')

    # Adding colorbars to both test and predicted plots
    for ax, sc in [(ax_test, sc_test), (ax_pred, sc_pred)]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)

# Adjust layout
#$plt.tight_layout()

# Save or show the figure
plt.tight_layout()
plt.savefig(result_path + 'Final_midplane_comparison_with_velocity_magnitude.png')
plt.show()


# %%
mpl.rcParams['font.size'] = 20
size_point=20

# Calculate midplane values
x_mid = np.median(x_test)
y_mid = np.median(y_test)
z_mid = np.median(z_test)

# Define tolerance for midplane filtering
tolerance = 0.5

# Create filters for midplane slices
x_filter = np.abs(x_test - x_mid) <= tolerance
y_filter = np.abs(y_test - y_mid) <= tolerance
z_filter = np.abs(z_test - z_mid) <= tolerance

# Prepare the data for plotting
data_pairs = [
    (u_test, u_pred, 'log10(u FD', 'log10 u Pred'),
    (v_test, v_pred, 'log10 v FD', 'log10 v Pred'),
    (w_test, w_pred, 'log10 w FD', 'log10 w Pred'),
    (V_test, V_pred, 'log10 V FD', 'log10 V Pred')
]
data_pairs = [
    ( np.log10(np.abs(u_test)),  np.log10(np.abs(u_pred)), 'log10(u FD)', 'log10(u Pred)'),
    ( np.log10(np.abs(v_test)),  np.log10(np.abs(v_pred)), 'log10(v FD)', 'log10(v Pred)'),
    ( np.log10(np.abs(w_test)),  np.log10(np.abs(w_pred)), 'log10(w FD)', 'log10(w Pred)'),
    ( np.log10(np.abs(V_test)),  np.log10(np.abs(V_pred)), 'log10(V FD)', 'log10(V Pred)')
]

# Define colormap
cmap = plt.get_cmap('plasma')

# Start plotting
fig, axes = plt.subplots(nrows=12, ncols=3, figsize=(20, 40))#, constrained_layout=True)  # Adjusted for additional rows for error plots

for i, (test_data, pred_data, test_label, pred_label) in enumerate(data_pairs):
    # Compute global vmin and vmax for each pair from the test data
    vmin = np.min(test_data)
    vmax = np.max(test_data)
    error_data = np.abs(test_data-pred_data)/ np.abs(test_data)  # Calculate absolute error
    idx_remove=np.argwhere(error_data>0.3).flatten()
    error_data[idx_remove]=np.nan
    error_data=error_data
    print(len(idx_remove),len(test_data))
    print(len(idx_remove)/len(test_data))


    for j in range(3):
        if j == 0:  # X Midplane: Plot Y vs Z
            x, y = z_test, y_test
            xlabel, ylabel = 'Z', 'Y'
            filter = x_filter
        elif j == 1:  # Y Midplane: Plot X vs Z
            x, y = x_test, z_test
            xlabel, ylabel = 'X', 'Z'
            filter = y_filter
        else:  # Z Midplane: Plot X vs Y
            x, y = x_test, y_test
            xlabel, ylabel = 'X', 'Y'
            filter = z_filter

        # Index calculation for subplot
        ax_test = axes[i*3, j]
        ax_pred = axes[i*3+1, j]
        ax_error = axes[i*3+2, j]  # Additional axis for error

        # Plot test data
        sc_test = ax_test.scatter(x[filter], y[filter], c=test_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_test.set_title(f'{test_label} - {xlabel}-{ylabel} Plane')
        ax_test.set_xlabel(xlabel)
        ax_test.set_ylabel(ylabel)
        ax_test.axis('equal')
        ax_test.axis('off')
        divider = make_axes_locatable(ax_test)
        cax_test = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_test, cax=cax_test)

        # Plot predicted data
        sc_pred = ax_pred.scatter(x[filter], y[filter], c=pred_data[filter], cmap=cmap, vmin=vmin, vmax=vmax,s=size_point)
        ax_pred.set_title(f'{pred_label} - {xlabel}-{ylabel} Plane')
        ax_pred.set_xlabel(xlabel)
        ax_pred.set_ylabel(ylabel)
        ax_pred.axis('equal')
        ax_pred.axis('off')
        divider = make_axes_locatable(ax_pred)
        cax_pred = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_pred, cax=cax_pred)

        # Plot error data
        sc_error = ax_error.scatter(x[filter], y[filter], c=error_data[filter], cmap=cmap, vmin=0,s=size_point)#, vmax=vmax-vmin)
        ax_error.set_title(f'Relative error - {xlabel}-{ylabel} Plane')
        ax_error.set_xlabel(xlabel)
        ax_error.set_ylabel(ylabel)
        ax_error.axis('equal')
        ax_error.axis('off')
        divider = make_axes_locatable(ax_error)
        cax_error = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc_error, cax=cax_error)
plt.tight_layout()
plt.show()

# Save or show the figure


# %%
key='Permeability'
Dataset2={}
Dataset2['Permeability']=X_permeability
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]*k_char
max_log_10=np.max((K_pred))
K_pred=K_pred/(max_log_10)
K_real  =Dataset2[key][:,5:6]*k_char
max_log_10=np.max((K_real))
K_real=K_real/max_log_10
K_SI=K_pred*k_char
RL2_K = relative_error2(K_pred, K_real)
print(f'Relative L2 errors for K:{RL2_K:.4e}')
wasserstein_dist = wasserstein_distance(K_real.flatten(),K_pred.flatten())
print(f"Wasserstein distance between K_real and K_pred: {wasserstein_dist}")


# %%
mpl.rcParams['font.size'] = 20

t,x, y, z= Dataset2[key][:,0], Dataset2[key][:,1], Dataset2[key][:,2], Dataset2[key][:,3]
# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.3  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [np.log10(K_pred),np.log10(K_real),np.log10(np.abs(K_pred-K_real))/np.log10(K_real)]  # Use the scalar field for coloring
variable_names = ['K-Pred','K-Real','Relative Log Error','K-Real','K-Pred','K-Error']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed
    vmin=variables[1].min()
    vmax=variables[1].max()
    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y' 
        print(variable_names[i])
        # Use the scalar field for coloring
        if not(variable_names[i]=='Relative Log Error'):
            scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2,vmin=vmin,vmax=vmax)
        else:
            
            scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)

        ax.set_title(f'{variable_names[i]} -{["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.savefig(images_path+'Final_results.png')
plt.show()


# %%

def get_pdf(u_hist, bins=30):
    hist, bins = np.histogram(u_hist, bins=bins, density=True)
    prob_density = hist
    return prob_density, bins
def get_colors_plot(cmap='RdBu', n_colors=2):
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    return colors
def plot_pdf_fields_with_error(datasets, field_names, colors, bins=100, figsize=(8, 8)):
    num_fields = len(datasets)
    fig, axs = plt.subplots(num_fields, 2, figsize=figsize)  # 3 columns for data, RBA, and error

    for i, ((data_piv, data_aiv), field_name) in enumerate(zip(datasets, field_names)):
        prob_density_piv, bins_piv = get_pdf(data_piv, bins=bins)
        prob_density_aiv, bins_aiv = get_pdf(data_aiv, bins=bins)
        error = data_piv - data_aiv
        prob_density_error, bins_error = get_pdf(error, bins=bins)

        # Plotting data
        axs[i, 0].plot(bins_piv[:-1], prob_density_piv, color=colors[0], label=f'FD Data')
        axs[i, 0].fill_between(bins_piv[:-1], prob_density_piv, color=colors[0], alpha=0.2)
        axs[i, 0].set_xlabel(f'{field_name}')
        axs[i, 0].plot(bins_aiv[:-1], prob_density_aiv, color=colors[-2], label='RBA')
        axs[i, 0].fill_between(bins_aiv[:-1], prob_density_aiv, color=colors[-2], alpha=0.2)
        axs[i, 0].set_xlabel(f'{field_name}')
        axs[i, 0].set_ylabel('Prob. Density')
        axs[i, 0].legend(loc='upper right')
        axs[i, 0].set_yscale('log')  # Apply log scale to the y-axis

        # Plotting error
        axs[i, 1].plot(bins_error[:-1], prob_density_error, color=colors[-2], label='Error FT Data')
        axs[i, 1].fill_between(bins_error[:-1], prob_density_error, color=colors[-2], alpha=0.2)
        axs[i, 1].set_xlabel(f'Error in {field_name}')
        axs[i, 1].legend(loc='upper right')
        axs[i, 1].set_yscale('log')  # Apply log scale to the y-axis

    plt.tight_layout()
    return fig


# Extracting and reshaping data
u_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,4:5])*U_char).reshape(-1)
u_aiv_hist = np.array(jax.device_get(u_pred)).reshape(-1)
v_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,5:6])*U_char).reshape(-1)
v_aiv_hist = np.array(jax.device_get(v_pred)).reshape(-1)
w_piv_hist = np.array(jax.device_get(dataset['Test_crop'][:,6:7])*U_char).reshape(-1)
w_aiv_hist = np.array(jax.device_get(w_pred)).reshape(-1)  # Assuming w_pred is defined

bins = 40

# Calculating probability densities
prob_density_u_piv, bins_u_piv = get_pdf(u_piv_hist, bins=bins)
prob_density_u_aiv, bins_u_aiv = get_pdf(u_aiv_hist, bins=bins)
prob_density_v_piv, bins_v_piv = get_pdf(v_piv_hist, bins=bins)
prob_density_v_aiv, bins_v_aiv = get_pdf(v_aiv_hist, bins=bins)
prob_density_w_piv, bins_w_piv = get_pdf(w_piv_hist, bins=bins)
prob_density_w_aiv, bins_w_aiv = get_pdf(w_aiv_hist, bins=bins)


colors = get_colors_plot(cmap='RdBu_r', n_colors=8)
datasets = [(u_piv_hist, u_aiv_hist), (v_piv_hist, v_aiv_hist), 
            (w_piv_hist, w_aiv_hist),(K_real,K_pred)]
field_names = ['x-velocity', 'y-velocity', 'z-velocity', 'k']
mpl.rcParams['font.size'] = 12

fig = plot_pdf_fields_with_error(datasets, field_names, colors)

plt.savefig(result_path + '/AIV-Prob_dist_epochs.png')
plt.show()



# %%
key='Permeability'
Dataset2={}
Dataset2['Permeability']=X_permeability
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]*k_char
max_log_10=np.max((K_pred))
K_pred=K_pred/(max_log_10)
K_real  =Dataset2[key][:,5:6]*k_char
max_log_10=np.max((K_real))
K_real=K_real/max_log_10
K_SI=K_pred*k_char
RL2_K = relative_error2(K_pred, K_real)
print(f'Relative L2 errors for K:{RL2_K:.4e}')
wasserstein_dist = wasserstein_distance(K_real.flatten(),K_pred.flatten())
print(f"Wasserstein distance between K_real and K_pred: {wasserstein_dist}")
wasserstein_dist_log10 = wasserstein_distance(np.log10(K_real).flatten(),np.log10(K_pred).flatten())
print(f"Wasserstein distance between K_real and K_pred: {wasserstein_dist_log10}")


# %%
name_file=args.Name+'_K.mat'
Data_save={
    'txyz_smm':Dataset2['Permeability'][:,0:layers_1[0]]*rec_fact[:,:4],
    'K_pred':K_pred,
    'K_test':K_real,
}
scipy.io.savemat(name_file,Data_save)

# %%
key='Permeability'
Dataset2={}
Dataset2['Permeability']=X_permeability
u_fx,v_fx,w_fx,p_fx,c_fx,D_fx,k_fx,c_std_fx  = u_model(params)
# Further computations for errors if any
K_pred  = vmap(k_fx, (0))(Dataset2[key][:,0:layers_ref[0]])[:,None]*k_char
max_log_10=np.max((K_pred))
K_pred=K_pred/(max_log_10)
K_real  =Dataset2[key][:,5:6]*k_char
max_log_10=np.max((K_real))
K_real=K_real/max_log_10
K_SI=K_pred*k_char
RL2_K = relative_error2(K_pred, K_real)
max_log_10=np.min(np.log10((K_real)))
print(f'Relative L2 errors for K:{RL2_K:.4e}')
thresh=max_log_10/2
K_pred=np.array(K_pred)
idx_high=np.argwhere(K_pred>10**(thresh)).flatten()
print(len(idx_high))
idx_low=np.argwhere(K_pred<10**(thresh)).flatten()
print(len(idx_low))
#replace
K_pred[idx_high]=1
K_pred[idx_low]=0
idx_high=np.argwhere(K_real>10**(thresh)).flatten()
print(len(idx_high))
idx_low=np.argwhere(K_real<10**(thresh)).flatten()
print(len(idx_low))
#replace
K_real[idx_high]=1
K_real[idx_low]=0
RL2_K = relative_error2(K_pred, K_real)
print(f'Relative L2 errors for K:{RL2_K:.4e}')

# %%
mpl.rcParams['font.size'] = 20

t,x, y, z= Dataset2[key][:,0], Dataset2[key][:,1], Dataset2[key][:,2], Dataset2[key][:,3]
# Calculate midplane values
x_midplane = np.median(x)
y_midplane = np.median(y)
z_midplane = np.median(z)

tolerance = 0.5  # Adjust based on your dataset's spread

# Define filters for each midplane
x_midplane_filter = np.abs(x - x_midplane) <= tolerance
y_midplane_filter = np.abs(y - y_midplane) <= tolerance
z_midplane_filter = np.abs(z - z_midplane) <= tolerance

# Prepare for visualization

# Variables for plotting
variables = [K_pred,K_real,np.log10(np.abs(K_pred-K_real))]  # Use the scalar field for coloring
variable_names = ['K-Pred','K-Real','Relative Log Error','K-Real','K-Pred','K-Error']

# Visualization
for i, var in enumerate(variables):
    fig = plt.figure(figsize=(12, 8))  # Adjust size as needed

    for j in range(3):
        ax = fig.add_subplot(1, 3, j + 1)
        if j == 0:
            filter_mask = x_midplane_filter
            scatter_x, scatter_y = y[filter_mask], z[filter_mask]
            xlabel, ylabel = 'Y', 'Z'
        elif j == 1:
            filter_mask = y_midplane_filter
            scatter_x, scatter_y = x[filter_mask], z[filter_mask]
            xlabel, ylabel = 'X', 'Z'
        else:  # j == 2
            filter_mask = z_midplane_filter
            scatter_x, scatter_y = x[filter_mask], y[filter_mask]
            xlabel, ylabel = 'X', 'Y' 

        # Use the scalar field for coloring
        scatter = ax.scatter(scatter_x, scatter_y, c=var[filter_mask], cmap=cmap, s=2)
        ax.set_title(f'{variable_names[i]} -{["X", "Y", "Z"][j]}')
        #ax.set_xlabel(xlabel)
        #ax.set_ylabel(ylabel)
        ax.axis('equal')
        ax.axis('off')  # Change to 'off' if you prefer no axis

        # Adding a colorbar to each subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)

plt.tight_layout()
plt.savefig(images_path+'Final_results.png')
plt.show()


# %%
K_pred = np.array(K_pred)
K_real = np.array(K_real)

# Compute the intersection and union
intersection = np.sum((K_pred == 1) & (K_real == 1))
union = np.sum((K_pred == 1) | (K_real == 1))

# Compute IoU
iou = intersection / union if union != 0 else 0

print(f'Intersection over Union (IoU) for K: {iou:.4f}')


# %%


# %%



