# Libraries
import numpy as np
from instant_aiv.manage.plots import *
from instant_aiv.models.metrics import *
import cv2
import os
import tqdm
import h5py
import matplotlib.tri as tri
import jax.numpy as jnp
import jax
import importlib.util
import logging 

def create_save_name(Eqn, Mode, use_RBA=False, Mod_MLP=False, Adaptive_AF=False, Weight_Norm=False,resample=False,batch_size=1):
    save_name = f'{Eqn}:{Mode}'
    if use_RBA:
        save_name += 'RBA'
    if Mod_MLP:
        save_name += 'mMLP'
    if Adaptive_AF:
        save_name += 'AF'
    if Weight_Norm:
        save_name += 'WN'
    if resample:
        save_name += f'_BS:{batch_size}'
    return save_name

def create_save_nameSA(Eqn, Mode, use_RBA=False, Mod_MLP=False, Adaptive_AF=False, Weight_Norm=False,resample=False,batch_size=1):
    save_name = f'{Eqn}:{Mode}'
    if use_RBA:
        save_name += 'SA'
    if Mod_MLP:
        save_name += 'mMLP'
    if Adaptive_AF:
        save_name += 'AF'
    if Weight_Norm:
        save_name += 'WN'
    if resample:
        save_name += f'_BS:{batch_size}'
    return save_name
    
    
def create_and_return_directories(save_path, dataset_name, subdirectories):
    # Base directory
    result_path = os.path.join(save_path, dataset_name)

    # Creating subdirectories and storing their paths
    paths = {}
    for subdir in subdirectories:
        path = os.path.join(result_path, subdir+'/')
        os.makedirs(path, exist_ok=True)
        paths[subdir] = path

        # Printing the paths
        print(f"The {subdir.lower().replace('_', ' ')} path is: {path}")

    return paths



#Dataloader
class Dataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, sample_function) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.sample_function = sample_function

    def __iter__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]
        return iter(data)
    def __getitem__(self):
        batch_idx_list = self.sample_function(self.dataset, self.batch_size)
        data = [self.dataset[batch_idx] for batch_idx in batch_idx_list]

# Random Sampling
def random_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.random.permutation(N),
        np.arange(batch_size, N, batch_size),
    )
# Uniform Sampling
def get_batch(dataset: Dataset, batch_size: int):
    N = len(dataset)
    return np.split(
        np.arange(N),
        np.arange(batch_size, N, batch_size),
    )



#Rescale boundaries
def rescale_bcs(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    v_mbcs=mBCs[:,4,0:0+1].flatten()[:,None]
    u_mbcs=mBCs[:,5,0:0+1].flatten()[:,None]
    w_mbcs=mBCs[:,6,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        v_mbcs=mBCs[:,4,t_idx:t_idx+1].flatten()[:,None]
        u_mbcs=mBCs[:,5,t_idx:t_idx+1].flatten()[:,None]
        w_mbcs=mBCs[:,6,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points,u_mbcs,v_mbcs,w_mbcs))
    return mbcs_f,[dx,dy,dz]

#Rescale boundaries
def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def normalizeZ(X,X_mean,X_std):
    H = (X- X_mean)/X_std
    return H    

def normalize_between(X,X_min,X_max,lb=-1,ub=1):
    X = (ub-lb) * (X- X_min) / (X_max - X_min)+lb 
    return X    

def identity(X,X_min,X_max):
    return X


def make_video(image_folder, video_name, fps):
    video_name=image_folder+str(fps)+'fps-'+video_name
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort the images by name
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for i in tqdm.tqdm(range(len(images))):
        image=f'{image_folder}{images[i]}'
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()

def rescale_points(mBCs,BCs_ref,t_idx=0):
    t_mbcs=mBCs[:,0,0:0+1].flatten()[:,None]
    y_mbcs=mBCs[:,1,0:0+1].flatten()[:,None]
    x_mbcs=mBCs[:,2,0:0+1].flatten()[:,None]
    z_mbcs=mBCs[:,3,0:0+1].flatten()[:,None]
    X0_mbcs=np.hstack((x_mbcs,
                       y_mbcs,
                       z_mbcs,))

    #centroids
    cx,cy,cz=centroid(BCs_ref)
    mcx,mcy,mcz=centroid(X0_mbcs)
    bcs_centred=MoveScale3D(BCs_ref,dx=-cx,dy=-cy,dz=-cz,sx=1,sy=1,sz=1)
    mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1)
    #scales
    sc_bcs=1/np.max(np.abs(bcs_centred))
    sc_mbcs=1/np.max(np.abs(mbcs_centred))
    bcs_scaled=MoveScale3D(bcs_centred,dx=0,dy=0,dz=0,sx=sc_bcs,sy=sc_bcs,sz=sc_bcs)
    mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
    _=centroid(bcs_scaled)
    _=centroid(mbcs_scaled)
    dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
    print(f'The difference is {dx,dy,dz}')
    mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
    _=centroid(mbcs_scaled2)
    #
    if t_idx >0:
        t_mbcs=mBCs[:,0,t_idx:t_idx+1].flatten()[:,None]
        y_mbcs=mBCs[:,1,t_idx:t_idx+1].flatten()[:,None]
        x_mbcs=mBCs[:,2,t_idx:t_idx+1].flatten()[:,None]
        z_mbcs=mBCs[:,3,t_idx:t_idx+1].flatten()[:,None]
        X0_mbcs=np.hstack((x_mbcs,
                           y_mbcs,
                           z_mbcs,))
        mcx,mcy,mcz=centroid(X0_mbcs)
        mbcs_centred=MoveScale3D(X0_mbcs,dx=-mcx,dy=-mcy,dz=-mcz,sx=1,sy=1,sz=1) 
        sc_mbcs=1/np.max(np.abs(mbcs_centred))
        mbcs_scaled=MoveScale3D(mbcs_centred,dx=0,dy=0,dz=0,sx=sc_mbcs,sy=sc_mbcs,sz=sc_mbcs)
        _=centroid(mbcs_scaled)   
        mbcs_scaled2=MoveScale3D(mbcs_scaled,dx=dx,dy=dy,dz=dz)
        dx,dy,dz=np.max(bcs_scaled,axis=0)-np.max(mbcs_scaled,axis=0)
        print(f'The difference is {dx,dy,dz}')
        _=centroid(mbcs_scaled2)   
    #Scale to ref BCs
    scaled2_points=MoveScale3D(mbcs_scaled2,sx=1/sc_bcs,sy=1/sc_bcs,sz=1/sc_bcs)
    out_points=MoveScale3D(scaled2_points,dx=cx,dy=cy,dz=cz,sx=1,sy=1,sz=1)
    _=centroid(out_points)
    mbcs_f=np.hstack((t_mbcs,out_points))
    return mbcs_f,[dx,dy,dz]


def extract_arrays_from_params(params_test,Use_ResNet=False):
    # Extract arrays from the 'params' dictionary
    params_arrays = []
    if Use_ResNet:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))
    else:
        for param_dict in params_test['params']:
            for key in ['W', 'b', 'g']:
                if key in param_dict:
                    params_arrays.append(np.array(param_dict[key]))

    # Extract arrays from the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_arrays = [np.array(params_test['mMLP'][0][key]) for key in mMLP_keys if key in params_test['mMLP'][0]]

    # Extract arrays from the 'AdaptiveAF' dictionary
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    AdaptiveAF_arrays = []
    for adaptive_dict in params_test['AdaptiveAF']:
        for key in AdaptiveAF_keys:
            AdaptiveAF_arrays.append(np.array(adaptive_dict[key]))

    # Combine all extracted arrays
    all_arrays = params_arrays + mMLP_arrays + AdaptiveAF_arrays
    return all_arrays

def reconstruct_params(numpy_arrays_list, params_length, params_length_AF,Use_ResNet=False):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    # Reconstruct the 'params' dictionary
    params_dicts = []
    if Use_ResNet:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)
    else:
        for _ in range(params_length):
            param_dict = {}
            for key in ['W', 'b', 'g']:
                if pointer < len(numpy_arrays_list):
                    param_dict[key] = numpy_arrays_list[pointer]
                    pointer += 1
            params_dicts.append(param_dict)

    # Reconstruct the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_dict = {}
    for key in mMLP_keys:
        if pointer < len(numpy_arrays_list):
            mMLP_dict[key] = numpy_arrays_list[pointer]
            pointer += 1

    # Reconstruct the 'AdaptiveAF' dictionary
    AdaptiveAF_dicts = []
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    for _ in range(params_length_AF):
        adaptive_dict = {}
        for key in AdaptiveAF_keys:
            if pointer < len(numpy_arrays_list):
                adaptive_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        AdaptiveAF_dicts.append(adaptive_dict)

    # Combine all reconstructed dictionaries
    reconstructed_params_test = {
        'AdaptiveAF': AdaptiveAF_dicts,
        'mMLP': [mMLP_dict],
        'params': params_dicts,
    }
    
    return reconstructed_params_test

def reconstruct_params_ResNet(numpy_arrays_list, params_length, params_length_AF):
    # For ease, I'll use a pointer instead of popping items
    pointer = 0
    
    # Reconstruct the 'params' dictionary
    params_dicts = []
    for _ in range(params_length):
        param_dict = {}
        for key in ['W', 'b', 'g','W2', 'b2', 'g2','alpha']:
            if pointer < len(numpy_arrays_list):
                param_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        params_dicts.append(param_dict)

    # Reconstruct the 'mMLP' dictionary
    mMLP_keys = ['U1', 'U2', 'b1', 'b2', 'g1', 'g2']
    mMLP_dict = {}
    for key in mMLP_keys:
        if pointer < len(numpy_arrays_list):
            mMLP_dict[key] = numpy_arrays_list[pointer]
            pointer += 1

    # Reconstruct the 'AdaptiveAF' dictionary
    AdaptiveAF_dicts = []
    AdaptiveAF_keys = ['a0', 'a1', 'a2', 'f0', 'f1', 'f2']
    for _ in range(params_length_AF):
        adaptive_dict = {}
        for key in AdaptiveAF_keys:
            if pointer < len(numpy_arrays_list):
                adaptive_dict[key] = numpy_arrays_list[pointer]
                pointer += 1
        AdaptiveAF_dicts.append(adaptive_dict)

    # Combine all reconstructed dictionaries
    reconstructed_params_test = {
        'AdaptiveAF': AdaptiveAF_dicts,
        'mMLP': [mMLP_dict],
        'params': params_dicts,
    }
    
    return reconstructed_params_test

# Saving All params
def save_all_params(All_params,all_params_path):
    with h5py.File(all_params_path, 'w') as hf:
        for i, inner_list in tqdm.tqdm(enumerate(All_params)):
            group = hf.create_group(f"list_{i}")
            for j, arr in enumerate(inner_list):
                group.create_dataset(f"array_{j}", data=arr)


# Reading All params
def read_all_params(all_params_path):
    loaded_All_params = []
    with h5py.File(all_params_path, 'r') as hf:
        for key in tqdm.tqdm(sorted(hf.keys(), key=lambda x: int(x.split('_')[1]))):  # Ensure keys are processed in order
            inner_list = []
            for sub_key in sorted(hf[key].keys(), key=lambda x: int(x.split('_')[1])):
                data = hf[key][sub_key]
                if data.shape == ():  # Check if scalar
                    inner_list.append(data[()])
                else:
                    inner_list.append(data[:])
            loaded_All_params.append(inner_list)
    return loaded_All_params



def process_uneven_data(X,Y,V):
    n_x=np.unique(X).shape[0]
    n_y=np.unique(Y).shape[0]
    xi = np.linspace(np.min(X), np.max(X), n_x)
    yi = np.linspace(np.min(Y), np.max(Y), n_y)
    triang = tri.Triangulation(X, Y)
    interpolator = tri.LinearTriInterpolator(triang, V)
    x, y = np.meshgrid(xi, yi)
    Vi = interpolator(x, y)
    return x,y,Vi

def sample_points(batch_sizes, dataset):
    """
    This function randomly samples indices based on batch_sizes.
    
    Args:
    - batch_sizes: Dictionary of batch sizes for different data types.

    Returns:
    - Dictionary of sampled indices for each data type.
    """
    return {key: np.random.choice(len(dataset[key]), batch_sizes[key]) for key in batch_sizes}



def sample_points_pdf(subkey, batch_sizes, dataset, lambdas,k=1,c=0.5):
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices

          
    
def sample_points_PDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=lambdas_key/lambdas_key.sum())
    return batch_indices

    
def sample_points_jax(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],))
    return batch_indices
def sample_all(it, batch_sizes, dataset, lambdas=[],k=1,c=0.5):
    batch_indices = {}
    for key in batch_sizes:
        batch_indices[key] = jnp.arange(len(dataset[key]))
    return batch_indices


def sample_points_softPDF(it, batch_sizes, dataset, lambdas,k=1,c=0.5):
    key = jax.random.PRNGKey(it)
    key, subkey = jax.random.split(key)  
    batch_indices = {}
    for key in batch_sizes:
        lambdas_key = (jnp.sum(lambdas[key], axis=1))**k
        lambdas_key = lambdas_key / lambdas_key.mean()+c
        batch_indices[key] = jax.random.choice(subkey, len(dataset[key]), shape=(batch_sizes[key],), p=jax.nn.softmax(lambdas_key))
    return batch_indices

def generate_random_matrix(m, d):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    B_standard = jax.random.normal(subkey, (m, d))
    return B_standard 
    
def Encode_Fourier(X,M,N):
    t=X[0]
    x=X[1]
    y=X[2]
    P_x=2
    P_y=2
    n_num = jnp.arange(1, N+1)
    m_num = jnp.arange(1, M+1)
    n, m = jnp.meshgrid(n_num, m_num)
    n=n.flatten()
    m=m.flatten()
    w_x = 2.0 * jnp.pi / P_x
    w_y = 2.0 * jnp.pi / P_y    

    out = jnp.hstack([t,
                      x,
                      y,
                      jnp.cos(n* w_x * x)  * jnp.cos(m * w_y * y),
                      jnp.cos(n * w_x * x) * jnp.sin(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.cos(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.sin(m * w_y * y)])
    return out
def Encode_Fourier2(X, B, s=10):
    B = s * B
    out = jnp.multiply(B, X).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier_time(X, B, s=10):
    out =  s *jnp.multiply(B, X[1:]).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier_time_1(X, M1,M2):
    X_min,X_max=M1
    B,s=M2
    X = 2 * (X- X_min) / (X_max - X_min)-1 
    out =  s *jnp.multiply(B, X[1:]).flatten()
    combined_out = jnp.concatenate([X.flatten(),
                                    jnp.sin(out),
                                    jnp.cos(out)])
    return combined_out
def Encode_Fourier0(X,M,N):
    x=X[0]
    y=X[1]
    P_x=2
    P_y=2
    n_num = jnp.arange(1, N+1)
    m_num = jnp.arange(1, M+1)
    n, m = jnp.meshgrid(n_num, m_num)
    n=n.flatten()
    m=m.flatten()
    w_x = 2.0 * jnp.pi / P_x
    w_y = 2.0 * jnp.pi / P_y    

    out = jnp.hstack([jnp.cos(n* w_x * x)  * jnp.cos(m * w_y * y),
                      jnp.cos(n * w_x * x) * jnp.sin(m * w_y * y),
                      jnp.sin(n * w_x * x) * jnp.cos(m * w_y * y),])
    return out



def Encode_Chebysev_5(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),
                        ])
    return out
def Encode_hybrid(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        jnp.sin(10*out),jnp.cos(10*out),
                        jnp.exp(out),jnp.exp(-out),
                        jnp.sin(10*out)*jnp.cos(out),
                        jnp.sin(out)*jnp.cos(10*out),
                        ])
    return out
def Encode_Chebysev_10(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        T7(out),T8(out),T9(out),
                        T10(out),
                        ])
    return out
def Encode_Chebysev_12(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    out = X.flatten()
    out = jnp.concatenate([
                        T1(out),T2(out),T3(out),
                        T4(out),T5(out),T6(out),
                        T7(out),T8(out),T9(out),
                        T10(out),T11(out),T12(out),
                        ])
    return out
def Encode_Chebysev_time(X,M1,M2):
    X = 2 * (X- M2) / (M1 - M2)-1 
    X=X.flatten()
    out =X[1:] 
    out = jnp.concatenate([
                        X,T2(out),T3(out),
                        T4(out),T5(out)
                        ])
    return out
# Activation functions mapping
ACTIVATION_FUNCTIONS = {
    'sin': jnp.sin,
    'tanh': jnp.tanh,
    'tanh_08': lambda x: 0.8*jnp.tanh(x),
    'swish': lambda x: x * jax.nn.sigmoid(x),
    'leaky_relu': lambda x: jnp.where(x > 0, x, 0.01 * x),
    'custom': lambda x: jnp.where(x < 0, -1, jnp.where(x > 2, 1, 0.5*(x)**2-1)),
    'sigmoid':lambda x: jax.nn.sigmoid(x),
    'sigmoid_11':lambda x: 2/(1+jnp.exp(-x*1.1**2))-1

}

# Normalization functions and metrics mapping
NORMALIZATION_FUNCTIONS = {
    'plusminus1': {
        'fn': normalize_between,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'normal': {
        'fn': normalizeZ,
        'metric1': lambda x: x.mean(0, keepdims=False),
        'metric2': lambda x: x.std(0, keepdims=False)
    },
    'identity': {
        'fn': identity,
        'metric1': lambda x: 0,
        'metric2': lambda x: 0
    },
    'fourier': {
        'fn': Encode_Fourier2,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourierfull': {
        'fn': Encode_Fourier0,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourier_time': {
        'fn': Encode_Fourier_time,
        'metric1': lambda x: N,
        'metric2': lambda x: M
    },
    'fourier_time_1': {
        'fn': Encode_Fourier_time_1,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev5': {
        'fn': Encode_Chebysev_5,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev10': {
        'fn': Encode_Chebysev_10,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev12': {
        'fn': Encode_Chebysev_12,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'chebyshev_time': {
        'fn': Encode_Chebysev_time,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
    'hybrid': {
        'fn': Encode_hybrid,
        'metric1': lambda x: x.min(0, keepdims=False),
        'metric2': lambda x: x.max(0, keepdims=False)
    },
}

# Loss metric mapping
ERROR_FUNCTIONS = {
    'l2': MSE,
    'l1': MAE
}



def filter_Magnitude(BCs_frame,row=7,T_max=0.7,T_min=0.49):
    T  =BCs_frame[:,row]
    upper_limit = T_max
    lower_limit = T_min
    idx1=np.argwhere(T<upper_limit)
    idx2=np.argwhere(T>lower_limit)
    idxT=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idxT]
    return BCs_framef
def filter_Magnitude_inverse(BCs_frame,row=7,T_max=0.7,T_min=0.49):
    T  =BCs_frame[:,row]
    upper_limit = T_max
    lower_limit = T_min
    idx1=np.argwhere(T>upper_limit)
    idx2=np.argwhere(T<lower_limit)
    idxT=np.union1d(idx1,idx2)
    BCs_framef=BCs_frame[idxT]
    return BCs_framef

def filter_Z(BCs_frame,row=7,permissibility=3):
    u  =BCs_frame[:,row]
    #Z score FILTERING
    #Filter u
    mean_u = np.nanmean(u)
    std_u  = np.nanstd(u)
    upper_limit = mean_u + permissibility*std_u
    lower_limit = mean_u - permissibility*std_u
    idx1=np.argwhere(u<upper_limit)
    idx2=np.argwhere(u>lower_limit)
    idx=np.intersect1d(idx1,idx2)
    BCs_framef=BCs_frame[idx]
    return BCs_framef

# Chebyshev's Polynomials
def T0(x):
    return x*0+1
def T1(x):
    return x
def T2(x):
    return 2*x**2-1
def T3(x):
    return 4*x**3-3*x
def T4(x):
    return 8*x**4-8*x**2+1
def T5(x):
    return 16*x**5-20*x**3+5*x
def T6(x):
    return 32*x**6-48*x**4+18*x**2-1
def T7(x):
    return 64*x**7-112*x**5+56*x**3-7*x
def T8(x):
    return 128*x**8-256*x**6+160*x**4-32*x**2+1
def T9(x):
    return 256*x**9-576*x**7+432*x**5-120*x**3+9*x
def T10(x):
    return 512*x**10-1280*x**8+1120*x**6-400*x**4+50*x**2-1
def T11(x):
    return 1024*x**11-2816*x**9+2816*x**7-1232*x**5+220*x**3-11*x


def find_package_directory(package_name: str = 'instant_aiv') -> str | None:
    """
    Finds and returns the installation directory path of a given package (simplified).

    Args:
        package_name: The name of the package. Defaults to 'instant_aiv'.

    Returns:
        The absolute path to the root directory of the package if found,
        otherwise returns None. Returns None if the package is not found
        or doesn't have a standard location attribute.
    """
    # Attempt to find the package specification
    spec = importlib.util.find_spec(package_name)

    # Check if the specification was found
    if spec:
        if spec.origin:
            package_dir =  os.path.dirname(os.path.dirname(os.path.dirname(spec.origin)))
            return package_dir+'/'
        else:
            print('Error. Please define the saving path manually')

    return None

