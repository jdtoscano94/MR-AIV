 # Libraries

import numpy as np
from jax import jit, vmap, grad
import jax.numpy as jnp
from jax.nn import sigmoid
from typing import Tuple

from typing import Tuple, List, Dict, Sequence
import h5py
from instant_aiv.models.metrics import *
from instant_aiv.manage.dataloader import *
import optax

#Initialization
from typing import Tuple

def glorot_normal(in_dim: int, out_dim: int) -> jnp.ndarray:
    glorot_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return jnp.array(np.random.normal(loc=0.0, scale=glorot_stddev, size=(in_dim, out_dim)))


def init_params(layers: List[int], initialization_type: str = 'xavier',Network_type: str='mlp',degree: int =5,Use_ResNet: bool =False) -> dict: 
    def init_adaptive_params():
        F = 0.1 * jnp.ones(3 * len(layers) - 1)
        A = 0.1 * jnp.ones(3 * len(layers) - 1)
        return [{"a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                 "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]} 
                for i in range(len(layers) - 1)]
    #Define Models:
    def init_layer_mlp(in_dim, out_dim):
        if initialization_type == 'xavier':
            W = glorot_normal(in_dim, out_dim)
        elif initialization_type == 'normal':
            W = jnp.array(np.random.normal(size=(in_dim, out_dim)))
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}
    def init_layer_ResNet(in_dim, out_dim):
        if initialization_type == 'xavier':
            W1 = glorot_normal(in_dim, out_dim)
            W2 = glorot_normal(out_dim, out_dim)
            W3 = glorot_normal(out_dim, out_dim)
        elif initialization_type == 'normal':
            W1 = jnp.array(np.random.normal(size=(in_dim, out_dim)))
            W2 = jnp.array(np.random.normal(size=(out_dim, out_dim)))
            W3 = jnp.array(np.random.normal(size=(out_dim, out_dim)))
        b1  = jnp.zeros(out_dim)
        b2  = jnp.zeros(out_dim)
        b3  = jnp.zeros(out_dim)
        g1  = jnp.ones(out_dim)
        g2  = jnp.ones(out_dim)
        g3  = jnp.ones(out_dim)
        alpha=0.0
        return {"W": W1, "b": b1, "g": g1,
                "W2": W2, "b2": b2, "g2": g2,
                'alpha':alpha}
    def init_layer_kan(in_dim, out_dim,degree=degree):
        std=1 / (in_dim * (degree + 1))
        W =jnp.array(np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim,degree+1)))
        b = jnp.zeros(out_dim)
        g = jnp.ones(out_dim)
        return {"W": W, "b": b, "g": g}
    def init_layer_kan_ResNet(in_dim, out_dim,degree=degree):
        std=1 / (in_dim * (degree + 1))
        W1 =jnp.array(np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim,degree+1)))
        W2 =jnp.array(np.random.normal(loc=0.0, scale=std, size=(in_dim, out_dim,degree+1)))
        b1  = jnp.zeros(out_dim)
        b2  = jnp.zeros(out_dim)
        g1  = jnp.ones(out_dim)
        g2  = jnp.ones(out_dim)
        alpha=0.0
        return {"W": W1, "b": b1, "g": g1,
                "W2": W2, "b2": b2, "g2": g2,
                'alpha':alpha}
    #Select model
    if Network_type.lower()=='mlp':
        if Use_ResNet:
            init_layer_params=init_layer_ResNet
        else:
            init_layer_params=init_layer_mlp
    elif Network_type.lower()[:3]=='kan':
        if Use_ResNet:
            init_layer_params=init_layer_kan_ResNet
        else:
            init_layer_params=init_layer_kan
    else:
        print(f'Error: {Network_type.lower()} is not a valid option. The available options are:mlp and kan.')
    print(f'Initializing:{Network_type} parameters.')
    params = [init_layer_params(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    U1, b1, g1 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    U2, b2, g2 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    
    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]
    
    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }


                
def init_params_res(layers: List[int], initialization_type: str = 'xavier',Network_type: str='mlp',degree: int =5,Use_ResNet: bool =False) -> dict:
    def init_adaptive_params():
        F = 0.1 * jnp.ones(3 * len(layers) - 1)
        A = 0.1 * jnp.ones(3 * len(layers) - 1)
        return [{"a0": A[3*i], "a1": A[3*i + 1], "a2": A[3*i + 2],
                 "f0": F[3*i], "f1": F[3*i + 1], "f2": F[3*i + 2]} 
                for i in range(len(layers) - 1)]
                
    def init_layer_params(in_dim, out_dim):
        if initialization_type == 'xavier':
            W1 = glorot_normal(in_dim, out_dim)
            W2 = glorot_normal(out_dim, out_dim)
            W3 = glorot_normal(out_dim, out_dim)
        elif initialization_type == 'normal':
            W1 = jnp.array(np.random.normal(size=(in_dim, out_dim)))
            W2 = jnp.array(np.random.normal(size=(out_dim, out_dim)))
            W3 = jnp.array(np.random.normal(size=(out_dim, out_dim)))
        b1  = jnp.zeros(out_dim)
        b2  = jnp.zeros(out_dim)
        b3  = jnp.zeros(out_dim)
        g1  = jnp.ones(out_dim)
        g2  = jnp.ones(out_dim)
        g3  = jnp.ones(out_dim)
        alpha=0.0
        return {"W": W1, "b": b1, "g": g1,
                "W2": W2, "b2": b2, "g2": g2,
                'alpha':alpha}

    params = [init_layer_params(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
    
    U1, b1, g1 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    U2, b2, g2 = glorot_normal(layers[0], layers[1]), jnp.zeros(layers[1]), jnp.ones(layers[1])
    
    mMLP_params = [{"U1": U1, "b1": b1, "g1": g1, "U2": U2, "b2": b2, "g2": g2}]
    
    return {
        'params': params,
        'AdaptiveAF': init_adaptive_params(),
        'mMLP': mMLP_params
    }


def init_params_dict(layer_dict, initialization,Use_ResNet=False,Network_type='mlp',degree=5):
    print(f'You selected: Network {Network_type} with degree(if KAN) {degree}, initialization {initialization},Use_ResNet {Use_ResNet}')
    if Network_type.lower()=='mlp':
        if Use_ResNet:
            init_function=init_params_res
        else:
            init_function=init_params
    elif Network_type[:3].lower()=='kan':
            init_function=init_params
    initialized_params = {}
    for key, layer_structure in layer_dict.items():
        # Initialize parameters for each key
        params = init_function(layer_structure, 
                               initialization_type=initialization.lower(),
                               Network_type=Network_type,
                               degree=degree,
                               Use_ResNet=Use_ResNet)
        
        # Store in the dictionary
        initialized_params[key] = params

    return initialized_params




def FCN(params, X_in, M1, M2, activation_fn, norm_fn):
    """
    Fully Connected Network (FCN) with a given normalization and activation function.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation_fn: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    params_N = params["params"]
    inputs = norm_fn(X_in, M1, M2)
    
    for layer in params_N[:-1]:
        outputs = activation_fn(jnp.dot(inputs, layer["W"]) + layer["b"])
        inputs = outputs
    
    W = params_N[-1]["W"]
    b = params_N[-1]["b"]
    outputs = jnp.dot(inputs, W) + b
    
    return outputs


def FCN_WN(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """
    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    
    # Iterate through the layers
    for i, layer_params in enumerate(params["params"]):
        W, b, g = layer_params["W"], layer_params["b"], layer_params["g"]
        
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        
        # Linear transformation
        H = g * jnp.matmul(H, V) + b
        
        # Apply activation function for all layers except the last one
        if i != len(params["params"]) - 1:
            H = activation(H)
    
    return H


def FCN_WN_MMLP(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Weight Normalization and Modified MLP (MMLP) transformations.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and MMLP parameters.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Normalize the input
    H = norm_fn(X_in, M1, M2)

    # Unpack MMLP parameters and apply weight normalization
    mMLP_params = params["mMLP"][0]
    U1, U2, b1, b2, g1, g2 = (mMLP_params[key] for key in ["U1", "U2", "b1", "b2", "g1", "g2"])
    U1_norm, U2_norm = U1 / jnp.linalg.norm(U1, axis=0, keepdims=True), U2 / jnp.linalg.norm(U2, axis=0, keepdims=True)

    # Calculate U and V transformations
    U = activation(g1 * jnp.dot(H, U1_norm) + b1)
    V = activation(g2 * jnp.dot(H, U2_norm) + b2)

    # Iterate through the layers
    for idx, layer in enumerate(params["params"][:-1]):
        W, b, g = layer["W"], layer["b"], layer["g"]

        # Apply weight normalization
        W_norm = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        
        # Compute activations and apply MMLP combination step
        H = activation(g * jnp.dot(H, W_norm) + b)
        H = jnp.multiply(H, U) + jnp.multiply(1 - H, V)

    # Process the last layer
    W, b, g = params["params"][-1]["W"], params["params"][-1]["b"], params["params"][-1]["g"]
    W_norm = W / jnp.linalg.norm(W, axis=0, keepdims=True)
    H = g * jnp.dot(H, W_norm) + b

    return H



def FCN_MMLP(params, X_in, M1, M2, activation, norm_fn):
    """
    Fully Connected Network (FCN) with Modified MLP (MMLP) transformations.
    
    Parameters:
    - params: Dictionary containing parameters of the neural network layers and MMLP parameters.
    - X_in: Input tensor to the network.
    - M1, M2: Parameters for normalization function.
    - activation: Callable activation function.
    - norm_fn: Callable normalization function.
    
    Returns:
    - Output tensor from the network.
    """

    # Normalize the input
    inputs = norm_fn(X_in, M1, M2)

    # Unpack MMLP parameters
    mMLP_params = params["mMLP"][0]
    U1, U2, b1, b2 = mMLP_params["U1"], mMLP_params["U2"], mMLP_params["b1"], mMLP_params["b2"]

    # Calculate U and V transformations
    U = activation(jnp.dot(inputs, U1) + b1)
    V = activation(jnp.dot(inputs, U2) + b2)

    # Iterate through all layers except the last
    for layer in params["params"][:-1]:
        W, b = layer["W"], layer["b"]
        
        # Compute activations
        act_values = activation(jnp.dot(inputs, W) + b)
        
        # MMLP combination step
        inputs = jnp.multiply(act_values, U) + jnp.multiply(1 - act_values, V)

    # Compute output from the last layer
    W, b = params["params"][-1]["W"], params["params"][-1]["b"]
    outputs = jnp.dot(inputs, W) + b

    return outputs


# Save Resuls
def save_list(Loss,path,name='loss-'):
    filename=path+name+".npy"
    np.save(filename, np.array(Loss))
    
def save_MLP_params(params: List[Dict[str, np.ndarray]], save_path,WN=False,Mod_MLP=False):
    with h5py.File(save_path, "w") as f:
        for layer_idx, layer_params in enumerate(params):
            layer_group = f.create_group(f"Layer_{layer_idx/2.0:.2f}")
            if WN:
                if Mod_MLP:
                    W, b, g= layer_params.values()
                    layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                    layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)
                    layer_group.create_dataset("g", shape=g.shape, dtype=np.float32, data=g)
                else:
                    W, b, g= layer_params.values()
                    layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                    layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)
                    layer_group.create_dataset("g", shape=g.shape, dtype=np.float32, data=g)
            else:
                W, b = layer_params.values()
                layer_group.create_dataset("W", shape=W.shape, dtype=np.float32, data=W)
                layer_group.create_dataset("b", shape=b.shape, dtype=np.float32, data=b)  
            
def read_params(filename,WN=False):
    data = h5py.File(filename, 'r')
    recover_params=[]
    for layer in data.keys() :
        if WN:
            stored={'W':data[layer]['W'][:],'b': data[layer]['b'][:],'g': data[layer]['g'][:]}
        else:
            stored={'W':data[layer]['W'][:],'b': data[layer]['b'][:]}
        recover_params.append(stored)
    return recover_params




def select_model(WN=False, Mod_MLP=False,Use_ResNet=False,Adaptive=False,Light=False,Network_type='mlp',degree=5):
    if Network_type.lower()=='mlp':
        model_map = {
            (True, True,False, False, False): FCN_WN_MMLP,
            (True, False,False, False, False): FCN_WN,
            (False, True,False, False, False): FCN_MMLP,
            (False, False,False, False, False): FCN,
            (False, False,True, False, False): ResNet,
            (False, False,True, False, True): ResNet_light,
            (True, False,True, False, False): WN_ResNet,
            (True, False,True, False, True): WN_ResNet_light,
            (True, False,True, True, False): WN_ResNet_adaptive,
        }
        return model_map[(WN, Mod_MLP,Use_ResNet,Adaptive,Light)]
    elif Network_type.lower()=='kan':
        if degree==3:
            return KAN_Net3
        elif degree==5:
            if Use_ResNet:
                return KAN5_ResNet
            else:
                return KAN_Net5
        elif degree==7:
            return KAN_Net7
        elif degree==9:
            return KAN_Net9
        elif degree==11:
            return KAN_Net11
        else:
            return KAN_Net
    elif Network_type.lower()=='kan_theta':
            return KAN_Net_theta

def initialize_optimizer(lr0, decay_rate, lrf, decay_step, T_e,optimizer_type='Adam',weight_decay=1e-5):
    print('Optimizer',optimizer_type.lower())
    if optimizer_type.lower()=='adam':
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adam(lr0), decay_step
        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            return optax.adam(optax.exponential_decay(lr0, decay_step, decay_rate,)),decay_step
    elif optimizer_type.lower()=='adamw':
        print('Weight decay:',weight_decay)
        if decay_rate == 0 or lrf == lr0:
            print('No decay')
            return optax.adamw(learning_rate=lr0, weight_decay=weight_decay), decay_step

        else:
            if decay_step == 0:
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.adamw(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step
    elif optimizer_type.lower()=='lion':
        if decay_rate == 0 or lrf == lr0:
            weight_decay=weight_decay*3
            print('No decay')
            return optax.lion(learning_rate=lr0, weight_decay=weight_decay), decay_step
        else:
            if decay_step == 0:
                weight_decay=weight_decay*3
                decay_step = T_e * np.log(decay_rate) / np.log(lrf / lr0)
            print(f'The decay step will be {decay_step}')
            # Use adamw with the specified learning rate schedule
            return optax.lion(optax.exponential_decay(lr0, decay_step, decay_rate), weight_decay=weight_decay), decay_step

def load_params_dict(result_path, dataset_name, layer_dict, initialization, type='Test',Use_ResNet=False):
    loaded_params = {}
    for key in layer_dict.keys():
        # Construct the file path
        file_path = f"{result_path}{dataset_name}-{type}_params_{key}.h5"
        
        # Read parameters from the file
        raw_params = read_all_params(file_path)[0]

        # Initialize a test parameter set for getting lengths
        test_params = init_params(layer_dict[key], initialization_type=initialization.lower()) 
        params_length = len(test_params['params'])
        params_length_AF = len(test_params['AdaptiveAF'])

        # Reconstruct the parameters
        reconstructed_params = reconstruct_params(raw_params, params_length, params_length_AF,Use_ResNet)

        # Store in the dictionary
        loaded_params[key] = reconstructed_params

    return loaded_params

def save_params_dict(params,result_path,dataset_name,type='Test',Use_ResNet=False):
    # Assuming 'result_path' and 'dataset_name' are defined elsewhere in your code
    for key, params in params.items():
        # Construct the file name for each set of parameters
        output_path = f"{result_path}{dataset_name}-{type}_params_{key}.h5"
        
        # Extract arrays from params
        params_to_save = [extract_arrays_from_params(params,Use_ResNet)]
        # Save the parameters
        save_all_params(params_to_save, output_path)
        print(f'Params {key} have been saved!')


def transfer_params(params_s,params_t,levels=[0,1,2]):
    for level in levels:
        params_t['params'][level]=params_s['params'][level]
    return params_t
def L2_regularization(params,beta=0.001):
    Loss_L2=0
    for param in params:
        Loss_L2=jnp.mean(param['W']**2)+Loss_L2
    return beta*Loss_L2
def L1_regularization(params,beta=0.001):
    Loss_L2=0
    for param in params:
        Loss_L2=jnp.abs(jnp.mean(param['W']))+Loss_L2
    return beta*Loss_L2
def L2_regularization_ResNet(params,beta=0.001):
    Loss_L2=0
    for param in params:
        Loss_L2=jnp.sum(param['W']**2)+jnp.sum(param['W2']**2)+Loss_L2
    return beta*Loss_L2
# RESNET ARCHITECTURES
def ResNet(params, X_in, M1, M2, activation_fn, norm_fn):
    def linear_layer(H,layer_params,W_key="W",b_key="b"):
        W, b= layer_params[W_key], layer_params[b_key]
        H= jnp.dot(H,W) + b
        return H          
    params_N = params["params"]
    inputs = norm_fn(X_in, M1, M2)
    layer_params=params_N[0]
    inputs=activation_fn(linear_layer(inputs,layer_params,W_key="W",b_key="b"))
    for ly in range(1,len(params_N[:-1])):
        layer_params=params_N[ly]
        g= activation_fn(linear_layer(inputs,layer_params,W_key="W",b_key="b"))
        h= linear_layer(g,layer_params,W_key="W2",b_key="b2")
        inputs= activation_fn(h+inputs)
    outputs = jnp.dot(inputs, params_N[-1]["W"]) + params_N[-1]["b"]
   
    return outputs
def ResNet_light(params, X_in, M1, M2, activation_fn, norm_fn):
    def linear_layer(H,layer_params,W_key="W",b_key="b"):
        W, b= layer_params[W_key], layer_params[b_key]
        H= jnp.dot(H,W) + b
        return H          
    params_N = params["params"]
    inputs = norm_fn(X_in, M1, M2)
    layer_params=params_N[0]
    inputs=activation_fn(linear_layer(inputs,layer_params,W_key="W",b_key="b"))
    inputs0=jnp.copy(inputs)
    for ly in range(1,len(params_N[:-2])):
        layer_params=params_N[ly]
        g= activation_fn(linear_layer(inputs,layer_params,W_key="W",b_key="b"))
        h= linear_layer(g,layer_params,W_key="W2",b_key="b2")
        inputs= activation_fn(h)
    layer_params=params_N[-2]
    inputs=activation_fn(linear_layer(inputs+inputs0,layer_params,W_key="W",b_key="b"))
    layer_params=params_N[-1]
    outputs=activation_fn(linear_layer(inputs,layer_params,W_key="W",b_key="b"))
    
    return outputs
def WN_ResNet(params, X_in, M1, M2, activation, norm_fn):
    def linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"):
        W, b, g = layer_params[W_key], layer_params[b_key], layer_params[g_key]
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        # Linear transformation
        H = g * jnp.matmul(H, V) + b
        return H        
    params_N = params["params"]    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    # Encoder Layer
    H = activation(linear_layer(H,params_N[0],W_key="W",b_key="b",g_key="g"))
    # Iterate through the layers
    for i in range(1,len(params_N[:-1])):
        layer_params=params_N[i]
        F=activation(linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"))
        G=linear_layer(F,layer_params,W_key="W2",b_key="b2",g_key="g2")
        H=activation(G+H)
    H=linear_layer(H,params_N[-1],W_key="W",b_key="b",g_key="g")
    return H
def WN_ResNet_adaptive(params, X_in, M1, M2, activation, norm_fn):
    def linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"):
        W, b, g = layer_params[W_key], layer_params[b_key], layer_params[g_key]
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        # Linear transformation
        H = g * jnp.matmul(H, V) + b
        return H        
    params_N = params["params"]    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    # Encoder Layer
    H = activation(linear_layer(H,params_N[0],W_key="W",b_key="b",g_key="g"))
    # Iterate through the layers
    for i in range(1,len(params_N[:-1])):
        layer_params=params_N[i]
        F=activation(linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"))
        G=linear_layer(F,layer_params,W_key="W2",b_key="b2",g_key="g2")
        H=activation(layer_params["alpha"]*G+(1-layer_params["alpha"])*H)
    H=linear_layer(H,params_N[-1],W_key="W",b_key="b",g_key="g")
    return H

def WN_ResNet_light(params, X_in, M1, M2, activation, norm_fn):
    def linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"):
        W, b, g = layer_params[W_key], layer_params[b_key], layer_params[g_key]
        # Weight normalization
        V = W / jnp.linalg.norm(W, axis=0, keepdims=True)
        # Linear transformation
        H = g * jnp.matmul(H, V) + b
        return H        
    params_N = params["params"]    
    # Normalize the input
    H = norm_fn(X_in, M1, M2)
    # Encoder Layer
    H = activation(linear_layer(H,params_N[0],W_key="W",b_key="b",g_key="g"))
    H0=jnp.copy(H)
    # Iterate through the layers
    for i in range(1,len(params_N[:-2])):
        layer_params=params_N[i]
        F=activation(linear_layer(H,layer_params,W_key="W",b_key="b",g_key="g"))
        G=linear_layer(F,layer_params,W_key="W2",b_key="b2",g_key="g2")
        H=activation(G)
    H=activation(linear_layer(H+H0,params_N[-2],W_key="W",b_key="b",g_key="g"))
    H=linear_layer(H,params_N[-1],W_key="W",b_key="b",g_key="g")
    return H

def KAN_Net3(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer3(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer3(x,layer_params)
    return x


def KAN_Net5(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer5(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer5(x,layer_params)
    return x


def KAN_Net7(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer7(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x),
                    T6(x),
                    T7(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer7(x,layer_params)
    return x


def KAN_Net9(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer9(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x),
                    T6(x),
                    T7(x),
                    T8(x),
                    T9(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer9(x,layer_params)
    return x

def KAN_Net11(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer11(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params["W"]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x),
                    T6(x),
                    T7(x),
                    T8(x),
                    T9(x),
                    T10(x),
                    T11(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer11(x,layer_params)
    return x
# WN KAN
def WN_KAN_Net5(params, X_in, M1, M2, activation, norm_fn):  
    def Cheby_KAN_layer5(x,layer_params,expanded_arr=[]):
        # Read chebyshev coefficients and other params:
        cheby_coeffs= layer_params["W"]
        g= layer_params["g"]
        b= layer_params["b"]
        # Normalize
        cheby_coeffs= cheby_coeffs / jnp.linalg.norm(cheby_coeffs, axis=0, keepdims=True)
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return g * x+b 
    #Define params   
    params_N = params["params"]
    #Normalize inputs
    x = norm_fn(X_in, M1, M2)
    for ly in range(len(params_N)):
        layer_params=params_N[ly]
        x=Cheby_KAN_layer5(x,layer_params)
    return x
def KAN5_ResNet(params, X_in, M1, M2, activation, norm_fn):
    def Cheby_KAN_layer5(x,layer_params,W_key):
        # Read chebyshev coefficients:
        cheby_coeffs= layer_params[W_key]
        inputdim=cheby_coeffs.shape[0]
        outdim=cheby_coeffs.shape[1]
        # Normalize 
        x = activation(x)
        # Reshape
        x = x.reshape((-1, inputdim, 1))
        x=jnp.stack((T0(x),
                    T1(x),
                    T2(x),
                    T3(x),
                    T4(x),
                    T5(x)),axis=2)# Discard dummy dimension
        # Compute the Chebyshev interpolation
        x = jnp.einsum("bid0,iod->bo", x, cheby_coeffs)  # shape = (batch_size, output_dim)
        # Remove extra dimension
        x=  jnp.reshape(x, (outdim,))
        return x      
    params_N = params["params"]
    inputs = norm_fn(X_in, M1, M2)
    layer_params=params_N[0]
    inputs=Cheby_KAN_layer5(inputs,layer_params,W_key="W")
    for ly in range(1,len(params_N[:-1])):
        layer_params=params_N[ly]
        g= Cheby_KAN_layer5(inputs,layer_params,W_key="W")
        h= Cheby_KAN_layer5(g,layer_params,W_key="W2")
        inputs= h+inputs
    layer_params=params_N[-1]
    inputs=Cheby_KAN_layer5(inputs,layer_params,W_key="W")    
    return inputs


