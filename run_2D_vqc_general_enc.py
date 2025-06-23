# SOLUTION FOR BURGERS' EQUATION USING A VQC WITH 1 QUBIT

import os

import torch

import argparse
import json

import torch.nn as nn

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

import numpy as onp

from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import RegularGridInterpolator, griddata

import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def parse_args():
    parser = argparse.ArgumentParser(
        description="2D Burgers equation with Universal QML Circuit with batching"
    )
    parser.add_argument(
        '--encoding', choices=["linear", "nonlinear"], default="linear",
        help="Encoding strategy: 'linear' or 'nonlinear'"
    )
    parser.add_argument(
        '--nlayers', type=int, default=10,
        help='Number of variational layers'
    )
    parser.add_argument(
        '--npoints', type=int, default=1000,
        help='Number of collocation points'
    )
    parser.add_argument(
        '--epochs', type=int, default=500,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', '--learning-rate', type=float, default=0.1,
        help='Learning rate for the optimizer'
    )
    parser.add_argument(
        '--random_seed', type=int, default=42,
        help='Seed of the random number generators'
    )
    return parser.parse_args()


# -------------------------------------- Extra functions ---------------------------

# Latin Hypercube Samples for collocation points
def generate_lhs_points(npoints, rng):
    """
    Inputs:
    -----------
    npoints (int): Number of sample points.
    rng (np.random.Generator): A NumPy random number generator instance.

    Outputs:
    --------
    x_tensor (torch.Tensor): 1D tensor of shape (npoints, 1) with values scaled in the range [-1, 1].
    t_tensor (torch.Tensor): 1D tensor of shape (npoints, 1) with values scaled in the range [0, 1].
    """
    sampler = LatinHypercube(d=2,  rng=rng) 
    samples = sampler.random(n=npoints) 
    x_lhs = -1 + 2 * samples[:, 0]     
    t_lhs = samples[:, 1]               

    x_tensor = torch.from_numpy(x_lhs).double().view(-1, 1).requires_grad_()
    t_tensor = torch.from_numpy(t_lhs).double().view(-1, 1).requires_grad_()

    return x_tensor, t_tensor


def compute_exact_solution(nu, L=1, Nx=500, T=1, dt=0.0001):
    """
    Inputs:
    -----------
    nu (float): Viscosity coefficient of the Burgers' equation.
    L (float): Half-length of the spatial domain (the domain spans from -L to L). Default is 1.
    Nx (int): Number of spatial intervals (number of grid points = Nx + 1). Default is 500.
    T (float): Final time of the simulation. Default is 1.
    dt (float): Time step size. Default is 0.0001.

    Output:
    --------
    u_interp : A function that interpolates the solution u(x, t) at arbitrary (x, t) points within the computed domain using cubic interpolation.

    """
    # dt=0.0001
    dx = 2 * L / Nx  
    Nt = round(T / dt)  

    x = onp.linspace(-L, L, Nx + 1)
    t = onp.linspace(0, T, Nt + 1)

    U = onp.zeros((Nx + 1, Nt + 1))
    U[:, 0] = -onp.sin(np.pi * x)

    sigma = nu * dt / dx**2
    A = onp.diag((1 + sigma) * onp.ones(Nx - 1)) + onp.diag(-0.5 * sigma * onp.ones(Nx - 2), 1) + onp.diag(-0.5 * sigma * onp.ones(Nx - 2), -1)

    for n in tqdm.tqdm(range(Nt)):
        u_old = U[:, n].copy()
        u_new = u_old.copy()
        b = (1 - sigma) * u_old[1:Nx] + 0.5 * sigma * (u_old[:Nx-1] + u_old[2:Nx+1])

        for i in range(1, Nx):
            if u_old[i] > 0:
                du_dx = (u_old[i] - u_old[i-1]) / dx  
            else:
                du_dx = (u_old[i+1] - u_old[i]) / dx  
            b[i-1] += -dt * u_old[i] * du_dx  

        u_new[1:Nx] = onp.linalg.solve(A, b)
        u_new[0] = u_new[-1] = 0  
        U[:, n + 1] = u_new

    u_interp = RegularGridInterpolator((x, t), U, method="cubic", bounds_error=False, fill_value=None)
    return u_interp


# Non-linear encoding
def scaled_tanh_encoding(x, a, b):
    """
    Inputs:
    -----------
    x (array-like or float): Input values to encode.
    a (float): Scaling factor applied inside the tanh.
    b (float): Bias term applied inside the tanh.

    Output:
    --------
    encoded (array-like or float): Output values scaled to the range [0, π].
    """
    return (np.pi/2) * (np.tanh(a * x + b) + 1)


def linear_encoding(t, c, d):
    """
    Inputs:
    -----------
    t (array-like or float): Input values to encode.
    c (float): Scaling factor.
    d (float): Bias added after scaling.

    Output:
    --------
    encoded (array-like or float): Output values clipped to [0, π].
    """
    return np.clip(c * t + d, 0, np.pi)


def combined_linear_encoding(x, t, params):
    qml.RY(linear_encoding(x, params[0], params[1]), wires=0)
    qml.RZ(linear_encoding(x, params[2], params[3]), wires=0)
    qml.RY(linear_encoding(t, params[4], params[5]), wires=0)
    qml.RZ(linear_encoding(t, params[6], params[7]), wires=0)


def combined_non_linear_encoding(x, t, params):
    qml.RY(scaled_tanh_encoding(x, params[0], params[1]), wires=0)
    qml.RZ(scaled_tanh_encoding(x, params[2], params[3]), wires=0)
    qml.RY(linear_encoding(t, params[4], params[5]), wires=0)
    qml.RZ(linear_encoding(t, params[6], params[7]), wires=0)



def one_layer(x, t, params, encoding_type):
    if encoding_type == "linear":
        combined_linear_encoding(x, t, params)
    elif encoding_type == "nonlinear":
        combined_non_linear_encoding(x, t, params)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")



def qcircuit(nlayers, encoding_type):
    dev = qml.device("lightning.qubit", wires=1)
    @qml.qnode(dev)

    def circuit(x, t, params):
        for l in range(nlayers):
            one_layer(x, t, params[l], encoding_type)
        return qml.expval(qml.Z(0))
    
    return circuit



def def_loss(x_train, t_train, u_train, nlayers, encoding_type):
    circuit = qcircuit(nlayers, encoding_type)

    def rnd_param_init(seed=None):
        if seed is not None:
            np.random.seed(seed)
        params = qml.numpy.array(np.random.uniform(low=0, high=2 * np.pi, size=(nlayers, 8)), requires_grad=True)

        
        params[:,0] = (params[:,0] - np.pi) / 100
        params[:,2] = (params[:,2] - np.pi) / 100
        params[:,4] = (params[:,4] - np.pi) / 100
        params[:,6] = (params[:,6] - np.pi) / 100
        return params

    def cost(x, t, u, params):
        loss = 0
        for (x_, t_, u_) in zip(x, t, u):
            u_predict = circuit(x_, t_, params)
            loss += (u_predict - u_) ** 2
        return loss / u.size

    return rnd_param_init, circuit, cost



# ------------------------------------------------- main function ------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    # Create a dictionary of all parsed arguments
    args_dict = vars(args)

    nlayers = args.nlayers
    npoints = args.npoints
    epochs = args.epochs
    random_seed = args.random_seed
    lr = args.lr
    encoding = args.encoding

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    save_path = f"../results_VQC_gen_enc_50.50/{encoding}Enc_{nlayers}layers_{npoints}data_{epochs}epochs_{lr}lr_{random_seed}rng_"
    os.makedirs(save_path, exist_ok=True)

    params_path = save_path + "/params"
    grads_path = save_path + "/grads"

    os.makedirs(params_path, exist_ok=True)
    os.makedirs(grads_path, exist_ok=True)

    # Data
    nu = 0.01 / np.pi
    u_exact_function = compute_exact_solution(nu)
    x_values, t_values = generate_lhs_points(npoints=npoints, rng=random_seed)

    indices = np.random.choice(npoints, int(0.5 * npoints), replace=False)
    comp_idx = np.setdiff1d(np.arange(npoints), indices)

    x_train = np.array(x_values[indices].detach().numpy())
    t_train = np.array(t_values[indices].detach().numpy())
    x_test = np.array(x_values[comp_idx].detach().numpy())
    t_test = np.array(t_values[comp_idx].detach().numpy())

    u_train = np.array(
        u_exact_function((
            x_train.flatten(),
            t_train.flatten()
        )).reshape(-1, 1)
    )

    u_test = qml.numpy.array(
        u_exact_function((
            x_test.flatten(),
            t_test.flatten()
        )).reshape(-1, 1)
    )

    # Define model
    rnd_param_init, circuit, cost = def_loss(x_train, t_train, u_train, nlayers, args.encoding)
    opt = AdamOptimizer(lr)
    params = rnd_param_init()


    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        # Gradient computation
        grad_fn = qml.grad(lambda p: cost(x_train, t_train, u_train, p))
        gradients = grad_fn(params)
        np.save(file=grads_path + f"/grads_epoch{epoch}", arr=np.array(gradients))

        params = opt.step(lambda p: cost(x_train, t_train, u_train, p), params)
        np.save(file=params_path + f"/params_epoch{epoch}", arr=np.array(params))

        train_loss = cost(x_train, t_train, u_train, params)
        test_loss = cost(x_test, t_test, u_test, params)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}")


    # Save model and losses
    np.save(os.path.join(save_path, "final_params.npy"), params.numpy())
    np.save(os.path.join(save_path, "train_losses.npy"), train_losses)
    np.save(os.path.join(save_path, "test_losses.npy"), test_losses)

    args_dict.update({
        "Best training loss": float(np.min(np.array(train_losses))),
        "Best test loss": float(np.min(np.array(test_losses))),
        "Best Loss was at epoch": int(np.argmin(np.array(train_losses)) + 1)
    })


    # Save parsed arguments to JSON
    with open(os.path.join(save_path, "configuration.json"), "w") as json_file:
        json.dump(args_dict, json_file, indent=4)

    # Save final parameters in human-readable JSON format
    param_dict = {f"Layer_{i+1}": layer_params.tolist() for i, layer_params in enumerate(params)}
    with open(os.path.join(save_path, "final_params.json"), "w") as f:
        json.dump(param_dict, f, indent=4)


    # Plots

    # Plot Predictions
    t_fixed_vals = [0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1]
    x_plot = np.linspace(-1, 1, 100)

    plt.figure(figsize=(12, 8))
    colors = cm.viridis(np.linspace(0, 1, len(t_fixed_vals)))

    for t_fixed, color in zip(t_fixed_vals, colors):
        xt = np.column_stack((x_plot, np.full_like(x_plot, t_fixed)))
        u_true = u_exact_function(xt)
        u_pred = [circuit(x, t_fixed, params) for x in x_plot]

        plt.plot(x_plot, u_true, linestyle='-', color=color, label=f"Exact t={t_fixed:.1f}")
        plt.plot(x_plot, u_pred, linestyle='--', color=color, label=f"Pred t={t_fixed:.1f}")

    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Predictions.pdf"), bbox_inches="tight")

    # Plot Loss
    plt.figure()
    plt.semilogy(train_losses, label="Train loss")
    plt.semilogy(test_losses, label="Test loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Loss_curve.pdf"), bbox_inches="tight")


    def compute_errors(x, t, u_true, circuit, params):
        u_pred = np.array([circuit(xi.item(), ti.item(), params) for xi, ti in zip(x, t)])
        error = u_pred - u_true.flatten()
        x_vals = x.flatten()
        t_vals = t.flatten()
        return error, x_vals, t_vals

    
    # Compute errors
    error_train, x_train_np, t_train_np = compute_errors(x_train, t_train, u_train, circuit, params)
    error_test, x_test_np, t_test_np = compute_errors(x_test, t_test, u_test, circuit, params)

    # Save error data
    np.save(os.path.join(save_path, "error_train.npy"), error_train)
    np.save(os.path.join(save_path, "error_test.npy"), error_test)

    abs_error_train = np.abs(error_train)
    abs_error_test = np.abs(error_test)

    # Error statistics
    error_stats = {
        "train": {
            "mean_abs_error": float(np.mean(abs_error_train)),
            "median_abs_error": float(np.median(abs_error_train))
        },
        "test": {
            "mean_abs_error": float(np.mean(abs_error_test)),
            "median_abs_error": float(np.median(abs_error_test))
        }
    }

    with open(os.path.join(save_path, "error_stats.json"), "w") as f:
        json.dump(error_stats, f, indent=4)


    # Normalize errors for visualization
    size_train = 500 * (abs_error_train / abs_error_train.max() + 1e-12)  
    size_test = 500 * (abs_error_test / abs_error_test.max() + 1e-12)


    # Scatter plot of errors
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train_np, t_train_np, s=size_train, c='blue', alpha=0.5, label='Train', marker='o')
    plt.scatter(x_test_np, t_test_np, s=size_test, c='red', alpha=0.5, label='Test', marker='^')
    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Error_scatter.pdf"), bbox_inches="tight")

    
    # Heat map
    x_all = np.concatenate([x_train_np, x_test_np])
    t_all = np.concatenate([t_train_np, t_test_np])
    error_all = np.concatenate([error_train, error_test])

    # Create interpolation grid
    xi = np.linspace(-1, 1, 200)
    ti = np.linspace(0, 1, 200)
    Xi, Ti = np.meshgrid(xi, ti)

    Ei = griddata(
        points=(x_all, t_all),
        values=error_all,
        xi=(Xi, Ti),
        method='linear'
    )

    # Plot heatmap 1
    plt.figure(figsize=(10, 6))
    plt.imshow(Ei, vmin=-0.5, vmax=0.5, cmap="bwr",
            extent=[-1, 1, 0, 1],  # [x_min, x_max, t_min, t_max]
            #origin='lower',       # Places t=0 at the bottom
            aspect='auto')        # Avoids stretching
    cbar = plt.colorbar()
    cbar.set_label("Error prediction", fontsize=10)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(os.path.join(save_path, "Error_heatmap_imshow.pdf"), bbox_inches="tight")
    plt.show()


    # Plot heatmap 2
    plt.figure(figsize=(10, 6))
    plt.imshow(Ei,vmin=-0.5,vmax=0.5,cmap="bwr")
    cbar = plt.colorbar()
    cbar.set_label("Error prediction", fontsize=10)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(os.path.join(save_path, "Error_heatmap.pdf"), bbox_inches="tight")
    plt.show()


    # Heat map with the absolute value of the error 
    abs_error_all = np.concatenate([abs_error_train, abs_error_test])
    
    Ei_abs = griddata(
        points=(x_all, t_all),
        values=abs_error_all,
        xi=(Xi, Ti),
        method='linear'
    )

    plt.figure(figsize=(10, 6))
    plt.contourf(Xi, Ti, Ei_abs, levels=100, cmap="inferno")
    cbar = plt.colorbar()
    cbar.set_label("Prediction absolute error", fontsize=10)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig(os.path.join(save_path, "Abs_Error_heatmap.pdf"), bbox_inches="tight")


if __name__ == '__main__':
    main()
