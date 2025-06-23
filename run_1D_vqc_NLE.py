#!/usr/bin/env python3

# FITTING FOR BURGERS' EQUATION USING A PQC WITH 1 QUBIT AND NONLINEAR ENCODING STRATEGY

import os

import torch

import argparse
import json

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
        description="1D Burgers' equation fitting with nonlinear encoding strategy"
    )
    parser.add_argument(
        "--time", type=float, default=0,
        help="Time snapshot to load data from"
    )
    parser.add_argument(
        '--nlayers', type=int, default=10,
        help='Number of variational layers'
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

# Encoding strategies
def loglin_encoding(x, a, b):
    """
    f(x) = clip(a * x + b * sign(x) * log(|x| + 1e-6), 0, π)
    """
    eps = 1e-6
    log_part = b * np.sign(x) * np.log(np.abs(x) + eps)
    return np.clip(a * x + log_part, 0, np.pi)
    
def scaled_tanh_encoding(x, a, b):
    """
    f(x) = (π/2) * (tanh(a * x + b) + 1)
    """
    return (np.pi / 2) * (np.tanh(a * x + b) + 1)



def one_layer(x, params):
    qml.RY(scaled_tanh_encoding(x, params[0], params[1]), wires=0)
    qml.RZ(scaled_tanh_encoding(x, params[2], params[3]), wires=0)



def qcircuit(nlayers):
    dev = qml.device("lightning.qubit", wires=1)
    @qml.qnode(dev)

    def circuit(x, params):
        for l in range(nlayers):
            one_layer(x, params[l])
        return qml.expval(qml.Z(0))
    
    return circuit



def def_loss(x_train, u_train, nlayers):
    circuit = qcircuit(nlayers)

    def rnd_param_init(seed=None):
        if seed is not None:
            np.random.seed(seed)
        params = qml.numpy.array(np.random.uniform(low=0, high=2 * np.pi, size=(nlayers, 4)), requires_grad=True)

        
        params[:,0] = (params[:,0] - np.pi) / 100
        params[:,2] = (params[:,2] - np.pi) / 100
        return params

    def cost(x, u, params):
        loss = 0
        for (x_, u_) in zip(x, u):
            u_predict = circuit(x_, params)
            loss += (u_predict - u_) ** 2
        return loss / u.size

    return rnd_param_init, circuit, cost



# ------------------------------------------------- main function ------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    args_dict = vars(args)

    time = str(int(args.time)) if args.time.is_integer() else str(args.time)
    nlayers = args.nlayers
    epochs = args.epochs
    random_seed = args.random_seed
    lr = args.lr

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    save_path = f"../results_VQC_1D_NLE/t{time}_{nlayers}layers_{epochs}epochs_{lr}lr_{random_seed}rng"
    os.makedirs(save_path, exist_ok=True)

    params_path = save_path + "/params"
    #grads_path = save_path + "/grads"

    os.makedirs(params_path, exist_ok=True)
    #os.makedirs(grads_path, exist_ok=True)

    # Data
    data_path = "../"
    file_path = f"{data_path}burgers_solution_t{time}.npy"
    data = np.load(file_path, allow_pickle=True).item()
    x_exact = qml.numpy.array(data['x'].reshape(-1, 1))
    u_exact = qml.numpy.array(data['u'].reshape(-1, 1))


    indices = np.arange(0,500,5)
    x_train = x_exact[indices]
    u_train = u_exact[indices]

    all_indices = np.arange(len(x_exact))
    test_indices = np.setdiff1d(all_indices, indices)
    x_test = x_exact[test_indices]
    u_test = u_exact[test_indices]

    # Target
    plt.plot(x_train, u_train, marker=".")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.savefig(os.path.join(save_path, "Target function.pdf"), bbox_inches="tight")

    nu = 0.01 / np.pi

    # Define model
    rnd_param_init, circuit, cost = def_loss(x_train, u_train, nlayers)
    opt = AdamOptimizer(lr)
    params = rnd_param_init()


    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        params = opt.step(lambda p: cost(x_train, u_train, p), params)
        np.save(file=params_path + f"/params_epoch{epoch}", arr=np.array(params))

        train_loss = cost(x_train, u_train, params)
        test_loss = cost(x_test, u_test, params)
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

    # Save final parameters in JSON format
    param_dict = {f"Layer_{i+1}": layer_params.tolist() for i, layer_params in enumerate(params)}
    with open(os.path.join(save_path, "final_params.json"), "w") as f:
        json.dump(param_dict, f, indent=4)


    # Plotting
    u_predict = []
    for x in x_train:
        u_predict.append(circuit(x,params))
    plt.scatter(x_train, u_predict, label="Prediction")
    plt.scatter(x_train, u_train, label="Target")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()  
    plt.savefig("VQC_1D_Prediction.pdf")


    plt.figure(figsize=(7, 7 * 6 / 8), dpi=300)
    plt.plot(
        x_train, 
        u_predict, 
        marker=".",
        markersize=10,
        color="red", 
        label="Prediction",
        alpha=0.7
    )
    plt.plot(
        x_train, 
        u_train,
        marker=".",
        markersize=10,
        color="blue", 
        label="Target",
        alpha=0.7
    )
    plt.xlabel(r"x")
    plt.ylabel(r"u(x)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Predictions.pdf"), bbox_inches="tight")

    plt.figure()
    plt.semilogy(train_losses, label="Train loss")
    plt.semilogy(test_losses, label="Test loss", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Loss_curve.pdf"), bbox_inches="tight")


if __name__ == '__main__':
    main()
