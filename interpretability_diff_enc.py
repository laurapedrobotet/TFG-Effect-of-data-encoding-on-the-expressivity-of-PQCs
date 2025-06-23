import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

# List of depths (number of layers)
layer_list = [1, 2, 4, 6, 8, 10]

# Folder base — adjust if needed
base_path = "../results_diff_enc_50/nonlinearEnc_{L}layers_1000data_1000epochs_0.1lr_2rng"
save_path = "../results_diff_enc_50"
# Storage for mean absolute values of a, b, c across layers
a_means, b_means, c_means = [], [], []

for L in layer_list:
    print(f"\nProcessing L = {L}")

    # Format experiment directory path
    exp_dir = base_path.format(L=L)
    params_dir = os.path.join(exp_dir, "params")

    # Load test losses
    test_loss_path = os.path.join(exp_dir, "test_losses.npy")
    if not os.path.exists(test_loss_path):
        print(f"  Warning: test_losses not found for L={L}")
        continue
    test_losses = np.load(test_loss_path)

    # Find best epoch
    best_epoch = np.argmin(test_losses)
    print(f"  Best epoch: {best_epoch} | Test loss: {test_losses[best_epoch].item():.4e}")

    # Load corresponding parameter file from 'params/' subdirectory
    param_file = os.path.join(params_dir, f"params_epoch{best_epoch}.npy")
    if not os.path.exists(param_file):
        print(f"  Warning: params for epoch {best_epoch} not found at {param_file}")
        continue

    best_params = np.load(param_file, allow_pickle=True)

    # Extract a, b, c from each layer (params[0], params[1], params[2])
    a_vals, b_vals, c_vals = [], [], []
    for l in range(L):
        layer_params = best_params[l]
        a_vals.append(np.abs(layer_params[0]))  # a = multiplies x
        b_vals.append(np.abs(layer_params[1]))  # b = multiplies arccos(x)
        c_vals.append(np.abs(layer_params[2]))  # c = multiplies tanh

    # Store the mean across layers
    a_means.append(np.mean(a_vals))
    b_means.append(np.mean(b_vals))
    c_means.append(np.mean(c_vals))

# === Plot ===
# plt.figure(figsize=(8, 5))
plt.figure(figsize=(5, 5*6/8))
plt.plot(layer_list, a_means, '-o', label='|a| (Fourier)', linewidth=2)
plt.plot(layer_list, b_means, '-s', label='|b| (Chebyshev)', linewidth=2)
plt.plot(layer_list, c_means, '-^', label='|c| (Inductive bias)', linewidth=2)

plt.xlabel("#Layers", fontsize=12)
plt.ylabel("Mean amplitude of parameters", fontsize=12)
#plt.title("Basis preference vs. model depth (seed = 2)", fontsize=14)
plt.xticks(layer_list)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.ylim(bottom=0) 
plt.tight_layout()
plt.savefig(os.path.join(save_path, "Interpretability.pdf"), bbox_inches="tight")
