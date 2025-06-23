import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
base_dir = "../results_VQC_gen_enc_50"
layers = [1, 2, 4, 6, 8, 10]
seeds = range(10)
epochs = 1000

# Store results: {nlayers: [avg_variance_epoch0, epoch1, ..., epoch999]}
layer_variance_curves = {}

for nlayers in layers:
    print(f"Processing {nlayers} layers...")
    epoch_variances = []

    for epoch in range(epochs):
        gradients = []

        for seed in seeds:
            folder_name = f"nonlinearEnc_{nlayers}layers_1000data_1000epochs_0.1lr_{seed}rng_"
            grad_path = os.path.join(base_dir, folder_name, "grads", f"grads_epoch{epoch}.npy")

            if not os.path.exists(grad_path):
                print(f"  Missing: {grad_path}")
                continue

            grad = np.load(grad_path)          # shape: (nlayers, 8)
            gradients.append(grad.flatten())   # shape: (nlayers * 8,)

        if len(gradients) < 2:
            epoch_variances.append(np.nan)  # not enough data to compute variance
            continue

        matrix = np.stack(gradients)        # shape: (n_seeds, nlayers * 8)
        variances = np.var(matrix, axis=0)  # shape: (nlayers * 8,)
        avg_variance = np.mean(variances)   # scalar
        epoch_variances.append(avg_variance)

    layer_variance_curves[nlayers] = epoch_variances

# Plotting
plt.figure(figsize=(12, 6))
for nlayers, variances in sorted(layer_variance_curves.items()):
    plt.plot(variances, label=f"{nlayers}")

plt.xlabel("Epoch")
plt.ylabel("Avg. gradient variance (across seeds)")
plt.yscale("log")
#plt.title("Average Gradient Variance vs Epoch for Different Layer Counts")
plt.legend(title="Layers")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "Gradients_plot_log.pdf"), bbox_inches="tight")

