import os
import numpy as np
import matplotlib.pyplot as plt

base_path = "../results_VQC_gen_enc_50"

plot_dir = "../plots_2D_50"
os.makedirs(plot_dir, exist_ok=True)

layers = [1, 2, 4, 6, 8, 10]
seeds = list(range(10))  

encodings = ["linear", "nonlinear"]
colors = {"linear": "tab:blue", "nonlinear": "tab:red"}

results = {enc: [] for enc in encodings}
log_results = {enc: [] for enc in encodings}

for encoding in encodings:
    for nlayers in layers:
        losses_per_seed = []
        for seed in seeds:
            folder = f"{encoding}Enc_{nlayers}layers_1000data_1000epochs_0.1lr_{seed}rng_"
            path = os.path.join(base_path, folder, "test_losses.npy")
            if os.path.exists(path):
                test_losses = np.load(path)
                min_loss = np.min(test_losses)
                losses_per_seed.append(min_loss)
            else:
                print(f"File not found: {path}")
        # We store the average and standard deviation of the losses for this number of layers.
        mean_loss = np.mean(losses_per_seed)
        std_loss = np.std(losses_per_seed)
        results[encoding].append((mean_loss, std_loss))

        # Symmetric error bars in log scale
        log_losses = np.log(losses_per_seed)
        log_mean_loss = np.mean(log_losses)
        std_log_mean_loss = np.std(log_losses)
        log_results[encoding].append((log_mean_loss, std_log_mean_loss))


plt.figure(figsize=(5, 5*6/8))

for encoding in encodings:
    log_means = [x[0] for x in log_results[encoding]]
    stds_log = [x[1] for x in log_results[encoding]]

    means = np.exp(log_means)
    lower = np.exp(np.array(log_means) - stds_log)
    upper = np.exp(np.array(log_means) + stds_log)
    yerr = [means - lower, upper - means]  # Error bars in original scale

    plt.errorbar(layers, means, yerr=yerr, label=f"{encoding.capitalize()} encoding",
                 color=colors[encoding], marker='o', capsize=5, alpha=0.75)


plt.xlabel("#Layers", fontsize=12)
plt.ylabel("$\epsilon$ (MSE)", fontsize=12)
plt.yscale("log")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
output_file = os.path.join(plot_dir, "sym_log_final_plot.pdf")
plt.savefig(output_file)
