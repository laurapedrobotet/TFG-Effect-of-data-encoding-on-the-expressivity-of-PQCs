import os
import numpy as np
import matplotlib.pyplot as plt

# Configuration
times = [0, 0.3, 0.5, 1]
layers = [1, 2, 4, 6, 8, 10]
seeds = list(range(10))
encodings = ["linear", "nonlinear"]
colors = {"linear": "tab:blue", "nonlinear": "tab:red"}

base_paths = {
    "linear": "../results_VQC_1D_LE",
    "nonlinear": "../results_VQC_1D_NLE/ScaledTanh"
}

plot_dir = "../plots_1D"
os.makedirs(plot_dir, exist_ok=True)

summary_data = {t: {enc: [] for enc in encodings} for t in times}
summary_log_data = {t: {enc: [] for enc in encodings} for t in times}

for time in times:
    t_str = f"t{time}"
    for encoding in encodings:
        base_path = os.path.join(base_paths[encoding], t_str)
        for nlayers in layers:
            losses_per_seed = []
            for seed in seeds:
                folder = f"{t_str}_{nlayers}layers_1000epochs_0.1lr_{seed}rng"
                path = os.path.join(base_path, folder, "test_losses.npy")
                if os.path.exists(path):
                    test_losses = np.load(path)
                    min_loss = np.min(test_losses)
                    losses_per_seed.append(min_loss)
                else:
                    print(f"File not found: {path}")

            mean_loss = np.mean(losses_per_seed)
            std_loss = np.std(losses_per_seed)
            summary_data[time][encoding].append((mean_loss, std_loss))

            log_losses = np.log(losses_per_seed)
            log_mean_loss = np.mean(log_losses)
            std_log_mean_loss = np.std(log_losses)
            summary_log_data[time][encoding].append((log_mean_loss, std_log_mean_loss))

    # Plot individual por tiempo (log con barras simétricas)
    plt.figure(figsize=(5, 5*6/8))
    for encoding in encodings:
        log_means = [x[0] for x in summary_log_data[time][encoding]]
        stds_log = [x[1] for x in summary_log_data[time][encoding]]
        means = np.exp(log_means)
        lower = np.exp(np.array(log_means) - stds_log)
        upper = np.exp(np.array(log_means) + stds_log)
        yerr = [means - lower, upper - means]
        plt.errorbar(layers, means, yerr=yerr, label=f"{encoding.capitalize()} encoding",
                     color=colors[encoding], marker='o', capsize=5, alpha=0.75)
    plt.xlabel("#Layers", fontsize=12)
    plt.ylabel("$\epsilon$ (MSE)", fontsize=12)
    plt.yscale("log")
    plt.title(f"t = {time}", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"1D_sym_log_t{time}.pdf"))
    plt.close()

# Plot resumen tipo grid (2x2)
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
axs = axs.flatten()

for idx, time in enumerate(times):
    ax = axs[idx]
    for encoding in encodings:
        log_means = [x[0] for x in summary_log_data[time][encoding]]
        stds_log = [x[1] for x in summary_log_data[time][encoding]]
        means = np.exp(log_means)
        lower = np.exp(np.array(log_means) - stds_log)
        upper = np.exp(np.array(log_means) + stds_log)
        yerr = [means - lower, upper - means]
        ax.errorbar(layers, means, yerr=yerr, label=f"{encoding.capitalize()}",
                    color=colors[encoding], marker='o', capsize=4, alpha=0.8)
    ax.set_title(f"t = {time}")
    ax.set_yscale("log")
    ax.grid(True, linestyle='--', alpha=0.5)
    if idx in [2, 3]:
        ax.set_xlabel("#Layers", fontsize=10)
    if idx in [0, 2]:
        ax.set_ylabel("$\epsilon$ (MSE)", fontsize=10)

fig.legend(["Linear", "Nonlinear"], loc="upper center", ncol=2, fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(plot_dir, "summary_sym_log_1D.pdf"))
plt.close()
