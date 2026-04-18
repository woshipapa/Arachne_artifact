import pickle
import matplotlib.pyplot as plt

# Load results
with open("vae_results.pkl", "rb") as f:
    results = pickle.load(f)

# Collect all unique batch sizes from all entries
all_bs = set()
for bs_time_dict in results.values():
    all_bs.update(bs_time_dict.keys())

# Sort batch sizes and ensure they are ints
all_bs = sorted(int(bs) for bs in all_bs)

# Start plotting
plt.figure(figsize=(10, 6))

for shape, bs_time_dict in results.items():
    num_frames, H, W = shape
    sorted_bs = sorted(bs_time_dict.keys())
    times = [bs_time_dict[bs] for bs in sorted_bs]

    label = f"{num_frames} frames, {H}x{W}"
    plt.plot(sorted_bs, times, marker='o', label=label)

# English labels
plt.title("VAE Inference Time vs Batch Size")
plt.xlabel("Batch Size")
plt.ylabel("Average Inference Time (seconds)")
plt.legend(title="Input Shape")
plt.grid(True)

# Set x-axis ticks to match only available batch sizes
plt.xticks(all_bs)

plt.tight_layout()
plt.savefig(f"vae_bs_shape.png")
