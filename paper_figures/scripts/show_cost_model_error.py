import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
np.random.seed(42)

data = []
sp_sizes = [1, 2, 4, 8]
n_samples_per_config = 50

for sp in sp_sizes:
    for _ in range(n_samples_per_config):
        vae_err = np.random.normal(0, 2.5) # Mean=0, Std=2.5
        data.append({'Module': 'VAE', 'SP Size': sp, 'Error (%)': vae_err})
        
        dit_std = 2.0 + (sp * 0.3)
        dit_err = np.random.normal(-1.0, dit_std) 
        data.append({'Module': 'DiT', 'SP Size': sp, 'Error (%)': dit_err})

df = pd.DataFrame(data)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
sns.set(style="ticks", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

def plot_error_box(ax, module_name):
    subset = df[df['Module'] == module_name]
    
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Pred.')
    
    sns.boxplot(
        x='SP Size', 
        y='Error (%)', 
        data=subset, 
        ax=ax,
        width=0.5,
        color='white',
        linecolor='black',
        showfliers=True,
        fliersize=3
    )
    
    # sns.stripplot(x='SP Size', y='Error (%)', data=subset, ax=ax, 
    #               color='black', alpha=0.3, jitter=True, size=3)

    ax.set_title(f'{module_name} Prediction Error', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sequence Parallel Size')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plot_error_box(axes[0], 'VAE')
plot_error_box(axes[1], 'DiT')

axes[0].set_ylabel('Prediction Error (%) \n (Predicted - Actual) / Actual')
axes[1].set_ylabel('')

axes[0].set_ylim(-20, 20) 

plt.tight_layout()
plt.show()