import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation


def plot_macro_distributions(df, scenario_weights, step=100):
    gdp = df['GDP_Growth'].values
    inf = df['Inflation'].values
    scenario = df['Scenario'].values

    plt.style.use('seaborn-talk')
    purple_bg = '#232136' 
    axes_bg   = '#232136'   

    plt.rcParams.update({
    'figure.facecolor': purple_bg,
    'axes.facecolor': axes_bg,
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': "#EDEDF3",
    'legend.edgecolor': 'white',
    'legend.labelcolor': 'white',
    'legend.fontsize': 11
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Macroeconomic Simulations", fontsize=14, fontweight='bold')

    gdp_bins = np.histogram_bin_edges(gdp, bins='auto')
    inf_bins = np.histogram_bin_edges(inf, bins='auto')

    gdp_counts = np.zeros(len(gdp_bins) - 1)
    inf_counts = np.zeros(len(inf_bins) - 1)

    axes[0].set_xlim(gdp_bins[0], gdp_bins[-1])
    axes[0].set_ylim(0, len(gdp) * 0.05)
    axes[0].set_title("GDP Growth")
    axes[0].set_xlabel("GDP Growth")

    _, _, gdp_patches = axes[0].hist(
        gdp_bins[:-1],
        bins=gdp_bins,
        weights=gdp_counts,
        color='#eb6f92',
        edgecolor='#eb6f92',
        alpha=0.8
    )

    axes[1].set_xlim(inf_bins[0], inf_bins[-1])
    axes[1].set_ylim(0, len(inf) * 0.05)
    axes[1].set_title("Inflation")
    axes[1].set_xlabel("Inflation")

    _, _, inf_patches = axes[1].hist(
        inf_bins[:-1],
        bins=inf_bins,
        weights=inf_counts,
        color='#ea9a97',
        edgecolor='#ea9a97',
        alpha=0.8
    )

    title = fig.text(0.5, 0.92, "", ha='center', fontsize=11)

    gdp_bin_idx = np.digitize(gdp, gdp_bins) - 1
    inf_bin_idx = np.digitize(inf, inf_bins) - 1

    frames = np.arange(0, len(df), step)

    def update(i):
        idx = frames[i]
        for j in range(idx, min(idx + step, len(df))):
            if 0 <= gdp_bin_idx[j] < len(gdp_counts):
                gdp_counts[gdp_bin_idx[j]] += 1
            if 0 <= inf_bin_idx[j] < len(inf_counts):
                inf_counts[inf_bin_idx[j]] += 1

        for rect, h in zip(gdp_patches, gdp_counts):
            rect.set_height(h)
        for rect, h in zip(inf_patches, inf_counts):
            rect.set_height(h)

        title.set_text(f"Simulation Run: {idx} | {scenario_weights}")
        return gdp_patches + inf_patches + (title,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=30,
        blit=False,
        repeat=False
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    return True

if __name__ == "__main__":
    scenario_weights = {'Base': 0.82129, 'Pessimistic': 0.08952, 'Optimistic': 0.08919}
    sim_df = pd.read_csv(r"results\macro_simulations.csv", index_col=0)
    plot_macro_distributions(sim_df,scenario_weights)