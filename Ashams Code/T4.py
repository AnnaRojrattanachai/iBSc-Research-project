import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm

file_path = r"C:\Users\mamoh\PycharmProjects\Trial Project\Research project\SPIKETIMEFILES\M24028_20240426_File1_Shank0_goodSpikeTimes.csv"


def load_spike_data(file_path):
    spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
    return spike_data


def bin_spikes(spike_data, bin_width):
    max_time = spike_data["spike_time"].max()
    bins = np.arange(0, max_time + bin_width, bin_width)

    binned_spikes = {}
    for unit_id, group in spike_data.groupby("unit_id"):
        spike_counts, _ = np.histogram(group["spike_time"], bins=bins)
        binned_spikes[unit_id] = spike_counts

    return binned_spikes


def calculate_pairwise_correlations(binned_spikes):
    units = list(binned_spikes.keys())
    spike_matrix = np.array([binned_spikes[unit] for unit in units])

    correlation_matrix = np.corrcoef(spike_matrix)

    correlations = {}
    total_operations = len(units) * (len(units) - 1) // 2

    with tqdm(total=total_operations, desc="Calculating Pairwise Correlations") as pbar:
        for i, j in combinations(range(len(units)), 2):
            correlations[(units[i], units[j])] = correlation_matrix[i, j]
            pbar.update(1)

    return correlations


def visualise_correlation_matrix(correlations, units, font_size):
    correlation_matrix = pd.DataFrame(index=units, columns=units, data=np.nan)
    for (unit1, unit2), r_value in correlations.items():
        correlation_matrix.loc[unit1, unit2] = r_value
        correlation_matrix.loc[unit2, unit1] = r_value
    np.fill_diagonal(correlation_matrix.values, 1)

    plt.figure(figsize=(20, 12))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f",
                cbar_kws={'label': 'Pearson Correlation'},
                xticklabels=units, yticklabels=units)
    plt.xticks(fontsize=font_size, rotation=90)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Neuron ID", fontsize=font_size + 5)
    plt.ylabel("Neuron ID", fontsize=font_size + 5)
    plt.title("Neuron Pairwise Correlation Matrix", fontsize=font_size + 8)
    # plt.show(block=False)
    plt.draw()

def create_cross_correlograms(binned_spikes, max_lag=20):
    units = list(binned_spikes.keys())
    num_units = len(units)
    lags = range(-max_lag, max_lag + 1)
    mean_adjusted = {unit: binned_spikes[unit] - np.mean(binned_spikes[unit]) for unit in units}

    cross_corr_matrices = []

    # Calculate cross-correlation for each pair and lag
    total_operations = (num_units * (num_units - 1)) // 2
    with tqdm(total=total_operations, desc="Calculating Cross-Correlograms") as pbar:
        for i, j in combinations(range(num_units), 2):
            unit1, unit2 = units[i], units[j]
            cross_corr_full = np.correlate(mean_adjusted[unit1], mean_adjusted[unit2], mode='full')
            cross_corr_lagged = cross_corr_full[
                                len(cross_corr_full) // 2 - max_lag: len(cross_corr_full) // 2 + max_lag + 1]
            cross_corr_matrices.append((i, j, cross_corr_lagged))
            pbar.update(1)

    # 1. Display Cross-Correlation Heatmap for each lag
    fig, axes = plt.subplots(4, int(np.ceil(len(lags) / 4)), figsize=(15, 10))
    fig.suptitle("Cross-Correlation Heatmap for Each Lag", fontsize=16)

    for idx, lag in enumerate(lags):
        ax = axes[idx // 4, idx % 4]
        lag_matrix = np.zeros((num_units, num_units))

        for i, j, cross_corr in cross_corr_matrices:
            lag_matrix[i, j] = cross_corr[idx + max_lag]
            lag_matrix[j, i] = lag_matrix[i, j]

        sns.heatmap(lag_matrix, vmin=-1, vmax=1, cmap="coolwarm", cbar=False, ax=ax)
        ax.set_title(f'Lag {lag}', fontsize=8)
        ax.axis('off')

    plt.colorbar(plt.cm.ScalarMappable(cmap="coolwarm"), ax=axes, orientation='horizontal', pad=0.05,
                 label='Cross-Correlation Coefficient')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show(block=False)

    # 2. Display individual Cross-Correlograms for each neuron pair
    plt.figure(figsize=(15, 10))
    for i, j, cross_corr in cross_corr_matrices:
        plt.plot(lags, cross_corr, label=f'Unit {units[i]} - Unit {units[j]}')
    plt.xlabel("Lag (bins)")
    plt.ylabel("Cross-Correlation")
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.title("Cross-Correlogram for Each Neuron Pair")
    plt.show()

    # 3. Lag Heatmap showing max cross-correlation for each neuron pair
    max_corr_matrix = np.zeros((num_units, num_units))
    for i, j, cross_corr in cross_corr_matrices:
        max_corr_matrix[i, j] = np.max(cross_corr)
        max_corr_matrix[j, i] = max_corr_matrix[i, j]
    plt.figure(figsize=(10, 8))
    sns.heatmap(max_corr_matrix, vmin=-1, vmax=1, cmap="coolwarm",
                xticklabels=units, yticklabels=units, cbar_kws={'label': 'Max Cross-Correlation'})
    plt.title("Lag Heatmap (Max Cross-Correlation for Each Pair)")
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")
    plt.show(block=False)


def analyze_spike_data(file_path, bin_width=0.1, max_lag=20):
    spike_data = load_spike_data(file_path)
    binned_spikes = bin_spikes(spike_data, bin_width=bin_width)

    correlations = calculate_pairwise_correlations(binned_spikes)
    visualise_correlation_matrix(correlations, list(binned_spikes.keys()), font_size=5)

    print(f"Correlation matrix for file {os.path.basename(file_path)}:")
    for (unit1, unit2), corr in correlations.items():
        print(f"Unit {unit1} - Unit {unit2}: Pearson correlation = {corr:.4f}")

    create_cross_correlograms(binned_spikes, max_lag=max_lag)


analyze_spike_data(file_path, bin_width=0.1, max_lag=20)
