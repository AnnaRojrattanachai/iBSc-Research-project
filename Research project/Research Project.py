import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr
from tqdm import tqdm

file_path = r"C:\Users\mamoh\PycharmProjects\Trial Project\Research project\SPIKETIMEFILES\M24028_20240426_File1_Shank0_goodSpikeTimes.csv"

def load_spike_data(file_path):
    spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
    return spike_data

def bin_spikes(spike_data, bin_width):
    max_time = spike_data["spike_time"].max()
    # num_bins = int(np.ceil(max_time / bin_width))     #----------for debugging
    bins = np.arange(0, max_time + bin_width, bin_width)

    binned_spikes = {}

    # Group data by unit and bin the spikes for each unit
    for unit_id,group in spike_data.groupby("unit_id"):
        spike_counts, _ = np.histogram(group["spike_time"], bins=bins)  # spike_counts finds the frequency in each/ also the bins is a range so eg. 0.0 0.5 means from 0.0-0.5
        binned_spikes[unit_id] = spike_counts # this puts it into the dictionary so for each unit id it will have frequency in bins

    return binned_spikes

# bin_spikes(load_spike_data(file_path), bin_width=5)

def calculate_pairwise_correlations(binned_spikes):
    correlations = {}
    units = list(binned_spikes.keys())

    # Calculate Pearson correlation for each pair of units
    for (unit1, unit2) in combinations(units, 2):
        r_value, _ = pearsonr(binned_spikes[unit1], binned_spikes[unit2])
        correlations[(unit1, unit2)] = r_value

    return correlations

# ---------------optional code for progress bar-----------------
def calculate_pairwise_correlations_with_progress(binned_spikes):
    correlations = {}
    units = list(binned_spikes.keys())
    total_operations = len(units) * (len(units) - 1) // 2  # Calculate total pairwise correlations

    # Initialize progress bar for pairwise correlations
    with tqdm(total=total_operations, desc="Calculating Pairwise Correlations") as pbar:
        for (unit1, unit2) in combinations(units, 2):
            r_value, _ = pearsonr(binned_spikes[unit1], binned_spikes[unit2])
            correlations[(unit1, unit2)] = r_value
            pbar.update(1)  # Update the progress bar with each calculation

    return correlations
# ---------------optional code for progress bar-----------------

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
    plt.draw()

def analyze_spike_data(file_path, bin_width=0.1, use_progress=False):

    spike_data = load_spike_data(file_path)
    binned_spikes = bin_spikes(spike_data, bin_width=bin_width)
    correlations = calculate_pairwise_correlations(binned_spikes)

    if use_progress:
        try:
            correlations = calculate_pairwise_correlations_with_progress(binned_spikes)
        except NameError:
            print("Progress function not available; Running without progress bar")
            correlations = calculate_pairwise_correlations(binned_spikes)
    else:
        correlations = calculate_pairwise_correlations(binned_spikes)

    print(f"Correlation matrix for file {os.path.basename(file_path)}:")
    for (unit1, unit2), corr in correlations.items():
        print(f"Unit {unit1} - Unit {unit2}: Pearson correlation = {corr:.4f}")

    visualise_correlation_matrix(correlations, list(binned_spikes.keys()), font_size=5)


analyze_spike_data(file_path, bin_width=10, use_progress=True)


#-----------------------------------------------------------------------------------------------------------------------
#
# folder_path = r""C:\Users\mamoh\PycharmProjects\Trial Project\Research project\SPIKETIMEFILES""
#
#
# def load_spike_data(file_path):
#     spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
#     return spike_data
#
#
# def bin_spikes(spike_data, bin_width):
#     max_time = spike_data["spike_time"].max()
#     bins = np.arange(0, max_time + bin_width, bin_width)
#     binned_spikes = {}
#     for unit_id, group in spike_data.groupby("unit_id"):
#         spike_counts, _ = np.histogram(group["spike_time"], bins=bins)
#         binned_spikes[unit_id] = spike_counts
#     return binned_spikes
#
#
# def calculate_pairwise_correlations(binned_spikes, progress_bar):
#     correlations = {}
#     units = list(binned_spikes.keys())
#
#     for (unit1, unit2) in combinations(units, 2):
#         r_value, _ = pearsonr(binned_spikes[unit1], binned_spikes[unit2])
#         correlations[(unit1, unit2)] = r_value
#         progress_bar.update(1)  # Update the single shared progress bar
#
#     return correlations
#
#
# def visualise_correlation_matrix(correlations, units, font_size):
#     correlation_matrix = pd.DataFrame(index=units, columns=units, data=np.nan)
#     for (unit1, unit2), r_value in correlations.items():
#         correlation_matrix.loc[unit1, unit2] = r_value
#         correlation_matrix.loc[unit2, unit1] = r_value
#     np.fill_diagonal(correlation_matrix.values, 1)
#
#     plt.figure(figsize=(20, 12))
#     sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False,
#                 cbar_kws={'label': 'Pearson Correlation'},
#                 xticklabels=units, yticklabels=units)
#     plt.xticks(fontsize=font_size, rotation= 90)
#     plt.yticks(fontsize=font_size)
#     plt.xlabel("Neuron ID", fontsize=font_size + 5)
#     plt.ylabel("Neuron ID", fontsize=font_size + 5)
#     plt.title("Neuron Pairwise Correlation Matrix", fontsize=font_size + 8)
#     plt.show()
#     # plt.show(block=False)
#     # plt.pause(1)   # --- optional, to move on after 1 second
#     # plt.savefig(f"correlation_matrix_{filename}.png")  # --- another option is saving it all to a file
#
#
# def analyze_spike_data_folder(folder_path, bin_width):
#     file_list = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
#     total_operations = 0
#
#     for filename in file_list:
#         file_path = os.path.join(folder_path, filename)
#         spike_data = load_spike_data(file_path)
#         binned_spikes = bin_spikes(spike_data, bin_width=bin_width)
#         num_units = len(binned_spikes.keys())
#         total_operations += num_units * (num_units - 1) // 2  # Total pairwise correlations for this file
#
#     # Initialize a single progress bar for all operations
#     with tqdm(total=total_operations, desc="Processing All Files and Correlations") as pbar:
#         # Process each file and update the same progress bar
#         for filename in file_list:
#             file_path = os.path.join(folder_path, filename)
#             print(f"\nProcessing file: {file_path}")
#
#             spike_data = load_spike_data(file_path)
#             binned_spikes = bin_spikes(spike_data, bin_width=bin_width)
#             correlations = calculate_pairwise_correlations(binned_spikes, progress_bar=pbar)
#
#             visualise_correlation_matrix(correlations, list(binned_spikes.keys()), font_size= 4)
#
#
# analyze_spike_data_folder(folder_path, bin_width=0.1)