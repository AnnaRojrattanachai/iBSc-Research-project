from menuinst.platforms.win_utils.knownfolders import folder_path
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from itertools import combinations
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import os
import pandas as pd
import numpy as np
import logging

output_dir = r"C:\Users\mamoh\PycharmProjects\GitHub\iBSc-Research-project\Mohamed Asham - Code\Result of SPIKETIMEFILES"
folder_path_main = r"C:\Users\mamoh\PycharmProjects\GitHub\iBSc-Research-project\Mohamed Asham - Code\SPIKETIMEFILES"

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

#---------------------------------------------------------------------------------
def find_matching_files(folder_path):
    files = os.listdir(folder_path)

    # Regex pattern to match file names
    pattern = r"(M240\d+)_\d{8}_File\d+_Shank[01]_goodSpikeTimes"

    # Group files by M240 ID and date
    grouped_files = {}
    for file_name in files:
        match = re.match(pattern, file_name)
        if match:
            m240_id = match.group(1)
            date = re.search(r"_(\d{8})_", file_name).group(1)
            shank = re.search(r"_Shank(\d)_", file_name).group(1)

            key = (m240_id, date)
            if key not in grouped_files:
                grouped_files[key] = {}
            grouped_files[key][shank] = os.path.join(folder_path, file_name)

    # Yield all possible matching pairs
    for key, shank_files in grouped_files.items():
        if "0" in shank_files and "1" in shank_files:
            yield shank_files["0"], shank_files["1"]

    print("No matching files found.")
    return None, None
def load_spike_data(file_path):
    spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
    return spike_data
def combine_spike_data(file_path1, file_path2):
    # Load the data from both files
    spike_data1 = load_spike_data(file_path1)
    spike_data2 = load_spike_data(file_path2)

    # Adjust neuron IDs in the second dataset
    spike_data2["unit_id"] += 1000

    # Concatenate the two datasets
    combined_spike_data = pd.concat([spike_data1, spike_data2], ignore_index=True)

    return combined_spike_data
def bin_spikes(spike_data, bin_width):
    max_time = spike_data["spike_time"].max()
    bins = np.arange(0, max_time + bin_width, bin_width)
    binned_spikes = {}

    for unit_id, group in spike_data.groupby("unit_id"):
        spike_counts, _ = np.histogram(group["spike_time"], bins=bins)
        binned_spikes[unit_id] = spike_counts  # Store the frequency in bins

    return binned_spikes
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
def calculate_pairwise_correlations(binned_spikes):
    units = list(binned_spikes.keys())
    spike_matrix = np.array([binned_spikes[unit] for unit in units])
    correlation_matrix = np.corrcoef(spike_matrix)

    correlations = {}
    for i, j in combinations(range(len(units)), 2):
        correlations[(units[i], units[j])] = correlation_matrix[i, j]

    return correlation_matrix, correlations
def visualise_correlation_matrix(correlations, units, font_size, bin_width):
    correlation_matrix = pd.DataFrame(index=units, columns=units, data=np.nan)
    for (unit1, unit2), r_value in correlations.items():
        correlation_matrix.loc[unit1, unit2] = r_value
        correlation_matrix.loc[unit2, unit1] = r_value
    np.fill_diagonal(correlation_matrix.values, 1)

    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f",
                cbar_kws={'label': 'Pearson Correlation'},
                xticklabels=units, yticklabels=units)
    plt.xticks(fontsize=font_size, rotation=90)
    plt.yticks(fontsize=font_size)
    plt.xlabel("Neuron ID", fontsize=font_size)
    plt.ylabel("Neuron ID", fontsize=font_size)
    plt.title(f"Neuron Pairwise Correlation Matrix at {bin_width}", fontsize=font_size + 8)

    output_file = os.path.join(output_dir,f"_{result1}_Correlation_Matrix_BinWidth_{bin_width}sec.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()  # Close the plot to free memory
    print(f"Correlation matrix visualization saved as {output_file}")
def cluster_neurons(correlation_matrix, num_clusters, bin_width, units):
    print("Performing K-means clustering...")

    # Step 2: Use k-means++ initialization with multiple initializations
    kmeans = KMeans(
        n_clusters=num_clusters,
        init="k-means++",
        n_init=20,  # Run k-means 20 times with different initializations
        random_state=42,
    )
    # kmeans_labels = kmeans.fit_predict(normalized_matrix)
    kmeans_labels = kmeans.fit_predict(correlation_matrix)

    # Reorder the correlation matrix and neuron IDs based on clustering
    sorted_indices = np.argsort(kmeans_labels)
    reordered_corr_matrix = correlation_matrix[sorted_indices, :][:, sorted_indices]
    reordered_units = np.array(units)[sorted_indices]

    # Determine split point based on the unit ID (file origin)
    split_point = sum(np.array(units) < 1000)  # Units < 1000 from file_path1 (V1), units >= 1000 from file_path2 (PFC)

    # Create the heatmap
    plt.figure(figsize=(20, 16))
    sns.heatmap(reordered_corr_matrix, cmap="coolwarm", annot=False, fmt=".2f",
                xticklabels=reordered_units, yticklabels=reordered_units,
                cbar_kws={'label': 'Pearson Correlation'})

    # Add annotations for V1 and PFC
    plt.text(split_point / 2, 0, 'V1, Hippocampus, Thalamus', ha='center', va='bottom', fontsize=10, color='blue', transform=plt.gca().transData)
    plt.text(split_point + (len(units) - split_point) / 2, len(units) + 10, 'PFC', ha='center', va='bottom', fontsize=10,
             color='red')

    # Add dashed lines to separate V1 and PFC regions
    # plt.axhline(split_point, color='black', linestyle='--', linewidth=1)
    # plt.axvline(split_point, color='black', linestyle='--', linewidth=1)

    # Highlight V1 and PFC groups on axes with brackets
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), split_point, split_point,
                                       edgecolor='blue', fill=False, linewidth=2))
    plt.gca().add_patch(plt.Rectangle((split_point - 0.5, split_point - 0.5),
                                       len(units) - split_point, len(units) - split_point,
                                       edgecolor='red', fill=False, linewidth=2))

    # Finalize plot
    plt.title(f"Neuron Clustering (K-means++) at {bin_width}s", fontsize=16)
    plt.xticks(fontsize=2, rotation=90)
    plt.yticks(fontsize=2)
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")

    # Save the plot
    output_file = os.path.join(output_dir, f"_{result1}_Kmeans_Clustering_{bin_width}s.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"K-means clustering visualization saved as {output_file}")

    return kmeans_labels
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
def calculate_and_visualize_cross_correlograms(binned_spikes, max_lag, bin_width):
    units = list(binned_spikes.keys())
    num_units = len(units)
    lags = range(-max_lag, max_lag + 1)

    # Convert binned spikes to a NumPy array for efficient computation
    binned_spikes_matrix = np.array([binned_spikes[unit] for unit in units])

    # Normalise spike trains (zero-mean correlation)
    normalised_spikes = binned_spikes_matrix - binned_spikes_matrix.mean(axis=1, keepdims=True)

    # Precompute cross-correlations for all lags
    cross_corr_results = []

    print("Calculating cross-correlograms...")
    total_operations = len(lags) * (num_units * (num_units - 1)) // 2
    with tqdm(total=total_operations, desc="Computing Cross-Correlograms") as pbar:
        for lag in lags:
            lag_matrix = np.zeros((num_units, num_units))
            for i, j in combinations(range(num_units), 2):
                cross_corr = np.correlate(normalised_spikes[i], normalised_spikes[j], mode='full')
                lag_index = len(cross_corr) // 2 + lag
                if 0 <= lag_index < len(cross_corr):
                    lag_matrix[i, j] = cross_corr[lag_index]
                    lag_matrix[j, i] = lag_matrix[i, j]  # Symmetric for (i, j) and (j, i)
                pbar.update(1)
            cross_corr_results.append(lag_matrix)

    avg_corr_matrix = np.mean(np.array(cross_corr_results), axis=0)

    return avg_corr_matrix, cross_corr_results, lags
def visualize_all_cross_correlogram_components(cross_corr_results, units, lags, bin_width):
    num_units = len(units)

    # 1. Cross-Correlation Heatmap for Each Lag
    print("Creating and saving heatmap subplots for each lag...")
    n_rows = 4  # Number of rows for subplots
    n_cols = int(np.ceil(len(lags) / n_rows))  # Dynamically determine columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()  # Flatten to index each subplot easily

    for idx, lag in enumerate(lags):
        ax = axes[idx]
        heatmap = sns.heatmap(cross_corr_results[idx], vmin=-1, vmax=1, cmap="coolwarm", cbar=False, ax=ax)
        ax.set_title(f"Lag {lag}", fontsize=8)
        ax.axis("off")

    # Add a single shared colour bar
    fig.colorbar(heatmap.get_children()[0], ax=axes, orientation='horizontal', pad=0.05,
                 label='Cross-Correlation Coefficient')
    fig.suptitle("Cross-Correlation Heatmap for Each Lag", fontsize=16)
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    output_file = os.path.join(output_dir, f"Cross_Correlation_Heatmaps_BinWidth_{bin_width}sec.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Heatmap visualisation for each lag saved as {output_file}")

    # 2. Max Cross-Correlation Heatmap
    print("Creating and saving lag heatmap showing maximum correlations...")
    max_corr_matrix = np.max(cross_corr_results, axis=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(max_corr_matrix, vmin=-1, vmax=1, cmap="coolwarm",
                xticklabels=units, yticklabels=units, cbar_kws={'label': 'Max Cross-Correlation'})
    plt.title("Lag Heatmap (Max Cross-Correlation for Each Pair)")
    plt.xlabel("Neuron ID")
    plt.ylabel("Neuron ID")
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)

    output_file = os.path.join(output_dir, f"Max_Cross_Correlation_Heatmap_BinWidth_{bin_width}sec.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Maximum cross-correlation heatmap saved as {output_file}")
#---------------------------------------------------------------------------------


#---------------------------------------------------------------------------------
def analyze_spike_data(file_path1, file_path2=None, bin_width = 0, num_clusters = 0, compute_correlograms=False):

    logging.info(f"Processing data with Bin Width: {bin_width}, Clusters: {num_clusters}\n")

    logging.info("Loading and Combining Spike Data with Labels...\n")
    if file_path2:
        # logging.info(f"Combining spike data from {file_path1} and {file_path2}...")
        spike_data = combine_spike_data(file_path1, file_path2)
    else:
        # logging.info(f"Loading spike data from {file_path1}...")
        spike_data = load_spike_data(file_path1)

    def extract_mouse_id(file_path):
        pattern = r"M240(\d+)_(\d{8}_File\d+)_Shank(\d+)"
        match = re.search(pattern, file_path)
        return f"{match.group(1)}_Shank_{match.group(3)}" if match else "UNKNOWN"

    extracted_id1 = extract_mouse_id(file_path1)
    extracted_id2 = extract_mouse_id(file_path2)

    logging.debug(f"SpikeTimes 1:  {extracted_id1} --- SpikeTimes 2:  {extracted_id2}\n")

    logging.debug("Combined Spike Data (First & Last 5 Rows):\n%s\n", pd.concat([spike_data.head(), spike_data.tail()]))

    logging.info("Binning Spike Data...\n")
    binned_spikes = bin_spikes(spike_data, bin_width=bin_width)
    sample_units = list(binned_spikes.keys())[:5]
    log_messages = []
    logging.debug("Example of Binned Spike Data (First 5 Units):")
    for unit in sample_units:
        log_messages.append(f"Unit {unit}: {binned_spikes[unit][:10]}")
    logging.debug("\n        ".join(log_messages) + "\n")

    logging.info("Calculating Pairwise Correlations...\n")
    correlation_matrix, correlations = calculate_pairwise_correlations(binned_spikes)
    logging.debug("Pairwise Correlation Matrix (First & Last 5 Rows):\n%s\n",
                  pd.concat([pd.DataFrame(correlation_matrix).head(), pd.DataFrame(correlation_matrix).tail()]))

    # logging.info("Visualizing pairwise correlation matrix...\n")
    # visualise_correlation_matrix(correlations, list(binned_spikes.keys()), font_size=2, bin_width=bin_width)

    if compute_correlograms:
        logging.info("Calculating and visualizing cross-correlograms...")
        avg_corr_matrix, cross_corr_results, lags = calculate_and_visualize_cross_correlograms(
            binned_spikes, max_lag=20, bin_width=bin_width)
        visualize_all_cross_correlogram_components(cross_corr_results, list(binned_spikes.keys()), lags,
                                                   bin_width=bin_width)

    # logging.info("Clustering and visualising pairwise correlation matrix...")
    # cluster_neurons(correlation_matrix, num_clusters=num_clusters, bin_width=bin_width, units=list(binned_spikes.keys()))

    logging.info(f"""Analysis complete for {extracted_id1} & {extracted_id2}
----------------------------------------------------------------------------------------\n""")

def shutdown_computer():
    print("Shutting down the computer...")
    os.system("shutdown /s /t 1")
#---------------------------------------------------------------------------------


pattern = r"M240(\d{2})"

# bin_widths = [0.05, 0.01, 0.1, 0.5, 1, 5, 10]
#
# cluster_numer = [2, 3, 4]

bin_widths = [10]

cluster_numer = [2]

for num in cluster_numer:
    for values in bin_widths:
        print(f"Processing with bin width: {values}")

        for file_path1, file_path2 in find_matching_files(folder_path_main):
            # print(f"Processing pair: {file_path1} and {file_path2}")

            try:
                match1 = re.search(pattern, file_path1)
                match2 = re.search(pattern, file_path2)
            except Exception as e:
                print(f"Error during regex matching: {e}")
                continue

            if match1 and match2:
                result1 = match1.group(1)
                result2 = match2.group(1)
                print(f"Extracted results: {result1}, {result2}")
            else:
                print(f"Pattern not found in one or both files: {file_path1}, {file_path2}")
                continue

            try:
                print(f"Processing pair: {file_path1} and {file_path2} with bin width {values}")
                analyze_spike_data(file_path1, file_path2, bin_width=values, num_clusters = num, compute_correlograms=False)
            except Exception as e:
                print(f"Error processing files {file_path1} and {file_path2} with bin width {values}: {e}")


# shutdown_computer()