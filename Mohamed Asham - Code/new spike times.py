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

# DEBUG -- Detailed info (for debugging)
# INFO -- General info (e.g., progress updates)
# WARNING -- Something unusual, but the program can continue
# ERROR -- Something went wrong, but the program may still run
# CRITICAL -- A serious error; the program might stop

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s - %(message)s")

output_dir = r"C:\Users\mamoh\Desktop\new 7"
folder_path_main = r"C:\Users\mamoh\PycharmProjects\GitHub\iBSc-Research-project\Mohamed Asham - Code\NEW_SPIKETIMEFILES"

#------------------------------------------------------------------------------------------------------------
def find_matching_files(folder_path):
    """
    (1)
    This file is the first to run and takes folder_path_main and sorts it based on if its shank 1 or 0
    and chesk what type it is, so spikeTime, spikeLabels or Behavior then yields this result:

    (  "path_to_Shank0_SpikeTimes", "path_to_Shank1_SpikeTimes",
       "path_to_Shank0_SpikeLabels", "path_to_Shank1_SpikeLabels",
       "path_to_Shank0_Behaviour", "path_to_Shank1_Behaviour"  )
    """
    files = os.listdir(folder_path)
    patterning = r"(M240\d+_\d{8}_File\d+)_Shank(\d+)"

    grouped_files = {}
    for file_name in files:
        match = re.match(patterning, file_name)
        if match:
            identifier = match.group(1)      # so this is interesting in the old code it sorts it also by date that way it finds matching dates
            shank = match.group(2)
            if identifier not in grouped_files:
                grouped_files[identifier] = {"Shank0": {}, "Shank1": {}}

            if "SpikeTimes" in file_name:
                grouped_files[identifier][f"Shank{shank}"]["SpikeTimes"] = os.path.join(folder_path, file_name)
            elif "SpikeLabels" in file_name:
                grouped_files[identifier][f"Shank{shank}"]["SpikeLabels"] = os.path.join(folder_path, file_name)
            elif "Behaviour" in file_name:
                grouped_files[identifier][f"Shank{shank}"]["Behaviour"] = os.path.join(folder_path, file_name)

    # logging.debug(f"\nGrouped Files:\n{grouped_files}")

    for identifier, shank_data in grouped_files.items():
        if all(key in shank_data["Shank0"] for key in ["SpikeTimes", "SpikeLabels", "Behaviour"]) and \
                all(key in shank_data["Shank1"] for key in ["SpikeTimes", "SpikeLabels", "Behaviour"]):
            # logging.debug(f"Yielding files for: {identifier}")
            yield (
                shank_data["Shank0"]["SpikeTimes"], shank_data["Shank1"]["SpikeTimes"],
                shank_data["Shank0"]["SpikeLabels"], shank_data["Shank1"]["SpikeLabels"],
                shank_data["Shank0"]["Behaviour"], shank_data["Shank1"]["Behaviour"]    )
#------------------------------------------------------------------------------------------------------------
def load_spike_data(file_path):
    if re.search(r"_SpikeTimes\.csv$", file_path):
        spike_data = pd.read_csv(file_path, header=None, names=["unit_id", "spike_time"])
        return spike_data
    else:
        raise ValueError(f"Invalid file for SpikeTimes: {file_path}")
def load_behaviour_file(file_path):
    if re.search(r"_Behaviour\.csv$", file_path):
        behaviour_data = pd.read_csv(file_path)
        return behaviour_data
    else:
        raise ValueError(f"Invalid file for Behaviour: {file_path}")
def load_spike_labels_file(file_path):
    if re.search(r"_SpikeLabels\.csv$", file_path):
        spike_labels_data = pd.read_csv(file_path)
        return spike_labels_data
    else:
        raise ValueError(f"Invalid file for SpikeLabels: {file_path}")
#------------------------------------------------------------------------------------------------------------
def combine_spike_data(file_path1, file_path2, spike_labels_1, spike_labels_2):
    """
    (2)
    This is the second fucntion that runs and this takes the data and combines the shank 0 and shank 1 or spike times
    and spike labels onto one dataframe. so the spikelabel file already has titles so no need to add them so what im
    talking about is that it calls the on the functions above load_spike_data and load_behaviour_file which puts the data
    into a data frame
    """
    spike_data1 = load_spike_data(file_path1)
    spike_data2 = load_spike_data(file_path2)
    spike_labels1 = load_spike_labels_file(spike_labels_1)
    spike_labels2 = load_spike_labels_file(spike_labels_2)

    # Offset neuron IDs for the second dataset
    spike_data2["unit_id"] += 1000
    spike_labels2["ClusterID"] += 1000

    # Merge spike data with labels
    spike_data1 = spike_data1.merge(spike_labels1, left_on="unit_id", right_on="ClusterID", how="left")
    spike_data2 = spike_data2.merge(spike_labels2, left_on="unit_id", right_on="ClusterID", how="left")

    # Combine both spike datasets
    combined_spike_data = pd.concat([spike_data1, spike_data2], ignore_index=True)

    return combined_spike_data
def bin_spikes(spike_data, bin_width):
    spike_data = spike_data[["unit_id", "spike_time"]]

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

    logging.debug(f"[DEBUG] Spike Matrix Shape: {spike_matrix.shape} \n")

    correlation_matrix = np.corrcoef(spike_matrix)
    correlations = {}
    for i, j in combinations(range(len(units)), 2):
        correlations[(units[i], units[j])] = correlation_matrix[i, j]

    return correlation_matrix, correlations
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
# def visualise_correlation_matrix(correlations, units, font_size, bin_width):
#     correlation_matrix = pd.DataFrame(index=units, columns=units, data=np.nan)
#     for (unit1, unit2), r_value in correlations.items():
#         correlation_matrix.loc[unit1, unit2] = r_value
#         correlation_matrix.loc[unit2, unit1] = r_value
#     np.fill_diagonal(correlation_matrix.values, 1)
#
#     plt.figure(figsize=(20, 16))
#     sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f",
#                 cbar_kws={'label': 'Pearson Correlation'},
#                 xticklabels=units, yticklabels=units)
#     plt.xticks(fontsize=font_size, rotation=90)
#     plt.yticks(fontsize=font_size)
#     plt.xlabel("Neuron ID", fontsize=font_size)
#     plt.ylabel("Neuron ID", fontsize=font_size)
#     plt.title(f"Neuron Pairwise Correlation Matrix at {bin_width}", fontsize=font_size + 8)
#
#     output_file = os.path.join(output_dir,f"_{result1}_Correlation_Matrix_BinWidth_{bin_width}sec.png")
#     plt.savefig(output_file, bbox_inches="tight", dpi=600)
#     plt.close()  # Close the plot to free memory
#     print(f"Correlation matrix visualization saved as {output_file}")

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
def analyze_spike_data(file_path1, file_path2, spike_labels_1, spike_labels_2, behaviour1, behaviour2, bin_width, num_clusters):
    """
    The order of functions:
    1. find_matching_files - this looks at whole folder and finds relevent files, so it groups shank 1 and 0 and within it
    the spikeTimes, spikeLabels and Behavior
    2. combine_spike_data - this takes the spikeTimes and spikeLabels seperatly and first makes it offset by 1000 then it
    joins them into one dataframe
    3. bin_spikes - the output from combine_spike_data which is spike_data is put into bin_spikes function
    4. calculate_pairwise_correlations - the binned_spikes is then used to find the pairwise correlation
    """
    logging.info(f"Processing data with Bin Width: {bin_width}, Clusters: {num_clusters}\n")

    def extract_mouse_id(file_path):
        pattern = r"M240(\d+)_(\d{8}_File\d+)_Shank(\d+)"
        match = re.search(pattern, file_path)
        return f"{match.group(1)}_Shank_{match.group(3)}" if match else "UNKNOWN"

    extracted_id1 = extract_mouse_id(file_path1)
    extracted_id2 = extract_mouse_id(file_path2)
    extracted_id_sl1 = extract_mouse_id(spike_labels_1)
    extracted_id_sl2 = extract_mouse_id(spike_labels_2)
    extracted_id_bh1 = extract_mouse_id(behaviour1)
    extracted_id_bh2 = extract_mouse_id(behaviour2)

    logging.debug(f"SpikeTimes 1:  {extracted_id1} --- SpikeTimes 2:  {extracted_id2}")
    logging.debug(f"SpikeLabels 1: {extracted_id_sl1} --- SpikeLabels 2: {extracted_id_sl2}")
    logging.debug(f"Behaviour 1:   {extracted_id_bh1} --- Behaviour 2:   {extracted_id_bh2}\n")

    logging.info("Loading and Combining Spike Data with Labels...\n")
    spike_data = combine_spike_data(file_path1, file_path2, spike_labels_1, spike_labels_2)
    logging.debug("Combined Spike Data (First & Last 5 Rows):\n%s\n", pd.concat([spike_data.head(), spike_data.tail()]))

    logging.info("Loading Behaviour Data...\n")

    behaviour_data1 = load_behaviour_file(behaviour1)
    logging.debug("Behaviour Data (Shank 0) - First & Last 5 Rows:\n%s\n",
                  pd.concat([behaviour_data1.head(), behaviour_data1.tail()]))
    behaviour_data2 = load_behaviour_file(behaviour2)
    logging.debug("Behaviour Data (Shank 1) - First & Last 5 Rows:\n%s\n",
                  pd.concat([behaviour_data2.head(), behaviour_data2.tail()]))

    logging.info("Binning Spike Data...\n")
    binned_spikes = bin_spikes(spike_data, bin_width)

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

    # logging.info("Performing Clustering...\n")
    # cluster_neurons(correlation_matrix, num_clusters=num_clusters, bin_width=bin_width,
    #                 units=list(binned_spikes.keys()))

    logging.info(f"""Analysis complete for {extracted_id1} & {extracted_id2}
----------------------------------------------------------------------------------------\n""")
def shutdown_computer():
    print("Shutting down the computer...")
    os.system("shutdown /s /t 1")
#---------------------------------------------------------------------------------



# bin_widths = [0.05, 0.01, 0.1, 0.5, 1, 5, 10]

bin_widths = [10]

# cluster_numer = [2, 3, 4]

cluster_numer = [2]

for num in cluster_numer:
    for values in bin_widths:
        for files in find_matching_files(folder_path_main):
            if None in files:
                logging.warning("Skipping this iteration due to missing files...")
                continue

            spike_times0, spike_times1, spike_labels0, spike_labels1, behaviour0, behaviour1 = files

            try:
                analyze_spike_data(spike_times0, spike_times1, spike_labels0, spike_labels1, behaviour0, behaviour1,
                                   bin_width = values, num_clusters = num)
            except Exception as e:
                logging.error(f"Error processing files {spike_times0} & {spike_times1}: {e}")


# shutdown_computer()