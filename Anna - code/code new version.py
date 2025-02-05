import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
import seaborn as sns

# Show all columns
pd.set_option('display.max_columns', None)



spiketime_file = "/Users/pitsinee/PycharmProjects/HelloWorld/research project/M24028_20240426_File1_Shank0_goodSpikeTimes.csv"

spiketime_M240208_20240427 = "/Users/pitsinee/Documents/UCL/iBSc research/M24028_20240427_File1_Shank0_SpikeTimes.csv"

spikelabels_M240208_20240427 = "/Users/pitsinee/Documents/UCL/iBSc research/M24028_20240427_File1_Shank0_SpikeLabels.csv"

spiketime_M24029_20240424 = "/Users/pitsinee/Documents/UCL/iBSc research/M24029_20240424_File1_Shank0_SpikeTimes.csv"
spikelabels_M24029_20240424 = "/Users/pitsinee/Documents/UCL/iBSc research/M24029_20240424_File1_Shank0_SpikeLabels.csv"


def openfile (spiketime_file, spikelabels_file):
    global df_spiketime
    global df_spikelabels
    df_spiketime = pd.read_csv(spiketime_file, names = ["unit_id", "spikeTime"])
    df_spikelabels = pd.read_csv(spikelabels_file) #names = ["unit_id", "channel", "unit_type", "unit_area"]
    df_spikelabels = df_spikelabels.rename(columns={'ClusterID': 'unit_id'})
    print ("original spiketime file:\n", df_spiketime)
    print ("spike labels: \n", df_spikelabels)
openfile(spiketime_M24029_20240424, spikelabels_M24029_20240424)



def group_STPR(og_df, neuron_ID): #changing all neuron IDs to -1 (neuron_ID = the one for the spike)
    new_df = og_df.copy()
    new_df.loc[new_df['unit_id'] != neuron_ID, 'unit_id'] = -1 #changing all the non-chosen neuron ids to -1
    return new_df
    # STPR_group_df = new_df.groupby('unit_id')['spikeTime'].apply(list).reset_index() # grouping the neuron ids tgt
    # return STPR_group_df

def group(file): #group the spike times per neuron
    grouped_df = file.groupby('unit_id')['spikeTime'].apply(list).reset_index()
    # grouped_df.to_csv('rearranged_output.csv', index=True)
    global maxtime
    maxtime = max(file['spikeTime'].tolist())
    return grouped_df
    # print("grouped file:\n", grouped_df)
    # print ("max time:", maxtime)

def merge(df_spiketime, df_spikelabels):
    merged_df = pd.merge(group(df_spiketime), df_spikelabels, on='unit_id', how="inner")
    return merged_df

def groupBin(grouped_file,binwidth):
    bins = np.arange(0, maxtime, binwidth) #returns array of bin edges
    allcount = [0] *(grouped_file.shape[0]) #list of lists of counts in each bin
    for i in range(grouped_file.shape[0]): #iterates through neuron ID
        x = grouped_file.iloc[i]['spikeTime'] #get the list of values of spike time for row i
        # print ("x:", x) #check for spike times
        counts, bin_edges = np.histogram(x, bins = bins)
        allcount[i] = counts
        # print ("all count:", len(allcount), "column[0]:", grouped_df.shape[0]) #check numbers of items if the same
        #print("Counts per bin for row", i, ":", counts)
    data = {
        "unit_id": grouped_file[grouped_file.columns[0]].tolist(),
        "spike number in each bin": allcount
    }
    binned_df = pd.DataFrame(data)
    binned_df.to_csv ("binnedData.csv", index = False)
    return binned_df
# binned_df = groupBin(grouped_df, 10)

def unlist(list):
    a = []
    for sublist in list:
        a += sublist
    a.sort()
    return a

merged_df = merge(df_spiketime, df_spikelabels)
df_neurons_THA = merged_df[merged_df['UnitArea'] == "TH"]
df_neurons_VCX =merged_df[merged_df['UnitArea'] == "VCX"]

print ("df_neurons_THA:\n",df_neurons_THA)
print ("df_neurons_VCX:\n",df_neurons_VCX)


def correlations (file): #get the correlations of each combination in the file, file = binned_df
    correlations = np.corrcoef(file[file.columns[1]].tolist()) #returns a matrix of correlations
    correlation_df = pd.DataFrame (correlations, index = file['unit_id'], columns = file['unit_id']) #give index as the unit_id
    correlation_df.to_csv("correlation_matrix", index = True)
    return correlation_df


def lag(grouped_neuron_df, lag_interval): # grouped_neuron_df = 2 row, the neuron, and the population
    lagged_grouped_neuron_df = grouped_neuron_df.copy()
    lagged_grouped_neuron_df.loc[grouped_neuron_df["unit_id"] == -1, 'spikeTime'] = grouped_neuron_df.loc[grouped_neuron_df["unit_id"] == -1, 'spikeTime'].apply(lambda x: x + lag_interval)
    return lagged_grouped_neuron_df

def correlations_w_lags_same_area (neuron_ID, df_neurons, binwidth): #get correlations between neuron 1 and groups of neurons in df_neurons
    # getting correlations between a neuron and the same brain area
    df_neurons = df_neurons.copy()
    df_neurons.loc[df_neurons["unit_id"] != neuron_ID, "unit_id"] = -1 # changing all the non-chosen neuron ids to -1
    grouped_neuron_df = df_neurons.groupby('unit_id')['spikeTime'].apply(list).reset_index() #combining the "-1" neurons together
    grouped_neuron_df['spikeTime'] = grouped_neuron_df['spikeTime'].apply(lambda x: unlist(x)) #correct the column "spiketime" that is now a list of list --> into just list
    print ("grouped\n", grouped_neuron_df)
    df_STPR = pd.DataFrame({
        'timeLag' : [],
        'STPR': []
    })

    lagrange = 0.5 #from -0.5 s to 0.5 s timelags
    increment = 0.01 #increasing each lag by 0.1 s
    #delete the time spikes that would not have the overlap
    for i in np.arange(-lagrange,lagrange + increment,increment):
        lagged_grouped_neuron_df = lag(grouped_neuron_df, i)
        # print(lagged_grouped_neuron_df)
        grouped_bin_neuron_df = groupBin(lagged_grouped_neuron_df, binwidth)
        # print (grouped_bin_neuron_df)

        ##deleting the left over on left and rigth side
        n = int(lagrange/binwidth) # n = number of bins to be removed
        grouped_bin_neuron_df["spike number in each bin"] = grouped_bin_neuron_df["spike number in each bin"].apply(lambda x: x[n:-n])

        correlation = correlations(grouped_bin_neuron_df).iloc[0,1]
        df_STPR.loc[len(df_STPR)] = [i,correlation]
    return df_STPR

def correlations_w_lags_diff_area (neuron_ID, df_neurons1, df_neurons2, binwidth): #get correlations between neuron 1 and groups of file in df_neurons
    # getting correlations between a neuron and the same brain area
    df_neurons2 = df_neurons2.copy()
    df_neurons2['unit_id'] = -1  # changing all the neurons in df_neurons2 (population) to -1
    print ("new df_neurons 2:\n", df_neurons2)
    # creating a one line df with the single neuron from df_neurons1
    df_single_neuron = df_neurons1.loc[df_neurons1["unit_id"] == neuron_ID]
    print ("single neuron:", df_single_neuron)
    #combine the single neuron and the neuron population
    merged_df_neuron = pd.concat([df_single_neuron, df_neurons2], axis = 0, ignore_index=False)
    grouped_neuron_df = merged_df_neuron.groupby('unit_id')['spikeTime'].apply(
        list).reset_index()  # combining the "-1" neurons together
    grouped_neuron_df['spikeTime'] = grouped_neuron_df['spikeTime'].apply(
        lambda x: unlist(x))  # correct the column "spiketime" that is now a list of list --> into just list
    print("grouped\n", grouped_neuron_df)
    df_STPR = pd.DataFrame({
        'timeLag': [],
        'STPR': []
    })
    lagrange = 0.5  # from -0.5 s to 0.5 s timelags
    increment = 0.01  # increasing each lag by 0.1 s
    # delete the time spikes that would not have the overlap
    for i in np.arange(-lagrange, lagrange + increment, increment):
        lagged_grouped_neuron_df = lag(grouped_neuron_df, i)
        # print(lagged_grouped_neuron_df)
        grouped_bin_neuron_df = groupBin(lagged_grouped_neuron_df, binwidth)
        # print(grouped_bin_neuron_df)

        ##deleting the left over on left and rigth side
        n = int(lagrange / binwidth)
        grouped_bin_neuron_df["spike number in each bin"] = grouped_bin_neuron_df["spike number in each bin"].apply(
            lambda x: x[n:-n])
        correlation = correlations(grouped_bin_neuron_df).iloc[0, 1]
        df_STPR.loc[len(df_STPR)] = [i, correlation]
    return df_STPR


df_STPR = correlations_w_lags_same_area (15, df_neurons_VCX, 0.1)
# df_diff_STPR = correlations_w_lags_diff_area(15, df_neurons_VCX, df_neurons_THA, 0.1)
print ("df same STPR:\n", df_STPR)
# print ("df diff STPR:\n", df_diff_STPR)

# for i in df_neurons_VCX[df_neurons_VCX["unit_id"]]:
#     correlations_w_lags_same_area(i, df_neurons_VCX, 0.01)




def plot_STPR(df, xaxis):
    plt.plot(df[str(xaxis)], df['STPR'], marker='o', label='Line')  # Line graph with markers
    plt.title('timeLag vs STPR Graph')  # Title
    plt.xlabel('timeLag')       # X-axis label
    plt.ylabel('STPR')       # Y-axis label
    plt.legend()               # Legend
    plt.grid(True)             # Grid
    plt.show()                 # Display the graph

plot_STPR(df_STPR, 'timeLag')
# plot_STPR(df_diff_STPR, 'timeLag')

""""--------------------------------"""
spike_times_df = pd.DataFrame({
    'neuron_id': [0, 1, 2],
    'spike_counts': [[2, 5, 6, 1], [0, 3, 7, 2], [4, 1, 0, 8]]  # Spike counts per bin
})

df_behaviour = pd.DataFrame ({
    'bin_index': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'alert_level': [1, 2, 3, 1, 3, 2, 1, 3, 2, 2, 1]
})

#have a df of behaviour (time bins vs behavioural state)
def split_by_behaviour (neuron_spike_df, binwidth, behaviour_df): #behaviour df = 2 columns, bin_index & behavioural state
    # make sure behaviour_df has the same binwidth
    binned_df = groupBin(group(neuron_spike_df), binwidth)
    expanded_data = []
    for _,row in binned_df.iterrows():
        unit_id = row['unit_id']
        for bin_index, spike_count in enumerate(row['spike number in each bin']):
            expanded_data.append ({'unit_id': unit_id, 'bin_index': bin_index, 'spike_count': spike_count})
    # print (expanded_data)
    expanded_data_df = pd.DataFrame (expanded_data)

    merged_df = pd.merge(expanded_data_df, behaviour_df, on='bin_index')

    #returns a df with specific alert level (3 columns: neuron ID, bin index, spike count)
    alert_1_df = merged_df[merged_df['alert_level'] == 1].drop(columns=['alert_level'])
    alert_2_df = merged_df[merged_df['alert_level'] == 2].drop(columns=['alert_level'])
    alert_3_df = merged_df[merged_df['alert_level'] == 3].drop(columns=['alert_level'])


    #group the spike counts arranged by bin index
    alert_1_df.sort_values (by='bin_index', ascending = True)
    grouped_alert_1_df = alert_1_df.groupby('unit_id')['spike_count'].apply(list).reset_index() #group the spike counts tgt, arranged by bin_index
    alert_1_correlation = correlations(grouped_alert_1_df) #get correlation of the new grouped df

    alert_2_df.sort_values (by='bin_index', ascending = True)
    grouped_alert_2_df = alert_2_df.groupby('unit_id')['spike_count'].apply(list).reset_index()
    alert_2_correlation = correlations(grouped_alert_2_df)

    alert_3_df.sort_values (by='bin_index', ascending = True)
    grouped_alert_3_df = alert_3_df.groupby('unit_id')['spike_count'].apply(list).reset_index()
    alert_3_correlation = correlations(grouped_alert_3_df)


    print(f"Alert 1 DataFrame: \n {grouped_alert_1_df} \n correlations 1: \n {alert_1_correlation}")
    print(f"Alert 2 DataFrame: \n {grouped_alert_2_df} \n correlations 2: \n {alert_2_correlation}")
    print(f"Alert 3 DataFrame: \n {grouped_alert_3_df} \n correlations 1: \n {alert_3_correlation}")



# split_by_behaviour(df, 5, df_behaviour)




def permean(groupedfile):
    proportions = []
    for i in range (len(groupedfile)): #add mean of each row into a list called "meanspike"
        mean = np.mean(groupedfile.iloc[i][groupedfile.columns[1]]) #get the mean for spike numbers in each row
        divided = [float(x/mean) for x in groupedfile.iloc[i][groupedfile.columns[1]]]
        # divided = divided.astype(float)
        proportions.append(divided) #list of list of (spike numbers in each bin/mean spikes over all bins)
    data = {
        "unit_id": groupedfile[groupedfile.columns[0]].tolist(),
        "spike_number_proportions": proportions
    }
    permean_df = pd.DataFrame(data) #each number in the 2nd column = spike number in the bin/mean spike over all bins
    return permean_df
# permean_df = permean (binned_df)



# correlations(binned_df)
def get_correlations (og_file, binwidth): #get the correlation matrix from the original file
    grouped_df = group (og_file) #group the spiketimes
    binned_df = groupBin(grouped_df,binwidth) #bin the spiketimes per bin width
    # print  ("correlation matrix for bin width:", binwidth, "seconds: \n")
    return correlations(binned_df) #return correlation matrix
correlation_matrix = get_correlations (df_spiketime, 2)
# print ("correlations:\n", get_correlations (df, 5))

def return_STRP_value (neuron_id, df, binwidth): #return correlation value of one neuron (neuron_id) against others
    STPR_df = group_STPR(df, neuron_id)  # change the OG files so that neuron ID are the chosen and -1
    correlation_matrix_df = get_correlations(STPR_df, binwidth)  # 2 by 2 matrix
    correlation = correlation_matrix_df.iloc[0, 1]
    return correlation
# print (return_STRP_value(0, df, 1))


def return_STRP_value_from_grouped_file (neuron_ID, df_neurons, binwidth):
    # getting correlations between a neuron and the same brain area
    df_neurons = df_neurons.copy()
    df_neurons.loc[df_neurons["unit_id"] != neuron_ID, "unit_id"] = -1 # changing all the non-chosen neuron ids to -1
    grouped_neuron_df = df_neurons.groupby('unit_id')['spikeTime'].apply(list).reset_index() #combining the "-1" neurons together
    grouped_neuron_df['spikeTime'] = grouped_neuron_df['spikeTime'].apply(lambda x: unlist(x)) #correct the column "spiketime" that is now a list of list --> into just list
    grouped_bin_neuron_df = groupBin(grouped_neuron_df, binwidth)
    correlation = correlations(grouped_bin_neuron_df).iloc[0, 1]
    return correlation


# df_STPR = pd.DataFrame({
#         'neurons': [],
#         'STPR': []
#     })
# for i in df_neurons_VCX["unit_id"]:
#     df_STPR.loc[len(df_STPR)] = [i, return_STRP_value_from_grouped_file(i,df_neurons_VCX, 0.1)]
# print (df_STPR)
# plot_STPR(df_STPR, 'neurons')



def get_all_STRP_values(df, binwidth):
    for i in group(df)['unit_id']: #i = neuron_id
        correlation = return_STRP_value (i, df, binwidth)
        print (f"correlation for neuron {i}: {correlation}")
#get_all_STRP_values (df, 1)
#
# get_all_STRP_values(df, 5)



def show_cluster(og_file):
    correlation_matrix = get_correlations (og_file, 5)
    # Step 2: Preprocessing
    # Optional: If the matrix is symmetrical, convert it into a condensed form (e.g., using upper triangle)
    # Here, we directly use the matrix as is for clustering.
    data = correlation_matrix.values  # Convert to NumPy array

    # Step 3: Perform K-means clustering
    # Choose the number of clusters (e.g., k=3)
    k = 9  # You can modify this based on your data
    kmeans = KMeans(n_clusters=5, random_state=42)
    row_clusters = kmeans.fit_predict(data)  # Cluster the rows (neurons)

    # Step 4: Sort rows and columns by clusters to preserve symmetry
    sorted_indices = np.argsort(row_clusters)
    sorted_matrix = correlation_matrix.iloc[sorted_indices, sorted_indices]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, cmap="coolwarm", xticklabels=False, yticklabels=False, square = True)
    plt.title("Heatmap of Neuron Correlation Matrix (Clustered)")
    plt.show()

    # Optional: Add cluster labels to original DataFrame for reference
    correlation_matrix['Cluster'] = cluster

    # Print cluster assignments
    print(correlation_matrix[['Cluster']])
# show_cluster()


def show_cluster_by_brain_area(correlation_matrix, df_spike_labels):

    # Merge the correlation matrix with the brain area information
    correlation_matrix_with_area = correlation_matrix.join(df_spike_labels.set_index('unit_id'), on='unit_id')

    #Sort by brain area
        #sort the rows according to brain area
    sorted_correlation_matrix = correlation_matrix_with_area.sort_values(by='UnitArea', axis=0)

        #rearrange the columns to match the rows
    sorted_correlation_matrix = sorted_correlation_matrix[sorted_correlation_matrix.index]

    # Step 3: Plot the heatmap with clustered brain areas
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_correlation_matrix.iloc[:, :-1].iloc[:-1, :], cmap="coolwarm", xticklabels=False, yticklabels=False, square=True)
    plt.title("Heatmap of Neuron Correlation Matrix (Clustered by Brain Area)")
    plt.show()

    # Optional: Print the sorted matrix or cluster labels
    print(sorted_correlation_matrix[['UnitArea']])

# show_cluster_by_brain_area(correlation_matrix, df_spikelabels)









