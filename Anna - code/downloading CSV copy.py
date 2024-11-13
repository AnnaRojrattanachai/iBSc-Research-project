import pandas as pd
import numpy as np
import matplotlib as plt
import itertools


# with open ("M24028_20240426_File1_Shank0_goodSpikeTimes.csv", "r") as f:
#    for line in f:
#        print (line)

file1 = "/Users/pitsinee/PycharmProjects/HelloWorld/research project/M24028_20240426_File1_Shank0_goodSpikeTimes.csv"

def openfile (file_path):
    global df
    df = pd.read_csv(file_path, names = ["unit_id", "spikeTime"])
    print (df)
openfile(file1)

def group(file):
    global grouped_df
    grouped_df = file.groupby('unit_id')['spikeTime'].apply(list).reset_index()
    grouped_df.to_csv('rearranged_output.csv', index=True)
    global maxtime
    maxtime = max(df[df.columns[1]].tolist())
    print(grouped_df)
    print ("max time:", maxtime)

group(df)

def groupBin(binwidth):
    bins = np.arange(0, maxtime, binwidth) #returns array of bin edges
    allcount = [0] *(grouped_df.shape[0]) #list of lists of counts in each bin
    for i in range(grouped_df.shape[0]):
        x = grouped_df.iloc[i]['spikeTime'] #get the list of values of spike time for row i
        # print ("x:", x) #check for spike times
        counts, bin_edges = np.histogram(x, bins = bins)
        allcount[i] = counts
        # print ("all count:", len(allcount), "column[0]:", grouped_df.shape[0]) #check numbers of items if the same
        #print("Counts per bin for row", i, ":", counts)
    data = {
        "unit_id": grouped_df[grouped_df.columns[0]].tolist(),
        "spike number in each bin": allcount
    }
    global binned_df
    binned_df = pd.DataFrame(data)
    binned_df.to_csv ("binnedData.csv", index = False)
    print ("binned_df: \n", binned_df)
groupBin(10)

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
    global permean_df
    permean_df = pd.DataFrame(data) #each number in the 2nd column = spike number in the bin/mean spike over all bins
    print (permean_df)
permean (binned_df)

def correlations (file): #get the correlations of each combination in the file
    correlations = np.corrcoef(file[file.columns[1]].tolist()) #returns a matrix of correlations
    global correlation_df
    correlation_df = pd.DataFrame (correlations, index = file['unit_id'], columns = file['unit_id']) #give index as the unit_id
    print ("correlation matrix: \n", correlation_df)
    correlation_df.to_csv("correlation_matrix", index = True)
    return (correlation_df)
            # pairs = list(itertools.combinations(file[file.columns[0]].tolist(), 2))
            # print("pairs:", pairs)
            # for i in range(1): #(len(pairs)
            #     list1 = file.loc[file['unit_id'] == pairs[i][0], file.columns[1]].values.tolist()[0]
            #     list2 = file.loc[file['unit_id'] == pairs[i][1], file.columns[1]].values.tolist()[0]
            #     print (type(list1))
            #     print ("list1",list1)
            #     print ("list2", list2)
            #     print (f"correlations between the pair {pairs[i][0]} and {pairs[i][1]}:", pairedcorrelations)

correlations(binned_df)

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
    plt.show()

visualise_correlation_matrix (correlation_df, units = 10, font_size = 8)


    # for i in df[grouped_df.columns[0]].tolist():
    #     counts, bin_edges = np.histogram(df[df.columns[1]].tolist(), bins=bins)
    #     print("Counts per bin:", counts)  # Output: [2, 2, 3]
    #     print("Bin edges:", bin_edges)
    #
    # for i in range (len(bins)-1):
    #     for x in df[df.columns[1]].tolist():
    #         if bins[i] <= x < bins[i+1]:

    # print (bins)




#kamaines hierarichal clustering






