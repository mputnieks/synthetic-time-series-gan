import pandas as pd
from scipy.spatial.distance import cosine
import itertools
import numpy as np
import os


def compute_pair_dissimilarity(data):
    """compute all possible data pair cosine dissimilarities and return the average, in %"""
    data = data.values  # convert to numpy array
    cosine_dissimilarities = []

    # Compute cosine similarities for all possible pairs
    for pair in itertools.combinations(range(data.shape[0]), 2):
        sample1 = data[pair[0]]
        sample2 = data[pair[1]]
        dissimilarity = cosine(sample1, sample2) * 100
        if dissimilarity < 0.1:
            print("\nPair: ", pair)
            print("Exceptionally small dissimilarity: " + str(round(dissimilarity,4)) + "%")
            print("\n")
        cosine_dissimilarities.append(dissimilarity)

    # Compute the average cosine similarity
    average_dissimilarity = sum(cosine_dissimilarities) / len(cosine_dissimilarities)
    
    return round(average_dissimilarity,4)


def real_to_synthetic_similarity(r_data, s_data, n_real_segments):
    """compute the average cosine similarity between each generated segment and n random real segments, in %"""

    r_data = r_data.values      # convert to numpy array
    s_data = s_data.values      # convert to numpy array
    avg_dissimilarities = []
    
    # Compute cosine similarities
    min_dissimilarity = None
    min_dissimilarity_index = None
    for i in range(s_data.shape[0]):
        cosine_dissimilarities = []
        random_indices = np.random.choice(r_data.shape[0], n_real_segments, replace=False)
        
        sample1 = s_data[i]
        
        for random_index in random_indices:
            sample2 = r_data[random_index]
            # Compute cosine dissimilarity and add to the list
            dissimilarity = cosine(sample1, sample2) * 100
            
            # Look if the dissimilarity is the smallest
            if min_dissimilarity is None or dissimilarity < min_dissimilarity:
                min_dissimilarity = dissimilarity
                min_dissimilarity_index = i, random_index
            
            cosine_dissimilarities.append(dissimilarity)
        
        # Compute the average cosine dissimilarity for the current segment
        avg_dissimilarity = sum(cosine_dissimilarities) / len(cosine_dissimilarities)
        avg_dissimilarities.append(avg_dissimilarity)

    print("Min RTS dissimilarity: " + str(min_dissimilarity) + "%")
    print("(index RD, index SD) : " + str(min_dissimilarity_index))
    
    # Return the mean average cosine dissimilarity across all segments, rounded to 4 decimals
    return round(sum(avg_dissimilarities) / len(avg_dissimilarities),4)


def get_roughness(data):
    """Evaluates the average smoothness of the time series data by 
    calculating the variance or standard deviation of the differences between consecutive values.
    A perfectly smooth straight line would return 0"""
    data = data.values  # convert to numpy array
    smoothness_scores = []
    for i in range(s_data.shape[0]):
        differences = np.diff(data[i])
        smoothness_score = np.std(differences)
        smoothness_scores.append(smoothness_score)
    # Calculate the average smoothness score
    return round(np.mean(smoothness_scores) ,4)


if __name__ == '__main__':
    # load the data
    os.chdir("src\data")                                        # Set working directory
    real_data = pd.read_csv("compiled_full_weekday.csv").iloc[:, :48]
    s_data = pd.read_csv("sd_11000_156_10_10_11000_decoded.csv").iloc[:, :48]
    
    # print("RTR dissimilarity: " + str(compute_pair_dissimilarity(real_data)) + "%")
    print("RTR dissimilarity: 6.9031%")
    print("STS dissimilarity: " + str(compute_pair_dissimilarity(s_data)) + "%")
    print("RTS dissimilarity: " + str(real_to_synthetic_similarity(real_data, s_data, 10)) + "%")
    
    print("Real CGM roughness: " + str(get_roughness(real_data.iloc[:, :48])))
    print("Synthetic CGM roughness: " + str(get_roughness(s_data.iloc[:, :48])))