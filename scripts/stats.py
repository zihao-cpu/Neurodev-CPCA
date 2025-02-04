import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t

def discriminability_score(patterns1, patterns2, sub_name, comp):
    n_sub = len(sub_name)
    args = dict(data = np.vstack((patterns1[:, comp].reshape(n_sub, -1).real, 
                                patterns2[:, comp].reshape(n_sub, -1))).real,
                labels = np.tile(sub_name, 2))
    return calculate_discriminability(**args)

def calculate_discriminability(data, labels):
    """
    Calculate the discriminability of the given dataset.

    Parameters:
    data (numpy.ndarray): A 2D array where each row represents a sample.
    labels (numpy.ndarray): A 1D array containing the label for each sample. Replicates should have the same label.

    Returns:
    float: The discriminability score.
    """

    # Step 1: Compute the pairwise distances
    dist_matrix = squareform(pdist(data))

    # Step 2: Identify replicated measurements
    n = len(labels)
    unique_labels = np.unique(labels)
    num_replicates = len(unique_labels)

    f = 0
    g = 0

    # Step 3: Compare within-item and across-item distances
    for i in range(n):
        within_item_distances = []
        across_item_distances = []
        for j in range(n):
            if labels[i] == labels[j] and i != j:
                within_item_distances.append(dist_matrix[i, j])
                g += 1
            elif labels[i] != labels[j]:
                across_item_distances.append(dist_matrix[i, j])
        
        # Compare distances
        for d in across_item_distances:
            if within_item_distances and d < min(within_item_distances):
                f += 1
    
    # Step 4: Calculate discriminability
    discriminability = 1 - (f / (n * (n - 1) - g))

    return discriminability




def circorr(A, B):
    n_t = len(A)
    return np.max([pearsonr(A.ravel(), np.roll(B, shift=t, axis=0).ravel())[0] for t in range(n_t)])

def EVR(S, n, axis=None):
    ev = S ** 2 / (n - 1)
    return ev / ev.sum(axis=axis, keepdims=True)


# Calculate the standard error
def get_ci(data):
    sample_std = np.std(data, ddof=1)
    sample_size = len(data)
    std_error = sample_std / np.sqrt(sample_size)

    # Calculate the critical value from the t-distribution
    confidence_level = 0.95
    degrees_of_freedom = sample_size - 1
    critical_value = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the confidence interval
    ci = critical_value * std_error
    # ci_low = sample_mean - critical_value * std_error
    # ci_high = sample_mean + critical_value * std_error
    return ci