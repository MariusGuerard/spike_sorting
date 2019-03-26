"""
Module that gather the different methods of feature extractions:

For features extraction:
- pca 
- wavelets + pca
- wavelets + multimodal selection
(- VAE)
(- t-SNE)

For Clustering:
- k-means
(- GMM)
(- DBSCAN)
(- SPC)



Author: Marius Guerard
"""
import numpy as np
import pywt

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


from scipy import stats
from scipy.optimize import curve_fit
from scipy import exp


########
# MATH #
########

def fit_gaussian(signal, bins=100):
    """Compute a distribution of the signal and fit a gaussian to this 
    distribution. Then compute the inverse of the normalized distance between
    the distribution and the fitted curve. This give an indicator of normality
    as the biggest is this value, the smallest is the distance between the
    distribution and its fitted gaussian.

    Args:
        signal (ndarray): Signal to evaluate its normality.
        bins (int): number of bins used to generate the distribution (default 100)

    Returns:
	(tuple of float): parameters of the fitted gaussian.
        (float): indicator of normality of the signal.

    """
    signal_distrib = np.histogram(signal, bins=bins)
    x = signal_distrib[1][:-1]
    y = signal_distrib[0]

    signal_mean = signal.mean()
    signal_std = signal.std()

    def gauss(x, a, x0, sigma):
        return a * exp(-(x - x0)**2 / (2 * sigma**2))

    popt, pcov = curve_fit(gauss, x, y, p0=[1, signal_mean, signal_std])

    fit_y = gauss(x, *popt)

    err2 = ((y - fit_y)**2/y.mean()).sum()

    # We want small value = not normal to match p_value test.
    norm_indicator = 1 / (err2 + 1e-15)
        
    return popt, norm_indicator



def normal_rank(signals, norm_test=fit_gaussian):
    """Take the last n normal signals to reduce dimensionality.
    (if not normal, probably multimodal, so probably containing multiple spikes)
    In this specific application, put the wavelets coefficients into signals...

    Note: what can you use on norm_test:
        For Shapiro: stats.shapiro
        For Dagostino: stats.normaltest
        For fitting a gaussina method on the distrib: fit_gaussian
        Also Try anderson-darling, lilifoers, dip test, KDE estimation.

    Args:
        signals (ndarray): coefficients that one wants to rank their normality.
        norm_test (function): method to evaluate normality (default fit_gaussian).

    Returns:
	(list of float): normality indicator for each distribution.
        (list of int): distribution index ranked from least normal to most normal.

    """
    pval_list = [norm_test(signals[:, i])[1]
                 for i in range(signals.shape[1])]
    
    return pval_list, np.argsort(pval_list)



############################
# DIMENSIONALITY REDUCTION #
############################


def pca(signals, low_dim=12):
    """Perform a PCA for extracting the features from the spikes.

    Args:
        signals (list of ndarray): contains all the wave_forms to be clustered.
        low_dim (int): Number of dimension of the PCA. (default 12)

    Returns:
	(list of ndarray): the signal projected on the 'low_dim' first components.
    """

    ### Apply min-max scaling.
    scaler = sk.preprocessing.MinMaxScaler()
    signal_norm = scaler.fit_transform(signals)
    ### Dimensionality Reduction.    
    pca = PCA(n_components=low_dim)
    signal_pca = pca.fit_transform(signals)
    #hf.plot_pca(wave_pca)
    return signal_pca


def dwt_pca(signals, wavelets_level=4, low_dim=12):
    """Perform a Discrete Wavelets Transform (DWT) and then a PCA 
    on the coefficients of the DWT.

    Args:
        signals (list of ndarray): contains all the wave_forms to be clustered.
        wavelets_level (int): Order of decomposition of the DWT. (default 4)
        low_dim (int): Number of dimension of the PCA. (default 12)

    Returns:
	(list of ndarray): the wavelets coefficients projected on the 
        'low_dim' first components.
    """
    coeffs = pywt.wavedec(signals, 'haar', level=wavelets_level)
    coeffs_concat = np.concatenate(coeffs, axis=1)
    return pca(coeffs_concat, low_dim=low_dim)



def dwt_multimodal(signals, wavelets_level=4, low_dim=12):
    """Perform a Discrete Wavelets Transform (DWT) and then choose the 
    coefficients with the 'least normal' distribution in order to select
    potential multimodal distribution candidates (see Quiroga 2004).

    Args:
        signals (list of ndarray): contains all the wave_forms to be clustered.
        wavelets_level (int): Order of decomposition of the DWT. (default 4)
        low_dim (int): Number of dimension to keep. (default 12)

    Returns:
	(list of ndarray): the wavelets coefficients projected on the 
        'low_dim' first components.
    """
    coeffs = pywt.wavedec(signals, 'haar', level=wavelets_level)
    coeffs_concat = np.concatenate(coeffs, axis=1)
    idx_least_normal = normal_rank(coeffs_concat)[1][:low_dim]
    return coeffs_concat[:, idx_least_normal]

##############
# CLUSTERING #
##############

def k_means(signal_low_dim, n_clus=3):
    """Clusterize the low dimension signal thanks to K-means algorithm.

    Args:
        signal_low_dim (list of ndarray): contains the signal to be clustered.
        n_clus (int): Number of clusters that gather the spikes. (default 3)

    Returns:
	(dic): contains the low-dimension signal and the clusters of all spikes.
    """
    ### K-means clustering.
    kmeans = KMeans(n_clusters=n_clus, random_state=0)
    #print(kmeans.cluster_centers_)
    signal_labels = kmeans.fit_predict(signal_low_dim)
    cluster_centers = kmeans.cluster_centers_
    
    ### Organize results.
    result = {}
    result['model'] = kmeans
    result['n_dim'] = signal_low_dim.shape[1]
    result['n_clus'] = n_clus
    result['low_dim_signal'] = signal_low_dim
    result['labels'] = signal_labels
    result['cluster_centers'] = cluster_centers
    
    return result





def dbscan(signal_low_dim, eps=2., min_sample=300):
    """Clusterize the low dimension signal thanks to DBSCAN algorithm.

    Args:
        signal_low_dim (list of ndarray): contains the signal to be clustered.

    Note: For this data eps=2, min_sample=500 works.

    Returns:
	(dic): contains the low-dimension signal and the clusters of all spikes.
    """
    ### DBSCAN clustering.
    signal_norm = StandardScaler().fit_transform(signal_low_dim)
    db = DBSCAN(eps=eps, min_samples=min_sample).fit(signal_norm)
    signal_labels = db.labels_
    
    ### Organize results.
    result = {}
    result['model'] = db
    result['n_dim'] = signal_low_dim.shape[1]
    result['n_clus'] = signal_labels.max() + 1
    result['low_dim_signal'] = signal_low_dim
    result['labels'] = signal_labels
    #result['cluster_centers'] = cluster_centers
    
    return result






def feat_clust(signals, feat_func=dwt_pca, clust_type=k_means, low_dim=10):
    low_dim_signal = feat_func(signals, low_dim=low_dim)
    return clust_type(low_dim_signal)

    

### Try all combinations of dimension reduction and clustering:
### dim. red. : PCA, wavelets, t-sne, VAE, NMF.
### clustering algo: k-means, SPC, DBSCAN

### Note: I think that even used in the cloud with computer power, these methods might be too long to be executed fast enough to be seamless. But it can still inform us on what relevant feature to look for during calibration and then just look for these relevant features.


#########
# DRAFT #
#########




# def spc(signal_low_dim):
#     """Clusterize the low dimension signal thanks to superparamagnetic (SPC)
#     algorithm.

#     Args:
#         signal_low_dim (list of ndarray): contains the signal to be clustered.

#     Returns:
# 	(dic): contains the low-dimension signal and the clusters of all spikes.
#     """
    
#     ### Organize results.
#     result = {}
#     result['model'] = 
#     result['n_dim'] = signal_low_dim.shape[1]
#     result['n_clus'] = n_clus
#     result['low_dim_signal'] = signal_low_dim
#     result['labels'] = signal_labels
#     result['cluster_centers'] = cluster_centers
    
#     return result

