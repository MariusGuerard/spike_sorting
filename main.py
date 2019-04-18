"""
Pipeline for Spike sorting algorithms.

To do:
-Check Why the thresholding parameter 'quiroga' gives such different results.

Author: Marius Guerard
"""


import helper_function as hf
import feature_extraction as fe
import visualization as vis


##############
# PARAMETERS #
##############


# Data path containing your Neuralynx data.
DATA_PATH = "../UCLA_data/CSC4.Ncs"
#DATA_PATH = "../2018-05-31_15-43-39/CSC1.ncs"

# Number of cluster (shoud be determined automatically by for example SPC).
N_CLUS = 3

# Window of time (in seconds) to plot some samples of spikes.
T_MIN_MAX = (0, 3)


########
# MAIN #
########


def spike_sort(data_path=DATA_PATH, n_clus=N_CLUS, t_min_max=T_MIN_MAX):
    """First draft of the pipeline use to cluster signal amplitude into the
    different spikes that initiated it.

    Args:
        data_path (str): path of your ncs file.
        n_clus (int): number of cluster to be set manually for now.
        t_min_max (tuple of int): containsthe range of data to be plotted.
    """

    ### Load ncs data and return the raw data, the sampling frequency,
    ### the amplitude, and the time vector.
    raw, sf, amp, time_vec = hf.load_ncs(data_path)

    ### Filter the amplitude with pass_band butter.
    filtered_amp = hf.pass_band_butter(amp, sf)

    ### Plot the raw signal and the filtered signal between time
    ### 'T_MIN_MAX[0]' and 'T_MIN_MAX[1]'.
    # vis.plot_amplitude(time_vec, amp, sf, t_min=t_min_max[0], t_max=t_min_max[1])
    # vis.plot_amplitude(time_vec, filtered_amp, sf, t_min=t_min_max[0],
    #                   t_max=t_min_max[1])

    #import pdb; pdb.set_trace()
    ### Exctract spikes from filtered signal and return the position of the
    ### spikes maxima and all the wave form around these maxima.
    pos, wave_form = hf.extract_spikes(filtered_amp, thresh_mode="quiroga")

    ### Plot a random subset of the spikes wave_form
    #vis.plot_random_spikes(wave_form)

    ### Extract features from spikes.
    wave_features = fe.dwt_multimodal(wave_form, low_dim=10)

    ### Clusterize those features.
    features_1 = fe.k_means(wave_features, n_clus=n_clus)

    ### Plot the cluster map and their average wave form.
    vis.plot_features_cluster(wave_form, sf, features_1, dim=3)

    return raw, pos, wave_form, features_1


if __name__ == "__main__":

    raw, pos, wave_form, features_1 = spike_sort()


    ### DRAFT
    # import matplotlib.pyplot as plt
    # import pywt
    # import numpy as np
    # import sklearn as sk
    # from sklearn.cluster import DBSCAN
    # from sklearn.preprocessing import StandardScaler
    # from scipy import stats


    # coeff_reduced = fe.dwt_multimodal(wave_form)


    # dbscan_list = []
    # eps_range = np.linspace(1.6, 1.9, 10)
    # for eps_i in eps_range:
    #     dbscan_list.append(fe.dbscan(coeff_reduced, eps=eps_i))
