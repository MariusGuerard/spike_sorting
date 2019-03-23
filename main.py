"""
Pipeline for Spike sorting algorithms.
Author: Marius Guerard
"""

### For debuging.
import matplotlib.pyplot as plt
import pywt
import numpy as np
import sklearn as sk
from scipy import stats

import helper_function as hf
import feature_extraction as fe

##############
# PARAMETERS #
##############

# Data path containing your Neuralynx data.
DATA_PATH = "../UCLA_data/CSC4.Ncs"

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

    ### Exctract spikes from filtered signal and return the position of the
    ### spikes maxima and all the wave form around these maxima.
    pos, wave_form = hf.extract_spikes(filtered_amp)


    ### Extract features from spikes and cluster them.
    features_1 = fe.feat_clust(wave_form, feat_func=fe.dwt_multimodal)

    ### Plot the cluster map and their average wave form.
    hf.plot_features_cluster(wave_form, sf, features_1)

    ### Plot the raw signal and the filtered signal between time
    ### 'T_MIN_MAX[0]' and 'T_MIN_MAX[1]'.
    # hf.plot_amplitude(time_vec, amp, sf, t_min=t_min_max[0], t_max=t_min_max[1])
    # hf.plot_amplitude(time_vec, filtered_amp, sf, t_min=t_min_max[0],
    #                   t_max=t_min_max[1])
    
    ### Plot the spikes maximum positions.
    # plt.vlines(time[pos], -200, 100)
    ### Plot a random subset of the spikes wave_form
    # hf.plot_random_spikes(wave_form)
 
    
    return raw, pos, wave_form, features_1

if __name__ == "__main__":
    raw, pos, wave_form, features_1 = spike_sort()



    # coeffs_concat = np.concatenate(pywt.wavedec(wave_form, 'haar', level=4),
    #                                axis=1)

    # coeff_reduced = fe.dwt_multimodal(coeffs_concat)

    
