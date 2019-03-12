"""
Pipeline for Spike sorting algorithms.
Author: Marius Guerard
"""

import helper_function as hf

##############
# PARAMETERS #
##############

# Data path containing your Neuralynx data.
DATA_PATH = "../UCLA_data/CSC4.Ncs"

# Number of cluster (shoud be determined automatically by for example SPC).
N_CLUS = 3

# Window of time (in seconds) to plot some samples of spikes.
T_MIN_MAX = [0, 3]

########
# MAIN #
########

def spike_sort(data_path=DATA_PATH, n_clus=N_CLUS, t_min_max=T_MIN_MAX):
    """
    First draft of the pipeline use to cluster signal amplitude into the
    different spikes that initiated it.    
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
    features = hf.feature_extraction_0(wave_form, n_clus=n_clus)

    ### Plot the cluster map and their average wave form.
    hf.plot_features(wave_form, sf, features)

    ### Plot the raw signal and the filtered signal between time
    ### 'T_MIN_MAX[0]' and 'T_MIN_MAX[1]'.
    # hf.plot_amplitude(time, amp, sf, t_min=T_MIN, t_max=T_MAX)
    # hf.plot_amplitude(time, filtered_amp, sf, t_min=T_MIN, t_max=T_MAX)
    
    ### Plot the spikes maximum positions.
    # plt.vlines(time[pos], -200, 100)
    ### Plot a random subset of the spikes wave_form
    # hf.plot_random_spikes(wave_form)
 
    
    return raw, pos, wave_form, features

if __name__ == "__main__":
    raw, pos, wave_form, features_0 = spike_sort()
