"""
Helper functions for spike sorting algorithms.
Author: Marius Guerard

Inspirations: 

https://github.com/akcarsten/akcarsten.github.io/blob/master/spike_sorting/Spike_sorting%20.ipynb

https://vis.caltech.edu/~rodri/papers/Spike_sorting.pdf

XXX Check energy-efficient algorithms for comparison.
"""
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
plt.ion()


########
# MATH #
########

# [doc]
def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    # should be faster to not use masked arrays.
    arr = np.ma.array(arr).compressed() 
    med = np.median(arr)
    return np.median(np.abs(arr - med))




#######################
# LOADING AND PARSING #
#######################

def load_ncs(data_path):
    """Load data from Neuralynx (by Carsten Klein).

    Note: The ravel of the data['Samples'] might cause problems because 
    consider that there is no pause between recordings...
    512/sf * 1e6 = 15974 != raw[1][0] - raw[0][0] = 15872 (100 microseconds 
    of difference is not much though so it might be negligible).

    Args:
        data_path (str): location of the ncs file.

    Returns:
        ndarray: The raw data
        uint32: the sampling frequency
        ndarray: the signal amplitude
        ndarray: the time vector
    
     """
    # Header has 16 kilobytes length.
    HEADER_SIZE = 16 * 1024
    # Open file.
    fid = open(data_path, 'rb')
    # Skip header by shifting position by header size.
    fid.seek(HEADER_SIZE)
    # Read data according to Neuralynx information
    data_format = np.dtype([('TimeStamp', np.uint64),
                            ('ChannelNumber', np.uint32),
                            ('SampleFreq', np.uint32),
                            ('NumValidSamples', np.uint32),
                            ('Samples', np.int16, 512)])
    raw = np.fromfile(fid, dtype=data_format)
    # Close file
    fid.close()
    # Get sampling frequency
    sf = raw['SampleFreq'][0]
    # Create data vector
    data = raw['Samples'].ravel()

    # Determine duration of recording in seconds
    dur_sec = len(data) / sf

    # Create time vector
    time_vec = np.linspace(0, dur_sec, len(data))

    return raw, sf, data, time_vec


###########################
# VISUALIZATION FUNCTIONS #
###########################
     
def plot_amplitude(time_vec, data_vec, sf, t_min=0, t_max=1):
    """Plot voltage amplitude between t_min seconds and t_max seconds.

    Args:
        time_vec (ndarray): time vector
        data_vec (ndarray): data to plot.
        sf (uint32): sampling frequency
        t_min (uint): start of x axis (default 0)
        t_max (uint): end of x axis (default 1)

    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_vec[t_min*sf:t_max*sf], data_vec[t_min*sf:t_max*sf])
    ax.set_title('Broadband; Sampling Frequency: {}Hz'.format(sf), fontsize=23)
    ax.set_xlim(time_vec[t_min*sf], time_vec[t_max*sf])
    ax.set_xlabel('time [s]', fontsize=20)
    ax.set_ylabel('amplitude [uV]', fontsize=20)


def plot_random_spikes(spike_data, n_spikes=100):
    """Plot 'n_spikes' random spikes among 'spike_data'.

    Args:
        spike_data (ndarray): each element contains a spike wave form
        n_spikes (uint32): number of random spikes to be plotted. (default 100)

    """
    np.random.seed(10)
    fig, ax = plt.subplots(figsize=(15, 5))
    
    for i in range(100):
        spike = np.random.randint(0, len(spike_data))
        ax.plot(spike_data[spike, :])

    ax.set_xlabel('# sample', fontsize=20)
    ax.set_ylabel('amplitude [uV]', fontsize=20)
    ax.set_title('spike waveforms', fontsize=23)
    plt.show()


def plot_pca(signal_pca, c_data=None):
    """Scatter plot the two first columns of signal_pca

    Args:
        signal_pca (ndarray): the pca to be plotted
        c_data (ndarray): the color value, 
        if None, use the 3rd columns of signal_pca (default None)
    """

    # If no color data are specified use the 3rd columns of signal_pca
    # (which represent the 3rd principal component).
    try: c_data.shape
    except: c_data = signal_pca[:, 2]
    
    # Plot the 1st principal component aginst the 2nd.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(signal_pca[:, 0], signal_pca[:, 1], c=c_data)
    ax.set_xlabel('1st principal component', fontsize=20)
    ax.set_ylabel('2nd principal component', fontsize=20)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()



def plot_features_cluster(signals, sample_freq, result_cluster):
    """Plot the clusters on the two first dimensions of the low_dimensional
    signal, also plot the average and std of the wave_forms of all clusters.

    Args:
        signals (list of ndarray): each arra
        sample_freq (float): sampling_frequency
        result_cluster (dic): result from a clusterization function

    """
    ### Extract variables from clusterization result.
    n_clus = result_cluster['n_clus']
    low_dim_signal = result_cluster['low_dim_signal']
    labels = result_cluster['labels']
    #cluster_centers = result_cluster['cluster_centers']

    ### Plot the clusters on the two first dimension of the low dim signal.
    plot_pca(low_dim_signal, c_data=labels)

    ### Plot the average wave_form (and std) for each cluster.
    cluster_mean = [signals[labels == i, :].mean(axis=0) for i in range(n_clus)]
    cluster_std = [signals[labels == 0, :].std(axis=0) for i in range(n_clus)]

    time = np.linspace(0, signals.shape[1]/sample_freq, signals.shape[1])*1000

    fig, ax = plt.subplots(figsize=(15, 5))
    for i in range(n_clus):
        clus_mean = cluster_mean[i]
        clus_std = cluster_std[i]
        ax.plot(time, clus_mean, label='Cluster {}'.format(i))
        ax.fill_between(time, clus_mean-clus_std, clus_mean+clus_std,
                        alpha=0.15)

    plt.legend()

    
#############
# FILTERING #
#############


def pass_band_butter(signal, sf, low_hz=500, high_hz=9000, order=2):
    """Implement a butter pass_band.

    Args:
    signal (ndarray): the signal to be filtered.
    sf (uint32): the sampling frequency.
    low_hz (the lower bound of the filter in Hz (default 500)
    high_hz (float): the higher bound of the filter in Hz (default 9000)
    order (uint): order of the filter. (default 2)

    Returns:
	ndarray: the filtered signal

    """
    # Nyquist frequency
    # (highest frequency of the original signal that is not aliased).
    nyq = sf /2

    # Frequency in term of Nyquist multipliers.
    low = low_hz / nyq
    high = high_hz / nyq

    # Calculate coefficients.
    b, a = butter(order, [low, high], btype='band')

    # Filtered signal.
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


### XXX Look other filters, parameters.

####################
# SPIKE EXTRACTION #
####################


def extract_spikes(signal, spike_window=80, thresh_coeff=5, offset=10,
                   max_thresh=350, spike_mode="quiroga"):
    """Extract spike waveforms from the data and align them together.
    It is probably better to compute threshold with median, but to be verified.

    Args:
        signal (ndarray): signal containing spikes mixed with noise.
        spike_window (int): number of acquisition that define a wave_form 
        (default 80)
        thresh_coeff (float): ratio between threshold and noise (default 5)
        offset (int): offset between the spike's maximum and the window center 
        (default 10)
        max_thresh (float): high-threshold to remove artifacts (default 350)
        spike_mode way to compute the threshold from the signal (default "median")

    Returns:
	list: positions of the spike's maximums
        list of ndarray: each array contains a spikes' wave form.
    """
    if spike_mode == 'quiroga':
        # Threshold based on median (see Quiroga 2004).
        thresh = np.median(np.abs(signal)/0.6745) * thresh_coeff
    elif spike_mode == 'mad':
        # Thresh based on median absolute deviation.
        thresh = mad(signal) * thresh_coeff
    else:
        # Calculate threshold based on mean.
        thresh = np.mean(np.abs(signal)) * thresh_coeff

    # Find positions wherere the threshold is crossed.
    pos_vec = np.where(signal > thresh)[0]
    # Test of which of these position are not too close from start or end.
    pos_in_simu = (pos_vec > spike_window) * \
                  (pos_vec < len(signal) - spike_window) 
    # Remove the position that are too close from start or end.
    pos_vec = pos_vec[pos_in_simu]
    
    # Store the position of the maximum of the spike.
    spike_pos = []
    # Store the spike signals centered on the window.
    wave_form_list = [] 

    for pos in pos_vec:
        # signal in the window around where the threshold is crossed.
        signal_window = signal[pos - spike_window:pos + spike_window]
        # Check if signal in window is below upper threshold (artifact filtering).
        if signal_window.max() < max_thresh:
            # Find sample with maximum data point in window.
            pos_max = np.argmax(signal_window) + pos
            # Re-center Window on maximum sample.
            wave_form = signal[pos_max - 2*spike_window : pos_max + spike_window]
            # Append data.
            spike_pos.append(pos_max)
            wave_form_list.append(wave_form)

    # Remove duplicates.
    ind_unique = np.where(np.diff(spike_pos) > 0)[0]
    spike_pos = np.array(spike_pos)[ind_unique]
    wave_form_list = np.array(wave_form_list)[ind_unique]
    return spike_pos, wave_form_list



