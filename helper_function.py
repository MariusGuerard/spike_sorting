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
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def load_ncs(data_path):
    """
    Load data from Neuralynx (by Carsten Klein).
    Note: The ravel of the data['Samples'] might cause problems because 
    consider that there is no pause between recordings...
    512/sf * 1e6 = 15974 != raw[1][0] - raw[0][0] = 15872 (100 microseconds 
    of difference is not much though so it might be negligible).
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
    """
    Plot voltage amplitude between t_min seconds and t_max seconds.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(time_vec[t_min*sf:t_max*sf], data_vec[t_min*sf:t_max*sf])
    ax.set_title('Broadband; Sampling Frequency: {}Hz'.format(sf), fontsize=23)
    ax.set_xlim(time_vec[t_min*sf], time_vec[t_max*sf])
    ax.set_xlabel('time [s]', fontsize=20)
    ax.set_ylabel('amplitude [uV]', fontsize=20)
    #plt.show()

def plot_random_spikes(spike_data, n_spikes=100):
    """
    Plot 'n_spikes' random spikes among 'spike_data'
    """
    np.random.seed(10)
    fig, ax = plt.subplots(figsize=(15, 5))
    
    for i in range(100):
        spike = np.random.randint(0, len(spike_data))
        ax.plot(spike_data[spike, :])

    #ax.set_xlim([0, 90])
    ax.set_xlabel('# sample', fontsize=20)
    ax.set_ylabel('amplitude [uV]', fontsize=20)
    ax.set_title('spike waveforms', fontsize=23)
    plt.show()


def plot_pca(signal_pca, c_data=None):
    """

    """
    try: c_data.shape
    except: c_data = signal_pca[:,2] 
    # Plot the 1st principal component aginst the 2nd and use the 3rd for color
    fig, ax = plt.subplots(figsize=(8, 8))
    # ax.scatter(signal_pca[:, 0], signal_pca[:, 1], c=signal_pca[:, 2])
    ax.scatter(signal_pca[:, 0], signal_pca[:, 1], c=c_data)
    ax.set_xlabel('1st principal component', fontsize=20)
    ax.set_ylabel('2nd principal component', fontsize=20)
    # ax.set_title('first 3 principal components', fontsize=23)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    #plt.legend()
    plt.show()

#############
# FILTERING #
#############

def pass_band_butter(signal, sf, low_hz=500, high_hz=9000, order=2):
    # Nyquist frequency.
    nyq = sf / 2

    # Frequency in term of Nyquist multipliers.
    low = low_hz / nyq
    high = high_hz / nyq

    # Calculate coefficients.
    b, a = butter (order, [low, high], btype='band')

    # Filtered signal.
    filtered_signal = lfilter(b, a, signal)

    return filtered_signal


### XXX Look other filters, parameters.

####################
# SPIKE EXTRACTION #
####################

def extract_spikes(data, spike_window=80, tf=5, offset=10, max_thresh=350,
                   spike_mode='median'):
    """
    Extract spike waveforms from the data and align them together.
    It is probably better to compute threshold with median, but to be verified.
    """

    if spike_mode == 'median':
        # Threshold based on median (see Quiroga 2004).
        thresh = np.median(np.abs(data)/0.6745) * tf
    else:
        # Calculate threshold based on data mean.
        thresh = np.mean(np.abs(data)) * tf

    # Find positions wherere the threshold is crossed.
    pos_vec = np.where(data > thresh)[0]
    # Test of which of these position are not too close from start or end.
    pos_in_simu = (pos_vec > spike_window) * (pos_vec < len(data) - spike_window) 
    # Remove the position that are too close from start or end.
    pos_vec = pos_vec[pos_in_simu]
    
    # Store the position of the maximum of the spike.
    spike_pos = []
    # Store the spike signals centered on the window.
    wave_form_list = [] #np.zeros(spike_window * 2)

    for pos in pos_vec:
        # signal in the window around where the threshold is crossed.
        signal_window = data[pos - spike_window:pos + spike_window]
        # Check if data in window is below upper threshold (artifact filtering).
        if signal_window.max() < max_thresh:
            # Find sample with maximum data point in window.
            pos_max = np.argmax(signal_window) + pos
            # Re-center Window on maximum sample.
            wave_form = data[pos_max - 2*spike_window : pos_max + spike_window]

            # Append data.
            spike_pos.append(pos_max)
            wave_form_list.append(wave_form)

    # Remove duplicates.
    ind_unique = np.where(np.diff(spike_pos) > 0)[0]
    spike_pos = np.array(spike_pos)[ind_unique]
    wave_form_list = np.array(wave_form_list)[ind_unique]
    return spike_pos, wave_form_list



######################
# FEATURE EXTRACTION #
######################

### With PCA and k-means clustering.

def feature_extraction_0(signal, pca_dim=12, n_clus=3):
    """

    """
    # Apply min-max scaling.
    # (XXX Should try with standard and robust scaling as well).    
    scaler = sk.preprocessing.MinMaxScaler()
    signal_norm = scaler.fit_transform(signal)
    # Dimensionality Reduction.    
    pca = PCA(n_components=pca_dim)
    signal_pca = pca.fit_transform(signal_norm)
    #hf.plot_pca(wave_pca)

    ### K-means clustering.
    kmeans = KMeans(n_clusters=n_clus, random_state=0)
    #print(kmeans.cluster_centers_)
    signal_labels = kmeans.fit_predict(signal_pca)
    cluster_centers = kmeans.cluster_centers_
    
    ### Organize results.
    result = {}
    result['n_dim_pca'] = pca_dim
    result['n_clus'] = n_clus
    result['pca'] = signal_pca
    result['labels'] = signal_labels
    result['cluster_centers'] = cluster_centers
    
    return result

def plot_features(signal, sample_freq, result_extraction):
    """

    """
    n_clus = result_extraction['n_clus']
    signal_pca = result_extraction['pca']
    labels = result_extraction['labels']
    cluster_centers = result_extraction['cluster_centers']

    
    cluster_mean = [signal[labels==i, :].mean(axis=0) for i in range(n_clus)]
    cluster_std = [signal[labels==0, :].std(axis=0) for i in range(n_clus)]

    # Plot the result
    plot_pca(signal_pca, c_data=labels)

    time = np.linspace(0, signal.shape[1]/sample_freq, signal.shape[1])*1000

    fig, ax = plt.subplots(figsize=(15, 5))
    for i in range(n_clus):
        clus_mean = cluster_mean[i]
        clus_std = cluster_std[i]
        ax.plot(time, clus_mean, label='Cluster {}'.format(i))
        ax.fill_between(time, clus_mean-clus_std, clus_mean+clus_std,
                        alpha=0.15)

    plt.legend()

    
### With Wavelets and magnetic clustering XXX.

### Try all combinations of dimension reduction and clustering:
### dim. red. : PCA, wavelets, t-sne, VAE, NMF.
### clustering algo: k-means, SPC, DBSCAN


