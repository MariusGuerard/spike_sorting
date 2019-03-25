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



