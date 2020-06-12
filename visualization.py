import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

    for i in range(n_spikes):
        spike = np.random.randint(0, len(spike_data))
        ax.plot(spike_data[spike, :])

    ax.set_xlabel('# sample', fontsize=20)
    ax.set_ylabel('amplitude [uV]', fontsize=20)
    ax.set_title('spike waveforms', fontsize=23)
    plt.show()


def plot_cluster(signal_pca, c_data=None, dim=2):
    """Scatter plot the two first columns of signal_pca

    Args:
        signal_pca (ndarray): the pca to be plotted
        c_data (ndarray): the color value,
        if None, use the 3rd columns of signal_pca (default None)
        dim (int): must be 2 for 2D plot or 3 for 3D plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # If no color data are specified use the 3rd columns of signal_pca
    # (which represent the 3rd principal component).
    try: c_data.shape
    except: c_data = signal_pca[:, dim]

    if dim == 2:
        # Plot the 1st principal component aginst the 2nd.
        ax.scatter(signal_pca[:, 0], signal_pca[:, 1], c=c_data)
    elif dim == 3:
        # Plot the 3 first Principal components (PC).
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(signal_pca[:, 0], signal_pca[:, 1], signal_pca[:, 2], c=c_data)

    ax.set_xlabel('1st PC', fontsize=20)
    ax.set_ylabel('2nd PC', fontsize=20)
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    

def plot_features_cluster(signals, sample_freq, result_cluster, dim=3):
    """Plot the clusters on the two first dimensions of the low_dimensional
    signal, also plot the average and std of the wave_forms of all clusters.

    Args:
        signals (ndarray): each array
        sample_freq (float): sampling_frequency
        result_cluster (dic): result from a clusterization function

    """
    ### Extract variables from clusterization result.
    n_clus = result_cluster['n_clus']
    low_dim_signal = result_cluster['low_dim_signal']
    labels = result_cluster['labels']

    ### Plot the clusters on the two first dimension of the low dim signal.
    plot_cluster(low_dim_signal, c_data=labels, dim=dim)

    ### Plot the average wave_form (and std) for each cluster.
    cluster_mean = [signals[labels == i, :].mean(axis=0) for i in range(n_clus)]
    cluster_std = [signals[labels == i, :].std(axis=0) for i in range(n_clus)]

    time = np.linspace(0, signals.shape[1]/sample_freq, signals.shape[1])*1000

    fig, ax = plt.subplots(figsize=(15, 5))
    for i in range(n_clus):
        clus_mean = cluster_mean[i]
        clus_std = cluster_std[i]
        ax.plot(time, clus_mean, label='Cluster {}'.format(i))
        ax.fill_between(time, clus_mean-clus_std, clus_mean+clus_std,
                        alpha=0.15)

    plt.legend()
    