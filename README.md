# Pipeline for Spike sorting algorithms

Most of the work is done in the script file (helper_function.py, feature_extraction.py, visualization.py) and the main.py uses these different functions to cluster signal amplitude into the different spikes that initiated it.  

The spike_sorting.ipynb notebook is the interactive version of the main.py

Ressource I used:
- https://towardsdatascience.com/using-signal-processing-to-extract-neural-events-in-python-964437dc7c0

- Quian Quiroga R, Nadasdy Z, Ben-Shaul Y. Unsupervised spike detection and sorting with wavelets and superparamagnetic clustering. Neural Comput 16: 1661–1687, 2004. doi:10.1162/089976604774201631. 

- See also new wave_clus implementation: A novel and fully automatic spike sorting implementation with variable number of features. F. J. Chaure, H. G. Rey and R. Quian Quiroga. Journal of Neurophysiology; 2018. https://doi.org/10.1152/jn.00339.2018