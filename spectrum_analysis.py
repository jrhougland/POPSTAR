from specparam import SpectralModel, SpectralGroupModel
from mne.io import Raw, RawArray
from mne import Epochs, EpochsArray
import numpy as np
import matplotlib.pyplot as plt

def parameterize_spectrum(data, fmin=1, fmax=120, max_peaks=6, peak_width_limits=[1, 10], min_peak_height=0.2, ap_mode ='fixed', freq_range=[1, 120], save_fig=None, **kwargs) -> tuple[list, list]:
    """Parameterize the spectrum of a MEG dataset using SpecParam.

    Parameters
    ----------
    data : mne.io.Raw, mne.Epoch, mne.EpochsArray, or spectrum (2D numpy array: frequencies, power values)
        The MEG dataset to analyze.
    fmin : float
        The minimum frequency to consider for the PSD.
    fmax : float
        The maximum frequency to consider for the PSD.
    max_peaks : int
        The maximum number of peaks to fit in the spectrum.
    peak_width_limits : list
        The limits for the width of the peaks to fit.
    min_peak_height : float
        The minimum height of the peaks to fit.
    ap_mode : str
        The aperiodic mode to use ('fixed', 'knee' or 'doubexp').
    freq_range : list
        The frequency range to consider for fitting the model.
    save_fig : str
        The path to save the figure of the spectrum with the fitted model. If None, the figure will not be saved.
    **kwargs
        Additional keyword arguments to pass to the SpecParam model report() function.
    ----------
    Returns
    -------
    periodic_peaks : list
        The parameters of the periodic peaks in the spectrum.
    aperiodic_params : list
        The parameters of the aperiodic component of the spectrum.
    """
    # get psd
    if isinstance(data, (Raw, Epochs, EpochsArray, RawArray)):
        MNE_psd = data.compute_psd(fmin=fmin, fmax=fmax)
        spectrum = MNE_psd.get_data()[0, :]
        freqs = MNE_psd.freqs
    elif isinstance(data, (np.ndarray, list)) and len(data) == 2:
        freqs = data[0]
        spectrum = data[1]   

    else: raise ValueError("data must be either an mne.io.Raw object or a numpy array representing the spectrum.")

    if spectrum.ndim == 1:
        spec_model = SpectralModel(max_n_peaks=max_peaks, 
                                    peak_width_limits=peak_width_limits, 
                                    min_peak_height=min_peak_height,
                                    aperiodic_mode=ap_mode)

        # fit model and get output
        spec_model.report(freqs, spectrum, freq_range=freq_range, plt_log=False, **kwargs)

        if save_fig:
            plt.savefig(save_fig, dpi=400)

        # design a filter based on parameterized spectrum
        periodic_peaks, aperiodic_params = spec_model.get_params('periodic'), spec_model.get_params('aperiodic')

    elif spectrum.ndim == 2:

        spec_group_model = SpectralGroupModel(max_n_peaks=max_peaks, 
                                                peak_width_limits=peak_width_limits, 
                                                min_peak_height=min_peak_height,
                                                aperiodic_mode=ap_mode)

        # fit model and get output
        spec_group_model.report(freqs, spectrum, freq_range=freq_range, plt_log=False, **kwargs)

        if save_fig:
            plt.savefig(save_fig, dpi=400)

        # design a filter based on parameterized spectrum
        periodic_peaks, aperiodic_params = spec_group_model.get_params('periodic'), spec_group_model.get_params('aperiodic')
    
    else:
        raise ValueError("spectrum must be either a 1D or 2D array.")

    return periodic_peaks, aperiodic_params
        

def compute_peak_bands(peaks, bands) -> tuple[list, list]:
    """Get the peaks in a specific frequency band.

    Parameters
    ----------
    peaks : list
        The parameters of the peaks in the spectrum.
    bands : list of lists
        The frequency range(s) of the band(s) and peak_idx by amplitude to include in that band. Use 'all' to include all peaks in a band.
        E.g.: [[8, 12, 1]] or [[13, 30, 'all']]



    Returns
    -------
    upper, lower, band_nr : list
        The upper and lower bounds of the peaks in the specified bands, and the corresponding band number
    """
    
    upper, lower, band_nr = [], [], []
    for i, band in enumerate(bands):
        band_peaks = [peak for peak in peaks if band[0] <= peak[0] <= band[1]]
        if len(band_peaks) > 0:
            sorted_peaks = sorted(band_peaks, key=lambda x: x[1], reverse=True)
            if isinstance(band[-1], int):
                peaks_idx = band[-1]
                selected_peak = sorted_peaks[peaks_idx]
                upper.append(selected_peak[0] + selected_peak[2] / 2 * 2.355)
                lower.append(selected_peak[0] - selected_peak[2] / 2 * 2.355)
                band_nr.append(i)
            elif isinstance(band[-1], str) and band[-1] == 'all':
                selected_peak = sorted_peaks     
                for peak in selected_peak:
                    upper.append(peak[0] + peak[2] / 2 * 2.355)
                    lower.append(peak[0] - peak[2] / 2 * 2.355)
                    band_nr.append(i)
            else:
                raise ValueError("peaks_idx must be either an int or 'all'.")
        else:
            print(f"Warning: No peaks found in band {band[0]}-{band[1]} Hz.")
        
            upper.append(None)
            lower.append(None)
            band_nr.append(None)

    return [upper, lower, band_nr]

