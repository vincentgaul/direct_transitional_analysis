from scipy.signal import savgol_filter
import numpy as np

def normalize_signal(signal):
    """
    Normalize the signal to the range [0, 1].

    Parameters
    ----------
    signal : 1D array
        The signal to normalize.

    Returns
    -------
    normalized_signal : 1D array
        The normalized signal.
    
    Notes
    -----
    Used within TA functions
        
    Example Usage
    -------
    normalize_signal(df['Signal'].values)
    """
    return (signal - signal.min()) / (signal.max() - signal.min())



def calculate_derivative(volume, signal):
    # Smooth the signal first to reduce noise
    signal_smooth = savgol_filter(signal, window_length=11, polyorder=3)
    # Calculate the derivative
    dydx = np.gradient(signal_smooth, volume)
    return dydx