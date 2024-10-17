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