from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def smooth_signal(df, window_length=30, polyorder=1, example_batch=None):
    # Group the dataframe by Batch
    """
    Smooth the signal for each batch in the dataframe and plot the result for an example batch.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    example_batch : str
        The batch to plot the smoothed signal for.
    window_length : int, optional
        The window length for the savgol_filter (default 30).
    polyorder : int, optional
        The polynomial order for the savgol_filter (default 1).

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the smoothed signal added as a column.
    
    Notes
    -----
    Used to smooth the signal for each batch in the dataframe.
    The smoothed signal is assigned to a new column called 'Smoothed_Signal'.
    A plot is generated for the example batch to show the smoothed signal.
    """
    grouped = df.groupby('Batch')
    
    # Apply savgol_filter to each group
    smoothed_signals = grouped['Signal'].apply(lambda x: savgol_filter(x, window_length, polyorder))
    
    # Flatten the result and assign it back to the dataframe
    df['Smoothed_Signal'] = np.concatenate(smoothed_signals.values)
    
    # Plot for the example batch
    
    
    if example_batch is not None:
        example_batch_data = df[df["Batch"] == example_batch]
    
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=example_batch_data, x='Volume', y='Signal', color='blue', label='Original Signal')
        sns.lineplot(data=example_batch_data, x='Volume', y='Smoothed_Signal', color='red', label='Smoothed Signal')
        plt.title(f'Signal vs Volume for {example_batch} with Smoothed Signal')
        plt.xlabel('Volume')
        plt.ylabel('Signal')
        plt.legend()
        plt.show()
    
    return df



def normalize_signal(df, example_batch=None):
    # Group the dataframe by Batch
    """
    Normalize the signal for each batch in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    example_batch : str or None, optional
        The batch to plot the normalized signal for. If None, no plot is generated.

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the normalized signal added as a column.

    Notes
    -----
    Used to normalize the signal for each batch in the dataframe.
    The normalized signal is assigned to a new column called 'Normalized_Signal'.
    A plot is generated for the example batch to show the normalized signal.
    """
    grouped = df.groupby('Batch')
    
    # Define the normalization function
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min())
    
    # Apply normalization to each group
    df['Normalized_Signal'] = grouped['Smoothed_Signal'].transform(normalize)
    
    # If an example batch is specified, plot it
    if example_batch is not None:
        example_batch_data = df[df["Batch"] == example_batch]
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=example_batch_data, x='Volume', y='Normalized_Signal', color='blue')
        
        plt.title(f'Signal vs Volume for {example_batch} with Normalized Signal')
        plt.xlabel('Volume')
        plt.ylabel('Normalized_Signal')
        plt.legend()
        plt.show()
    
    return df