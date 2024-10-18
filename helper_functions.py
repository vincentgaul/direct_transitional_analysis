from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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




def calculate_smooth_first_derivative(df, example_batch=None, smooth_original=True, smooth_derivative=True, window_length=11, polyorder=3):
    # Group the dataframe by Batch
    """
    Calculate the smoothed first derivative for each batch in the dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    example_batch : str, optional
        The batch to plot the smoothed first derivative for (default None).
    smooth_original : bool, optional
        Whether to smooth the original signal before calculating the derivative (default True).
    smooth_derivative : bool, optional
        Whether to smooth the derivative after calculation (default True).
    window_length : int, optional
        The window length for the savgol_filter (default 11).
    polyorder : int, optional
        The polynomial order for the savgol_filter (default 3).

    Returns
    -------
    df : pandas.DataFrame
        The dataframe with the smoothed first derivative added as a column.

    Notes
    -----
    Used to calculate the smoothed first derivative for each batch in the dataframe.
    A plot is generated for the example batch to show the smoothed first derivative.
    """
    
    grouped = df.groupby('Batch')
    
    # Define the smoothing function
    def smooth(x):
        return savgol_filter(x, window_length, polyorder)
    
    # Define the derivative function
    def derivative(group):
        signal = group['Signal']
        volume = group['Volume']
        
        if smooth_original:
            signal = smooth(signal)
        
        # Calculate the gradient
        grad = np.gradient(signal, volume)
        
        if smooth_derivative:
            grad = smooth(grad)
        
        return pd.Series(grad, index=group.index)
    
    # Apply smoothing to Normalized_Signal if required
    if smooth_original:
        df['Smoothed_Signal'] = grouped['Normalized_Signal'].transform(smooth)
    
    # Apply the derivative function to each group
    df['First_Derivative'] = grouped.apply(derivative).reset_index(level=0, drop=True)
    
    # If an example batch is specified, plot it
    if example_batch is not None:
        example_batch_data = df[df["Batch"] == example_batch]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot Normalized Signal
        sns.lineplot(data=example_batch_data, x='Volume', y='Normalized_Signal', ax=ax1, color='blue', label='Original')
        if smooth_original:
            sns.lineplot(data=example_batch_data, x='Volume', y='Smoothed_Signal', ax=ax1, color='red', label='Smoothed')
        ax1.set_title(f'{"Smoothed " if smooth_original else ""}Normalized Signal vs Volume for {example_batch}')
        ax1.set_ylabel('Normalized Signal')
        ax1.legend()
        
        # Plot First Derivative
        sns.lineplot(data=example_batch_data, x='Volume', y='First_Derivative', ax=ax2, color='green')
        ax2.set_title(f'{"Smoothed " if smooth_derivative else ""}First Derivative vs Volume for {example_batch}')
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('First Derivative')
        
        plt.tight_layout()
        plt.show()
    
    return df