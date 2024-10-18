import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from helper_functions import normalize_signal




def calculate_direct_af(volume, signal):
    # Sort the data by Volume
    """
    Calculate the Direct AF value from a pair of 1D arrays, 
    `volume` and `signal`, containing the volume and signal data, respectively.

    Parameters
    ----------
    volume : 1D array
        The volume data.
    signal : 1D array
        The signal data.

    Returns
    -------
    direct_af : float
        The Direct AF value.

    Notes
    -----
    Will calculate Direct AF on all data points in df. 
    Pass each batch individually to calculate the Direct AF for each batch.
    
    Example Usage
    -------
    calculate_direct_af(df['Volume'].values, df['Signal'].values)
    """
    sorted_indices = np.argsort(volume)
    volume = volume[sorted_indices]
    signal = signal[sorted_indices]



    # Create interpolation function
    interp_func = interp1d(signal, volume, kind='linear', bounds_error=False, fill_value="extrapolate")

    thresholds_low = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    thresholds_high = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    cv_mid = interp_func(0.5)

    ratios = []
    for low, high in zip(thresholds_low, thresholds_high):
        cv_a = interp_func(low)
        cv_b = interp_func(high)
        ratios.append((cv_b - cv_mid) / (cv_mid - cv_a))

    return np.mean(ratios)





def calculate_and_plot_direct_af(df, batch):
    # Filter the data for the specified batch
    """
    Calculate the Direct AF for a given batch of data and plot the result.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    batch : str
        The batch to calculate and plot the Direct AF for.

    Returns
    -------
    direct_af : float
        The Direct AF value.
    A matplotlib plot of the Direct AF calculation for the specified batch.

    Notes
    -----
    Expects a df with columns 'Batch', 'Volume', and 'Signal'. Filters for the specified batch.
    Pass each batch individually to calculate the Direct AF for each batch.

    Example Usage
    -------
    calculate_and_plot_direct_af(df, 'Batch_1')
    """
    batch_data = df[df['Batch'] == batch]
    
    # Check if the batch exists in the dataframe
    if batch_data.empty:
        print(f"Error: Batch '{batch}' not found in the dataframe.")
        return None
    
    # Sort the data by Volume
    batch_data = batch_data.sort_values('Volume')
    
    # Extract volume and signal data
    volume = batch_data['Volume'].values
    signal = batch_data['Signal'].values
    
    # Normalize the signal
    normalized_signal = normalize_signal(signal)
    
    # Create interpolation function
    interp_func = interp1d(normalized_signal, volume, kind='linear')
    
    # Define thresholds
    thresholds_low = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    thresholds_high = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    
    # Calculate cv_mid
    cv_mid = interp_func(0.5)
    
    # Calculate ratios and Direct AF
    ratios = []
    for low, high in zip(thresholds_low, thresholds_high):
        cv_a = interp_func(low)
        cv_b = interp_func(high)
        ratios.append((cv_b - cv_mid) / (cv_mid - cv_a))
    
    direct_af = np.mean(ratios)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot the normalized signal
    sns.lineplot(x=volume, y=normalized_signal, label='Normalized Signal')
    
    # Highlight the mid-point
    plt.scatter(cv_mid, 0.5, color='red', s=100, zorder=5, label='Mid-point')
    
    # Add vertical line at mid-point
    plt.axvline(x=cv_mid, color='red', linestyle='--', alpha=0.7, label='Mid-line')
    
    # Highlight the threshold points and add ratio values
    colors = plt.cm.rainbow(np.linspace(0, 1, len(thresholds_low)))
    for i, (low, high, ratio) in enumerate(zip(thresholds_low, thresholds_high, ratios)):
        cv_a = interp_func(low)
        cv_b = interp_func(high)
        plt.scatter([cv_a, cv_b], [low, high], color=colors[i], s=50, zorder=5)
        plt.plot([cv_a, cv_mid, cv_b], [low, 0.5, high], color=colors[i], linestyle=':', alpha=0.7)
        
        # Add ratio value to the plot
        mid_x = (cv_a + cv_b) / 2
        mid_y = (low + high) / 2
        plt.annotate(f'{ratio:.2f}', (mid_x, mid_y), color=colors[i], 
                     xytext=(5, 5), textcoords='offset points', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i], alpha=0.8),
                     ha='left', va='bottom')
    
    plt.title(f'Normalized Signal vs Volume for {batch} with Direct AF Visualization')
    plt.xlabel('Volume')
    plt.ylabel('Normalized Signal')
    plt.legend()
    
    # Add text annotation for Direct AF value
    plt.text(0.95, 0.05, f'Direct AF: {direct_af:.2f}', 
             horizontalalignment='right', verticalalignment='bottom',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return direct_af