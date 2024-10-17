from helper_functions import normalize_signal
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_transwidth(volume, signal):
    """
    Calculate the Transwidth metric from a pair of 1D arrays, 
    `volume` and `signal`, containing the volume and signal data, respectively.

    Parameters
    ----------
    volume : 1D array
        The volume data.
    signal : 1D array
        The signal data.

    Returns
    -------
    transwidth : float
        The Transwidth value.

    Notes
    -----
    Will calculate Transwidth on all data points in df. 
    Pass each batch individually to calculate the Transwidth for each batch.

    Example Usage
    -------
    calculate_transwidth(df['Volume'].values, df['Signal'].values)
    """
    
    normalized_signal = normalize_signal(signal)
    cv_5 = np.interp(0.05, normalized_signal, volume)
    cv_95 = np.interp(0.95, normalized_signal, volume)
    return cv_95 - cv_5




def calculate_and_plot_transwidth(df, batch):
    # Filter the data for the specified batch
    """
    Calculate the Transwidth metric and plot the result for a given batch of data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    batch : str
        The batch to calculate and plot the Transwidth for.

    Returns
    -------
    transwidth : float
        The Transwidth value.

    Notes
    -----
    Expects a df with columns 'Batch', 'Volume', and 'Signal'. Filters for the specified batch.
    Pass each batch individually to calculate the Transwidth for each batch.

    Example Usage
    -------
    calculate_and_plot_transwidth(df, 'Batch_1')
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
    
    # Calculate Transwidth
    cv_5 = interp_func(0.05)
    cv_95 = interp_func(0.95)
    transwidth = cv_95 - cv_5
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot the normalized signal
    sns.lineplot(x=volume, y=normalized_signal, label='Normalized Signal')
    
    # Highlight the 5% and 95% points
    plt.scatter([cv_5, cv_95], [0.05, 0.95], color='red', s=100, zorder=5, label='Transwidth Points')
    
    # Add vertical lines at 5% and 95% points
    plt.axvline(x=cv_5, color='green', linestyle='--', alpha=0.7, label='5% Line')
    plt.axvline(x=cv_95, color='green', linestyle='--', alpha=0.7, label='95% Line')
    
    # Add horizontal lines at 5% and 95% levels
    plt.axhline(y=0.05, color='orange', linestyle=':', alpha=0.7)
    plt.axhline(y=0.95, color='orange', linestyle=':', alpha=0.7)
    
    # Add arrow to show Transwidth
    plt.annotate('', xy=(cv_95, 0.5), xytext=(cv_5, 0.5),
                 arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    
    plt.title(f'Normalized Signal vs Volume for {batch} with Transwidth Visualization')
    plt.xlabel('Volume')
    plt.ylabel('Normalized Signal')
    plt.legend()
    
    # Add text annotation for Transwidth value
    plt.text((cv_5 + cv_95) / 2, 0.52, f'Transwidth: {transwidth:.2f}', 
             horizontalalignment='center', verticalalignment='bottom',
             bbox=dict(facecolor='white', edgecolor='purple', alpha=0.8))
    
    plt.show()
    
    return transwidth

