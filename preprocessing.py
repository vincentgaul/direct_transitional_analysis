

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def process_dataframe(df, batch_col, signal_col, 
                      window_length=100, polyorder=1, num_smoothing_passes=1,
                      smoothed_col_name='smoothed_signal', 
                      normalized_col_name='normalized_signal',
                      derivative_col_name='first_derivative'):
    """
    Process a pandas DataFrame by grouping by batch, applying smoothing multiple times,
    normalizing the smoothed signal, and calculating the first derivative, with custom column names.
    
    Args:
    df (pd.DataFrame): Input DataFrame
    batch_col (str): Name of the column containing batch information
    signal_col (str): Name of the column containing the signal to be processed
    window_length (int): The length of the filter window (must be odd). Default is 11.
    polyorder (int): The order of the polynomial used to fit the samples. Default is 3.
    num_smoothing_passes (int): Number of times to apply the Savitzky-Golay filter. Default is 1.
    smoothed_col_name (str): Name for the new column containing smoothed data. Default is 'smoothed_signal'.
    normalized_col_name (str): Name for the new column containing normalized data. Default is 'normalized_signal'.
    derivative_col_name (str): Name for the new column containing derivative data. Default is 'first_derivative'.
    
    Returns:
    pd.DataFrame: Processed DataFrame with new columns for smoothed, normalized, and derivative data
    """
    
    def smooth_normalize_and_derive(group):
        # Apply Savitzky-Golay filter for smoothing multiple times
        smoothed_signal = group[signal_col].values
        for _ in range(num_smoothing_passes):
            smoothed_signal = savgol_filter(smoothed_signal, window_length, polyorder)
        
        group[smoothed_col_name] = smoothed_signal
        
        # Normalize the smoothed signal
        min_val = np.min(smoothed_signal)
        max_val = np.max(smoothed_signal)
        group[normalized_col_name] = (smoothed_signal - min_val) / (max_val - min_val)
        
        # Calculate first derivative of the normalized signal
        group[derivative_col_name] = np.gradient(group[normalized_col_name])
        
        return group
    
    # Group by batch and apply the smoothing, normalization, and derivative calculations
    processed_df = df.groupby(batch_col, group_keys=False).apply(smooth_normalize_and_derive)
    
    return processed_df

