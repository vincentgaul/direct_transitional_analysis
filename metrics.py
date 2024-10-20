import pandas as pd
import numpy as np


def calculate_direct_af(df, volume_col, signal_col, batch_col=None):
    """
    Calculate the Direct AF value for each batch of data from a DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, signal, and optionally batch data.
    volume_col : str
        Column name for volume data.
    signal_col : str
        Column name for signal data.
    batch_col : str, optional
        Column name for batch data. If None, calculates for the entire dataset.

    Returns
    -------
    direct_af : pandas Series
        A Series of Direct AF values, one for each batch (or a single value if no batch_col is provided).
    """

    def compute_direct_af(volume, signal):
        # Sort the data by Volume
        sorted_indices = np.argsort(volume)
        volume = volume[sorted_indices]
        signal = signal[sorted_indices]

        thresholds_low = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        thresholds_high = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
        
        # Use np.interp for interpolation
        signal_range = signal.max() - signal.min()
        cv_mid = np.interp(0.5 * signal_range + signal.min(), signal, volume)
        cv_low = np.interp(np.array(thresholds_low) * signal_range + signal.min(), signal, volume)
        cv_high = np.interp(np.array(thresholds_high) * signal_range + signal.min(), signal, volume)

        ratios = (cv_high - cv_mid) / (cv_mid - cv_low)
        return np.mean(ratios)

    if batch_col:
        # Group by batch and apply the Direct AF calculation to each group
        return df.groupby(batch_col).apply(lambda x: compute_direct_af(x[volume_col].values, x[signal_col].values))
    else:
        # Apply the calculation on the entire dataset
        return compute_direct_af(df[volume_col].values, df[signal_col].values)


def calculate_transwidth(df, volume_col, signal_col, batch_col=None):
    """
    Calculate the Transwidth metric from a DataFrame, grouping by batches if a batch column is provided.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, signal, and optionally batch data.
    volume_col : str
        Column name for volume data.
    signal_col : str
        Column name for signal data.
    batch_col : str, optional
        Column name for batch data. If None, calculates for the entire dataset.

    Returns
    -------
    transwidth : pandas Series or float
        Transwidth values for each batch if batch_col is provided, or a single Transwidth value if no batch_col is given.

    Notes
    -----
    Transwidth is calculated as the difference between the volume values at the 0.05 and 0.95 signal levels.
    """

    def compute_transwidth(volume, signal):
        # Ensure volume and signal are sorted together by signal for interpolation
        sorted_indices = np.argsort(signal)
        volume = volume[sorted_indices]
        signal = signal[sorted_indices]
        
        # Interpolate to find volume values at 0.05 and 0.95 signal levels
        cv_5 = np.interp(0.05, signal, volume)
        cv_95 = np.interp(0.95, signal, volume)
        
        # Return the Transwidth value
        return cv_95 - cv_5

    if batch_col:
        # Group by batch and apply the Transwidth calculation to each group
        return df.groupby(batch_col).apply(lambda x: compute_transwidth(x[volume_col].values, x[signal_col].values))
    else:
        # Apply the calculation on the entire dataset
        return compute_transwidth(df[volume_col].values, df[signal_col].values)





def calculate_metrics(df, volume_col, signal_col, batch_col=None):
   
    # Calculate Direct AF values
    direct_af_values = calculate_direct_af(df, volume_col, signal_col, batch_col)
    
    # Calculate Transwidth values
    transwidth_values = calculate_transwidth(df, volume_col, signal_col, batch_col)
    
    # Combine results into a DataFrame
    if batch_col:
        result_df = pd.DataFrame({
            batch_col: direct_af_values.index,
            'Direct AF': direct_af_values.values,
            'Transwidth': transwidth_values.values
        })
    else:
        # If no batch, just create a DataFrame with a single row
        result_df = pd.DataFrame({
            'Batch': ['All Data'],
            'Direct AF': [direct_af_values],
            'Transwidth': [transwidth_values]
        })
    
    # Sort the DataFrame by the Batch column if batch_col is provided
    if batch_col:
        result_df = result_df.sort_values(by=batch_col)
    
    return result_df


def calculate_control_limits(df):
    
    
    
    # Convert numeric columns to float
    df['Direct AF'] = df['Direct AF'].astype(float)
    df['Transwidth'] = df['Transwidth'].astype(float)
    
    # Calculate means
    direct_af_mean = df['Direct AF'].mean()
    transwidth_mean = df['Transwidth'].mean()
    
    # Calculate standard deviations
    direct_af_std = df['Direct AF'].std()
    transwidth_std = df['Transwidth'].std()
    
    # Calculate control limits (mean ± 3 * standard deviation)
    direct_af_lcl = direct_af_mean - 3 * direct_af_std
    direct_af_ucl = direct_af_mean + 3 * direct_af_std
    transwidth_lcl = transwidth_mean - 3 * transwidth_std
    transwidth_ucl = transwidth_mean + 3 * transwidth_std
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Metric': ['Direct AF', 'Transwidth'],
        'Mean': [direct_af_mean, transwidth_mean],
        'LCL': [direct_af_lcl, transwidth_lcl],
        'UCL': [direct_af_ucl, transwidth_ucl]
    })
    
    return results_df

