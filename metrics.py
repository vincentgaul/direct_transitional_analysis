import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def calculate_direct_af(df, volume_col, signal_col, batch_col=None):
    """
    Calculate the Direct AF value for each batch of data from a DataFrame,
    maintaining the original order of batches.

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
    direct_af : pandas Series or float
        A Series of Direct AF values in the original batch order, or a single float if no batch_col is provided.
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
        # Get unique batches in their original order
        unique_batches = df[batch_col].unique()
        
        # Calculate Direct AF for each batch
        direct_af_values = []
        for batch in unique_batches:
            batch_data = df[df[batch_col] == batch]
            direct_af = compute_direct_af(batch_data[volume_col].values, batch_data[signal_col].values)
            direct_af_values.append(direct_af)
        
        # Create a Series with the original batch order
        return pd.Series(direct_af_values, index=unique_batches, name='direct_af')
    else:
        # Apply the calculation on the entire dataset
        return compute_direct_af(df[volume_col].values, df[signal_col].values)

# Usage example:
# df = pd.DataFrame({
#     'batch': ['A', 'A', 'B', 'B', 'C', 'C'],
#     'volume': [1, 2, 3, 4, 5, 6],
#     'signal': [10, 20, 30, 40, 50, 60]
# })
# result = calculate_direct_af(df, 'volume', 'signal', 'batch')
# print(result)



def calculate_transwidth(df, volume_col, signal_col, batch_col=None):
    """
    Calculate the Transwidth metric from a DataFrame, maintaining the original order of batches.

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
        A Series of Transwidth values in the original batch order, or a single float if no batch_col is provided.

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
        # Get unique batches in their original order
        unique_batches = df[batch_col].unique()
        
        # Calculate Transwidth for each batch
        transwidth_values = []
        for batch in unique_batches:
            batch_data = df[df[batch_col] == batch]
            transwidth = compute_transwidth(batch_data[volume_col].values, batch_data[signal_col].values)
            transwidth_values.append(transwidth)
        
        # Create a Series with the original batch order
        return pd.Series(transwidth_values, index=unique_batches, name='transwidth')
    else:
        # Apply the calculation on the entire dataset
        return compute_transwidth(df[volume_col].values, df[signal_col].values)

# Usage example:
# df = pd.DataFrame({
#     'batch': ['A', 'A', 'B', 'B', 'C', 'C'],
#     'volume': [1, 2, 3, 4, 5, 6],
#     'signal': [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
# })
# result = calculate_transwidth(df, 'volume', 'signal', 'batch')
# print(result)





def calculate_inflection_points(df, volume_col, deriv_col, batch_col=None):
    """
    Calculate the number of inflection points for each batch of data from a DataFrame,
    maintaining the original order of batches.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume and derivative data.
    volume_col : str
        Column name for volume data.
    deriv_col : str
        Column name for derivative data.
    batch_col : str, optional
        Column name for batch data. If None, calculates for the entire dataset.

    Returns
    -------
    inflection_points : pandas DataFrame or dict
        If batch_col provided: DataFrame with batch identifiers and number of inflection points
        If no batch_col: Dictionary with single inflection point count
    """
    def compute_inflection_points(volume, derivative):
        """
        Compute the number of inflection points for a set of volume and derivative data.

        Parameters
        ----------
        volume : numpy array
            Array of volume data points.
        derivative : numpy array
            Array of derivative values.

        Returns
        -------
        int
            Number of inflection points detected.
        """
        # Calculate minimum peak height threshold (10% of max derivative)
        min_height = 0.1 * np.max(derivative)
        
        # Find peaks in the derivative (inflection points in original data)
        peaks, _ = find_peaks(derivative, height=min_height)
        
        return len(peaks)

    if batch_col:
        # Get unique batches in their original order
        unique_batches = df[batch_col].unique()
        
        # Calculate inflection points for each batch
        results = []
        for batch in unique_batches:
            batch_data = df[df[batch_col] == batch]
            num_peaks = compute_inflection_points(
                batch_data[volume_col].values,
                batch_data[deriv_col].values
            )
            results.append({
                'batch': batch,
                'num_inflection_points': num_peaks
            })
        
        # Create DataFrame with results in original batch order
        return pd.DataFrame(results)
    else:
        # Apply the calculation on the entire dataset
        num_peaks = compute_inflection_points(
            df[volume_col].values,
            df[deriv_col].values
        )
        return {'num_inflection_points': num_peaks}











def calculate_metrics(df, volume_col, signal_col, deriv_col, batch_col=None):
    """
    Calculate multiple chromatography metrics for each batch of data.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, signal, and optionally batch data.
    volume_col : str
        Column name for volume data.
    signal_col : str
        Column name for signal data.
    deriv_col : str
        Column name for derivative data.
    batch_col : str, optional
        Column name for batch data. If None, calculates for the entire dataset.

    Returns
    -------
    pandas DataFrame
        A DataFrame containing all calculated metrics, with batch identifiers if provided.
    """
    # Calculate Direct AF values
    direct_af_values = calculate_direct_af(df, volume_col, signal_col, batch_col)

    # Calculate Transwidth values
    transwidth_values = calculate_transwidth(df, volume_col, signal_col, batch_col)

    # Calculate Inflection Points
    inflection_points = calculate_inflection_points(df, volume_col, deriv_col, batch_col)

    # Combine results into a DataFrame
    if batch_col:
        result_df = pd.DataFrame({
            batch_col: direct_af_values.index,
            'Direct AF': direct_af_values.values,
            'Transwidth': transwidth_values.values,
            'Inflection Points': inflection_points['num_inflection_points'].values
        })
    else:
        # If no batch, just create a DataFrame with a single row
        result_df = pd.DataFrame({
            'Batch': ['All Data'],
            'Direct AF': [direct_af_values],
            'Transwidth': [transwidth_values],
            'Inflection Points': [inflection_points['num_inflection_points']]
        })

    # Sort the DataFrame by the Batch column if batch_col is provided
    if batch_col:
        result_df = result_df.sort_index()

    return result_df


def calculate_control_limits(df):
    """
    Calculate control limits (mean ± 3 standard deviations) for each metric.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing 'Direct AF', 'Transwidth', and 'Inflection Points' columns.

    Returns
    -------
    pandas DataFrame
        A DataFrame containing mean and control limits for each metric.
    """
    # Convert numeric columns to float
    df['Direct AF'] = df['Direct AF'].astype(float)
    df['Transwidth'] = df['Transwidth'].astype(float)
    df['Inflection Points'] = df['Inflection Points'].astype(float)
    
    # Calculate means for each metric
    direct_af_mean = df['Direct AF'].mean()
    transwidth_mean = df['Transwidth'].mean()
    inflection_mean = df['Inflection Points'].mean()
    
    # Calculate standard deviations for each metric
    direct_af_std = df['Direct AF'].std()
    transwidth_std = df['Transwidth'].std()
    inflection_std = df['Inflection Points'].std()
    
    # Calculate control limits (mean ± 3 * standard deviation)
    direct_af_lcl = direct_af_mean - 3 * direct_af_std
    direct_af_ucl = direct_af_mean + 3 * direct_af_std
    transwidth_lcl = transwidth_mean - 3 * transwidth_std
    transwidth_ucl = transwidth_mean + 3 * transwidth_std
    inflection_lcl = inflection_mean - 3 * inflection_std
    inflection_ucl = inflection_mean + 3 * inflection_std
    
    # Create a DataFrame with the results
    results_df = pd.DataFrame({
        'Metric': ['Direct AF', 'Transwidth', 'Inflection Points'],
        'Mean': [direct_af_mean, transwidth_mean, inflection_mean],
        'LCL': [direct_af_lcl, transwidth_lcl, inflection_lcl],
        'UCL': [direct_af_ucl, transwidth_ucl, inflection_ucl]
    })
    
    return results_df

