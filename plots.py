import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from metrics import calculate_direct_af, calculate_transwidth



def plot_processed_dataframe(df, batch_col, raw_signal_col, smoothed_signal_col, derivative_col, example_batch):
    """
    Create a side-by-side plot of the raw and smoothed signals, and the first derivative for a specified batch.
    
    Args:
    df (pd.DataFrame): Input DataFrame containing the processed data
    batch_col (str): Name of the column containing batch information
    raw_signal_col (str): Name of the column containing the raw signal data
    smoothed_signal_col (str): Name of the column containing the smoothed signal data
    derivative_col (str): Name of the column containing the first derivative data
    example_batch: The specific batch to plot (should match a value in the batch_col)
    
    Returns:
    matplotlib.figure.Figure: The created figure containing the plots
    """
    # Filter the dataframe for the specified batch
    batch_data = df[df[batch_col] == example_batch].copy()
    batch_data = batch_data.sort_index()  # Ensure data is in order
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot raw and smoothed signals on the left subplot
    sns.lineplot(data=batch_data, x=batch_data.index, y=raw_signal_col, ax=ax1, label='Raw Signal')
    sns.lineplot(data=batch_data, x=batch_data.index, y=smoothed_signal_col, ax=ax1, label='Smoothed Signal')
    ax1.set_title(f'Raw and Smoothed Signals for Batch {example_batch}')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Signal Value')
    ax1.legend()
    
    # Plot first derivative on the right subplot
    sns.lineplot(data=batch_data, x=batch_data.index, y=derivative_col, ax=ax2)
    ax2.set_title(f'First Derivative for Batch {example_batch}')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Derivative Value')
    
    # Adjust layout and display the plot
    plt.tight_layout()
    
    return fig

# Example usage:
# fig = plot_signal_and_derivative(processed_df, 'batch_column', 'original_signal', 'smoothed_signal', 'first_derivative', 'Batch1')
# plt.show()



def create_control_charts(metrics_df, control_limits_df):
    # Ensure the required columns are present in both dataframes
    required_cols_metrics = ['Batch', 'Direct AF', 'Transwidth']
    required_cols_limits = ['Metric', 'Mean', 'LCL', 'UCL']
    
    if not all(col in metrics_df.columns for col in required_cols_metrics):
        raise ValueError(f"metrics_df is missing one or more required columns: {required_cols_metrics}")
    if not all(col in control_limits_df.columns for col in required_cols_limits):
        raise ValueError(f"control_limits_df is missing one or more required columns: {required_cols_limits}")

    # Create a control chart for each metric
    metrics = ['Direct AF', 'Transwidth']
    
    for metric in metrics:
        # Get the control limits for the current metric
        limits = control_limits_df[control_limits_df['Metric'] == metric].iloc[0]
        mean, lcl, ucl = limits['Mean'], limits['LCL'], limits['UCL']
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=metrics_df, x='Batch', y=metric, marker='o')
        
        # Add control limits
        plt.axhline(y=mean, color='g', linestyle='--', label='Mean')
        plt.axhline(y=lcl, color='r', linestyle='--', label='LCL')
        plt.axhline(y=ucl, color='r', linestyle='--', label='UCL')
        
        # Customize the plot
        plt.title(f'Control Chart for {metric}')
        plt.xlabel('Batch')
        plt.ylabel(metric)
        plt.legend()
        
        # Rotate x-axis labels by 45 degrees
        plt.xticks(rotation=45, ha='right')
        
        # Adjust the layout to prevent cut-off labels
        plt.tight_layout()
        
        # Show the plot
        plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_transwidth(df, volume_col, signal_col, batch_col, batch):
    """
    Plot the Transwidth calculation for a single batch using seaborn,
    with volume on the x-axis and signal on the y-axis.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, signal, and batch data.
    volume_col : str
        Column name for volume data (x-axis).
    signal_col : str
        Column name for signal data (y-axis).
    batch_col : str
        Column name for batch data.
    batch : str or int
        The specific batch to plot.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    This function uses the calculate_transwidth function to compute the Transwidth
    and visualizes the calculation process using a seaborn plot.
    """
    # Filter the dataframe for the specified batch
    batch_data = df[df[batch_col] == batch]

    if batch_data.empty:
        raise ValueError(f"No data found for batch '{batch}' in column '{batch_col}'")

    # Calculate Transwidth for the batch
    transwidth = calculate_transwidth(batch_data, volume_col, signal_col)

    # Sort the data by volume for plotting
    sorted_data = batch_data.sort_values(volume_col)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=sorted_data, x=volume_col, y=signal_col, alpha=0.6)
    
    # Interpolate to find volume values at 0.05 and 0.95 signal levels
    v_5 = np.interp(0.05, sorted_data[signal_col], sorted_data[volume_col])
    v_95 = np.interp(0.95, sorted_data[signal_col], sorted_data[volume_col])

    # Plot the Transwidth lines
    plt.vlines([v_5, v_95], 0, 1, colors=['red', 'red'], linestyles='dashed')
    plt.hlines([0.05, 0.95], sorted_data[volume_col].min(), sorted_data[volume_col].max(), colors=['green', 'green'], linestyles='dashed')

    # Annotate the plot
    plt.text(v_5, 0.5, f'V_5: {v_5:.2f}',   horizontalalignment='right')
    plt.text(v_95, 0.5, f'V_95: {v_95:.2f}',   horizontalalignment='left')
    plt.text((v_5 + v_95) / 2, 0.5, f'Transwidth: {transwidth:.2f}',  horizontalalignment='center')

    plt.title(f'Transwidth Calculation for {batch_col}: {batch}')
    plt.xlabel('Volume')
    plt.ylabel('Signal')

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_transwidth(results, volume_col="Volume", signal_col="normalized_signal", batch_col="Batch", batch="Batch_16")



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_direct_af(df, volume_col, signal_col, batch_col, batch):
    """
    Plot the Direct AF calculation for a single batch using seaborn,
    with volume on the x-axis and signal on the y-axis, including lines to demonstrate the pairs.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, signal, and batch data.
    volume_col : str
        Column name for volume data (x-axis).
    signal_col : str
        Column name for signal data (y-axis).
    batch_col : str
        Column name for batch data.
    batch : str or int
        The specific batch to plot.

    Returns
    -------
    None
        Displays the plot using matplotlib.

    Notes
    -----
    This function uses the calculate_direct_af function to compute the Direct AF
    and visualizes the calculation process using a seaborn plot.
    """
    # Filter the dataframe for the specified batch
    batch_data = df[df[batch_col] == batch]

    if batch_data.empty:
        raise ValueError(f"No data found for batch '{batch}' in column '{batch_col}'")

    # Calculate Direct AF for the batch
    direct_af = calculate_direct_af(batch_data, volume_col, signal_col)

    # Sort the data by volume for plotting
    sorted_data = batch_data.sort_values(volume_col)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=sorted_data, x=volume_col, y=signal_col, alpha=0.6)

    # Calculate the thresholds and corresponding volumes
    signal_range = sorted_data[signal_col].max() - sorted_data[signal_col].min()
    signal_min = sorted_data[signal_col].min()
    thresholds_low = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    thresholds_high = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]
    threshold_mid = 0.5

    cv_low = np.interp(np.array(thresholds_low) * signal_range + signal_min, sorted_data[signal_col], sorted_data[volume_col])
    cv_high = np.interp(np.array(thresholds_high) * signal_range + signal_min, sorted_data[signal_col], sorted_data[volume_col])
    cv_mid = np.interp(threshold_mid * signal_range + signal_min, sorted_data[signal_col], sorted_data[volume_col])

    # Plot the threshold lines
    plt.vlines(cv_low, 0, 1, colors='blue', linestyles='dashed', alpha=0.5, label='CV Low')
    plt.vlines([cv_mid], 0, 1, colors='red', linestyles='dashed', alpha=0.5, label='CV Mid')
    plt.vlines(cv_high, 0, 1, colors='green', linestyles='dashed', alpha=0.5, label='CV High')

    # Plot lines demonstrating the pairs
    for low, high in zip(cv_low, cv_high):
        plt.plot([low, cv_mid, high], [0.4, 0.5, 0.6], 'o-', color='purple', alpha=0.5)

    # Annotate the plot
    plt.text(cv_mid, 0.5, f'CV_mid: {cv_mid:.2f}', verticalalignment='center', horizontalalignment='right')
    plt.text(np.mean(cv_low), 0.3, 'CV_low',verticalalignment='bottom', horizontalalignment='center')
    plt.text(np.mean(cv_high), 0.7, 'CV_high', verticalalignment='top', horizontalalignment='center')
    plt.text(sorted_data[volume_col].max(), 0.5, f'Direct AF: {direct_af:.2f}', verticalalignment='center', horizontalalignment='right')

    plt.title(f'Direct AF Calculation for {batch_col}: {batch}')
    plt.xlabel('Volume')
    plt.ylabel('Signal')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_direct_af(results, volume_col="Volume", signal_col="normalized_signal", batch_col="Batch", batch="Batch_16")