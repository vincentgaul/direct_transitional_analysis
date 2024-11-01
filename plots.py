import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from metrics import calculate_direct_af, calculate_transwidth, calculate_inflection_points, calculate_max_rate_of_change
from scipy.signal import find_peaks



def plot_processed_dataframe(df, batch_col, raw_signal_col, smoothed_signal_col, derivative_col, example_batch):
    """
    Create a side-by-side plot of the raw and smoothed signals, and the first derivative for a specified batch.
    """
    # Filter the dataframe for the specified batch
    batch_data = df[df[batch_col] == example_batch].copy()
    batch_data = batch_data.sort_index()
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Set the figure background color to white
    fig.patch.set_facecolor('white')  # Add this line
    # Set the subplot background colors to white
    ax1.set_facecolor('white')  # Add this line
    ax2.set_facecolor('white')  # Add this line
    
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
    
    plt.tight_layout()
    plt.close()
    return fig

# Example usage:
# fig = plot_signal_and_derivative(processed_df, 'batch_column', 'original_signal', 'smoothed_signal', 'first_derivative', 'Batch1')
# plt.show()



def create_control_charts(metrics_df, control_limits_df):
    """
    Create control charts for Direct AF, Transwidth, Inflection Points, and (dC/dV)max metrics.

    Parameters
    ----------
    metrics_df : pandas DataFrame
        DataFrame containing metrics data with 'Batch', 'Direct AF', 'Transwidth', 
        'Inflection Points', and '(dC/dV)max' columns.
    control_limits_df : pandas DataFrame
        DataFrame containing control limits with 'Metric', 'Mean', 'LCL', 'UCL' columns.

    Returns
    -------
    None
        Displays the control charts.
    """
    # Ensure the required columns are present in both dataframes
    required_cols_metrics = ['Batch', 'Direct AF', 'Transwidth', 'Inflection Points', 'Max Rate of Change']
    required_cols_limits = ['Metric', 'Mean', 'LCL', 'UCL']
    
    if not all(col in metrics_df.columns for col in required_cols_metrics):
        raise ValueError(f"metrics_df is missing one or more required columns: {required_cols_metrics}")
    if not all(col in control_limits_df.columns for col in required_cols_limits):
        raise ValueError(f"control_limits_df is missing one or more required columns: {required_cols_limits}")

    # Create a control chart for each metric
    metrics = ['Direct AF', 'Transwidth', 'Inflection Points', 'Max Rate of Change']
    
    # Set up the figure layout for all four plots
    fig, axs = plt.subplots(4, 1, figsize=(12, 24))
    
    for idx, (metric, ax) in enumerate(zip(metrics, axs)):
        # Get the control limits for the current metric
        limits = control_limits_df[control_limits_df['Metric'] == metric].iloc[0]
        mean, lcl, ucl = limits['Mean'], limits['LCL'], limits['UCL']
        
        # Create the plot
        sns.lineplot(data=metrics_df, x='Batch', y=metric, marker='o', ax=ax)
        
        # Add control limits
        ax.axhline(y=mean, color='g', linestyle='--', label='Mean')
        ax.axhline(y=lcl, color='r', linestyle='--', label='LCL')
        ax.axhline(y=ucl, color='r', linestyle='--', label='UCL')
        
        # Customize the plot
        ax.set_title(f'Control Chart for {metric}')
        ax.set_xlabel('Batch')
        ax.set_ylabel(metric)
        ax.legend()
        
        # Rotate x-axis labels by 45 degrees
        ax.tick_params(axis='x', rotation=45)
        
        # Identify points outside control limits
        out_of_control = metrics_df[
            (metrics_df[metric] > ucl) | 
            (metrics_df[metric] < lcl)
        ]
        
        # Highlight out-of-control points if any exist
        if not out_of_control.empty:
            ax.scatter(
                out_of_control['Batch'],
                out_of_control[metric],
                color='red',
                s=100,
                zorder=5,
                label='Out of Control'
            )
            
            # Add batch labels for out-of-control points
            for idx, row in out_of_control.iterrows():
                ax.annotate(
                    f"Batch {row['Batch']}", 
                    (row['Batch'], row[metric]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    
    # Show all plots
    plt.show()

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
    plt.figure(figsize=(12, 8))
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
    plt.figure(figsize=(12, 8))
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




def plot_inflection_points(df, volume_col, deriv_col, batch_col, batch):
   """
   Plot the inflection points calculation for a single batch,
   showing the first derivative with detected peaks marked.

   Parameters
   ----------
   df : pandas DataFrame
       DataFrame containing the volume, derivative, and batch data.
   volume_col : str
       Column name for volume data (x-axis).
   deriv_col : str
       Column name for derivative data (y-axis).
   batch_col : str
       Column name for batch data.
   batch : str or int
       The specific batch to plot.

   Returns
   -------
   None
       Displays the plot using matplotlib.
   """
   # Filter the dataframe for the specified batch
   batch_data = df[df[batch_col] == batch]

   if batch_data.empty:
       raise ValueError(f"No data found for batch '{batch}' in column '{batch_col}'")

   # Calculate inflection points for the batch
   inflection_result = calculate_inflection_points(
       batch_data, 
       volume_col=volume_col,
       deriv_col=deriv_col
   )
   num_peaks = inflection_result['num_inflection_points']

   # Sort the data by volume for plotting
   sorted_data = batch_data.sort_values(volume_col)
   
   # Calculate peak locations
   derivative = sorted_data[deriv_col].values
   min_height = 0.1 * np.max(derivative)  # 10% of max derivative as threshold
   peaks, properties = find_peaks(derivative, height=min_height)
   
   # Create the plot
   fig, ax = plt.subplots(figsize=(12, 8))  # Increased figure height
   
   # Plot derivative with some transparency
   ax.plot(sorted_data[volume_col], derivative, 'b-', alpha=0.7, label='First Derivative')
   
   # Mark peaks with transparency
   inflection_volumes = sorted_data[volume_col].iloc[peaks]
   inflection_derivatives = derivative[peaks]
   ax.scatter(inflection_volumes, inflection_derivatives, 
             color='red', marker='o', s=100, alpha=0.6, label='Inflection Points',
             edgecolor='darkred', linewidth=1)  # Added edge color for better visibility
   
   # Add threshold line
   ax.axhline(y=min_height, color='r', linestyle='--', 
              alpha=0.5, label='Peak Threshold')
   
   # Get the y-axis limits for calculating annotation positions
   ymin, ymax = ax.get_ylim()
   y_range = ymax - ymin
   
   # Create alternating offsets for annotations with increased spacing
   for i, (x, y) in enumerate(zip(inflection_volumes, inflection_derivatives)):
       # Alternate between top and bottom placement with larger offsets
       if i % 2 == 0:
           y_offset = 40  # Increased upward offset
           va = 'bottom'
       else:
           y_offset = -40  # Increased downward offset
           va = 'top'
       
       # Add horizontal offset alternating left and right
       x_offset = 20 if i % 4 >= 2 else -20
       
       ax.annotate(
           f'V={x:.2f}', 
           xy=(x, y),
           xytext=(x_offset, y_offset),
           textcoords='offset points',
           ha='center' if x_offset == 0 else ('right' if x_offset < 0 else 'left'),
           va=va,
           fontsize=8,
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', alpha=0.6)  # Added alpha to arrows
       )
   
   plt.title(f'First Derivative and Inflection Points for {batch_col}: {batch}\n(Number of Peaks: {num_peaks})')
   plt.xlabel('Volume')
   plt.ylabel('First Derivative')
   plt.grid(True, alpha=0.3)
   plt.legend()
   
   # Adjust plot limits to accommodate annotations with more padding
   current_ymin, current_ymax = ax.get_ylim()
   ax.set_ylim(current_ymin - 0.2 * y_range, current_ymax + 0.2 * y_range)  # Increased padding
   
   plt.tight_layout()
   plt.show()
   
   
def plot_max_rate_of_change(df, volume_col, deriv_col, batch_col, batch, window_size=10):
    """
    Plot the steepest slope (maximum rate of change) calculation for a single batch,
    showing both the original and smoothed derivative with the maximum slope point marked.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing the volume, derivative, and batch data.
    volume_col : str
        Column name for volume data (x-axis).
    deriv_col : str
        Column name for derivative data (y-axis).
    batch_col : str
        Column name for batch data.
    batch : str or int
        The specific batch to plot.
    window_size : int, optional
        Size of the moving average window (default: 10)

    Returns
    -------
    matplotlib.figure.Figure
        The created figure containing the plot
    """
    # Filter the dataframe for the specified batch
    batch_data = df[df[batch_col] == batch]

    if batch_data.empty:
        raise ValueError(f"No data found for batch '{batch}' in column '{batch_col}'")

    # Sort the data by volume for plotting
    sorted_data = batch_data.sort_values(volume_col)
    original_derivative = sorted_data[deriv_col].values
    
    # Calculate smoothed derivative
    smoothed_derivative = pd.Series(original_derivative).rolling(
        window=window_size, center=True).mean()
    # Handle NaN values at edges
    smoothed_derivative = smoothed_derivative.fillna(method='bfill').fillna(method='ffill')
    smoothed_derivative = smoothed_derivative.values
    
    # Calculate second derivative
    second_derivative = np.gradient(smoothed_derivative, sorted_data[volume_col].values)
    
    # Find point of maximum slope
    max_slope_idx = np.argmax(np.abs(second_derivative))
    max_volume = sorted_data[volume_col].iloc[max_slope_idx]
    max_derivative = smoothed_derivative[max_slope_idx]
    slope = second_derivative[max_slope_idx]
    
    # Create the plot with white background
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot original derivative with low opacity
    ax.plot(sorted_data[volume_col], original_derivative, 'b-', 
            alpha=0.3, label='Original Derivative')
            
    # Plot smoothed derivative
    ax.plot(sorted_data[volume_col], smoothed_derivative, 'b-', 
            alpha=0.7, label='Smoothed Derivative')
    
    # Mark maximum slope point with transparency
    ax.scatter(max_volume, max_derivative, 
              color='red', marker='o', s=100, alpha=0.6, 
              label='Maximum Slope Point', edgecolor='darkred', linewidth=1)
    
    # Add slope line
    line_length = (sorted_data[volume_col].max() - sorted_data[volume_col].min()) * 0.2
    x_range = line_length
    y_range = slope * x_range
    x_left = max_volume - x_range/2
    x_right = max_volume + x_range/2
    y_left = max_derivative - y_range/2
    y_right = max_derivative + y_range/2
    
    # Plot slope line
    ax.plot([x_left, x_right], [y_left, y_right], 'r--', 
            linewidth=2, alpha=0.6, label='Maximum Slope')
    
    # Get the y-axis limits for calculating annotation positions
    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    
    # Add annotation for maximum slope point
    ax.annotate(
        f'Max Slope Point\nV={max_volume:.2f}\nSlope={slope:.2f}', 
        xy=(max_volume, max_derivative),
        xytext=(30, 30),
        textcoords='offset points',
        ha='left',
        va='bottom',
        fontsize=8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
        arrowprops=dict(arrowstyle='->', 
                       connectionstyle='arc3,rad=0.2', 
                       alpha=0.6)
    )
    
    plt.title(f'First Derivative and Maximum Slope for {batch_col}: {batch}\n' + 
              f'(Max Slope: {slope:.2f}, Window Size: {window_size})')
    plt.xlabel('Volume')
    plt.ylabel('First Derivative')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Adjust plot limits to accommodate annotations
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(current_ymin - 0.2 * y_range, 
                current_ymax + 0.2 * y_range)
    
    plt.tight_layout()
    plt.close()
    return fig

# Required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example usage
# fig = plot_max_rate_of_change(df, 'volume', 'derivative', 'batch', 'batch1')
# plt.show()