import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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

