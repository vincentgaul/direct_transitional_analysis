import pandas as pd
from direct_af import calculate_direct_af
from trans_width import calculate_transwidth

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

