#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:15:35 2024

@author: jacobvanalmelo
"""

'''
this script is to be the working document to retrieve and prepare the last x timesteps from a list of buoy ids
'''

import numpy as np
import requests
from lxml import html
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import timedelta



final_X_columns = ['new_df101_WVHT', 'new_df101_DPD', 'df000_WVHT', 'df000_DPD',
       'df201_WVHT', 'df201_DPD', 'new_df101_MWD_E', 'new_df101_MWD_ENE',
       'new_df101_MWD_ESE', 'new_df101_MWD_N', 'new_df101_MWD_NE',
       'new_df101_MWD_NNE', 'new_df101_MWD_NNW', 'new_df101_MWD_NW',
       'new_df101_MWD_S', 'new_df101_MWD_SE', 'new_df101_MWD_SSE',
       'new_df101_MWD_SSW', 'new_df101_MWD_SW', 'new_df101_MWD_W',
       'new_df101_MWD_WNW', 'new_df101_MWD_WSW', 'df000_MWD_E',
       'df000_MWD_ENE', 'df000_MWD_ESE', 'df000_MWD_N', 'df000_MWD_NE',
       'df000_MWD_NNE', 'df000_MWD_NNW', 'df000_MWD_NW', 'df000_MWD_S',
       'df000_MWD_SE', 'df000_MWD_SSE', 'df000_MWD_SSW', 'df000_MWD_SW',
       'df000_MWD_W', 'df000_MWD_WNW', 'df000_MWD_WSW', 'df201_MWD_ENE',
       'df201_MWD_N', 'df201_MWD_NE', 'df201_MWD_NNE', 'df201_MWD_NNW',
       'df201_MWD_NW', 'df201_MWD_W', 'df201_MWD_WNW', 'df201_MWD_WSW',
       'time']
buoy_ids = ['51101', '51000', '51201']
cols_to_keep = ['time', 'WVHT', 'DPD', 'MWD']
scaler_compatible_columns = ['time', 'new_df101_WVHT', 'new_df101_DPD', 'new_df101_MWD',
       'df000_WVHT', 'df000_DPD', 'df000_MWD', 'df201_WVHT', 'df201_DPD',
       'df201_MWD']
model_name = 'MAPE10.02_losshuber_optGD_7_lstm_2d_1d_batch32_LR0.004_epochs20.sav'
n_outputs = 1
n_timesteps = 16
num_outputs = 1
lead_time = 6
window = 7

def scrape_buoy_data(buoy_id):
    '''
    This function gets all of the last 12 houra of data from a buoy from national buoy datacenter
    '''
    url = f'https://www.ndbc.noaa.gov/station_page.php?station={buoy_id}'
    response = requests.get(url)
    
    # Parse the HTML
    tree = html.fromstring(response.content)

    # Extract table rows
    rows = tree.xpath('//*[@id="wxdata"]/div/table/tbody/tr')
    data = []
    for row in rows:
        # Get the time from the <th> element
        time = row.xpath('.//th//text()')
        time = ' '.join(time).strip()
        
        # Get the data from <td> elements
        values = row.xpath('.//td/text()')
        cleaned_values = [value.strip() for value in values]
        
        # Combine time with the rest of the data
        row_data = [time] + cleaned_values
        data.append(row_data)

    # Create a DataFrame
    # Assuming the first row of the data contains headers
    headers = ['time','WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES', 'PTDY', 'ATMP', 'WTMP', 'DEWP', 'SAL', 'VIS', 'TIDE']
    df = pd.DataFrame(data, columns=headers)
    setattr(df, 'name', 'buoy'+buoy_id[-3:])

    return df

def get_buoys(buoys, show=False):
    '''
    outputs a list of dfs to be used to predict
    '''
    dfxs = []
        
    for buoy_id in buoys:
        dfxs.append(scrape_buoy_data(buoy_id))
        
        # double check that I got what I thought I did.
    if show:
        for df in dfxs:
            print(df.head(), '\n\n\n')
    return dfxs     

#now walk through previous codebase and operate on this with the same recipe.
def prep_dfs(dfs, cols_to_keep):
    '''
    The idea here is to make each dataframe the same as the dataframes used in the training of the model. 
    '''
    prepped_dfs =[]
    for df in dfs:
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.round('30min')
        df.drop_duplicates(subset='time', keep='last', inplace=True)
        df.sort_values(by='time', ascending=True, inplace=True)
        df.replace(999, np.nan, inplace=True)
        df.replace(99, np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        prepped_df = df[cols_to_keep].copy()
        setattr(prepped_df, 'name', df.name)
        prepped_dfs.append(prepped_df)
        
    return prepped_dfs

def get_merged(df_list, oth=True):
    """ Merges two or more dataframes based on a common time column

    Parameters
    ----------
    df_list :   list
                A list of dataframes 
    oth :       boolean
                If True, only returns the records from the dataframe with 
                those where the minute value is 0 (on the hour)

    Returns
    -------
    df :        dataframe
                A merged dataframe with renamed columns
    """

    df_merged = df_list[0]
    for i in range(1, len(df_list)):
        df_merged = pd.merge(df_merged, df_list[i], on='time', suffixes=['_l', '_r'])
        df_merged.rename(columns={'WVHT_l': f'{df_list[i-1].name}_WVHT',
                                  'DPD_l': f'{df_list[i-1].name}_DPD',
                                  'MWD_l': f'{df_list[i-1].name}_MWD',
                                  'WVHT_r': f'{df_list[i].name}_WVHT',
                                  'DPD_r': f'{df_list[i].name}_DPD',
                                  'MWD_r': f'{df_list[i].name}_MWD'},
                     inplace=True)
        if i == len(df_list)-1:
            df_merged.rename(columns={'WVHT': f'{df_list[i].name}_WVHT',
                                  'DPD': f'{df_list[i].name}_DPD',
                                  'MWD': f'{df_list[i].name}_MWD'}, 
                     inplace=True)
    df = df_merged
    if oth:
        df = df[df['time'].dt.minute==0] #df_OnTheHour
    return df

def convert_columns_to_float(dfx):
    float_columns = [col for col in dfx.columns 
                     if col.lower() not in ['time'] and 'mwd' not in col.lower()]
    for column in float_columns:
        dfx[column] = pd.to_numeric(dfx[column], errors='coerce').astype(float)
    return dfx

# now I need to load the Xscaler, and carry out the encode and scale steps, but custom for this format where I'm already encoded in to cardinal bins (NNW etc)

def load_scalers(Xfilename='Xscaler202401.pkl', yfilename='yscaler000101205.pkl'):
    import joblib

    Xscaler = joblib.load(Xfilename)
    yscaler = joblib.load(yfilename)
    return Xscaler, yscaler

def encodeNscale(df, scaler):
    """
    Encode and scale a dataframe for ease-of-use in machine learning models
    
    df: X train dataframe)
    scaler: if there is a scaler provided, it will be used (for example, if operating on X_test), ow it will fit its own and return it
    
    Returns an encoded and scaled dataframe, as well as the scaling function if one is not provided
    
    20230705 just added funcitonality to specify which scaler to fit on X_train. this isn't perfect, but it works., just specify Xscaler
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Error: The dfy argument must be a Pandas DataFrame object")

    try:
        # encode the wave direction as categorical
        # either 4 or 16
        # df = encode_MWD(df, res=16)

        #   separate time, reset index so I can merge without nan
        df_time = pd.DataFrame(df['time'])
        df_time.reset_index(inplace=True)
        df_time.drop('index', axis=1,inplace=True)
            

        # separate categorical and numerical
        df_cat = df.select_dtypes(include = 'object') 
        df_num = df.select_dtypes(exclude = 'object').drop(df_time.columns, axis=1)
        num_columns = df_num.columns

        # encode dummies, reset index so I can merge without nan
        dummy_df_cat = pd.get_dummies(df_cat)
        dummy_df_cat.reset_index(inplace=True)
        dummy_df_cat.drop('index', axis=1,inplace=True)
        # scale numberical columns

        df_num_scaled = pd.DataFrame(scaler.transform(df_num), columns=num_columns)
        processed_df = pd.concat([df_num_scaled, dummy_df_cat, df_time], axis = 1)
        return processed_df
    
    except Exception as e:
        print(f'An error occurred while attempting to encode and scale the input dataframe: {e}')       

def align_dataframes(dfA, columns_list):
    # Get the set of columns that are in columns_list but not in dfA
    missing_cols = set(columns_list) - set(dfA.columns)
    
    # Add these columns to dfA, filled with zeros
    for col in missing_cols:
        dfA[col] = 0
    
    # Ensure that dfA has columns in the same order as columns_list
    dfA = dfA.reindex(columns=columns_list, fill_value=0)
    
    return dfA

def smooth_dataframe(df: pd.DataFrame, window_size: int=7, name: bool = False) -> pd.DataFrame:
    """
    Smooth numerical columns in a DataFrame using a rolling mean method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to smooth.
    window_size : int
        Window size to use for smoothing.
    name : bool, optional
        If True, append the window_size and "smoothed" to the DataFrame's name, by default False.

    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed numerical columns.

    Raises
    ------
    ValueError
        If window_size is less than 1.
    """

    if window_size < 1:
        raise ValueError("Window size must be at least 1.")

    # List all numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Compute rolling mean
    df_smoothed = df[numerical_cols].rolling(
        window=window_size, min_periods=1).mean()

    # If time column exists, add it to the smoothed dataframe
    if 'time' in df.columns:
        df_smoothed['time'] = df['time']

    # If name is true, append the window_size and "smoothed" to the dataframe's name
    if name:
        df_smoothed.name = f"{df.name}_{window_size}_smoothed"

    return df_smoothed

def get_data_contiguous_X_only(dfX, n_timesteps, lead_time, num_outputs, timestep_size=3600):
    """
    Extracts contiguous blocks of data from input dataframe and prepares them for machine learning modeling.
    This function walks through the dataset and makes one sample (X) at a time, and if the timesteps within it are contiguous, it is added to the data set.
    
    Args:
        dfX (pandas.DataFrame): The input dataframe containing the features. It should have a 'time' column and 
            other feature columns.
        n_timesteps (int): The number of consecutive timesteps to include in each input sample 
        lead_time (int): The number of timesteps between the last known value of X and the corresponding target value of y
        num_outputs (int): The number of consecutive timesteps to include in each target sample.
        timestep_size (int, optional): The duration of each timestep in seconds. Defaults to 3600 (seconds).
        
    Returns:
        numpy.ndarray: A 3-dimensional array representing the input samples, with shape 
                       (num_samples, n_timesteps, num_features).
    
    Raises:
        KeyError: If the 'time' column does not exist in the input dataframe.
        
    Notes:
        - This function assumes that the input dataframe has consistent timestamps and order.
        - Any missing or irregular timesteps in the input data will result in the corresponding samples being skipped.
        - Any nan or None values in the input data will be replaced with 0.
    """
    # Error checking
    if 'time' not in dfX.columns:
        raise KeyError("'time' column must exist in the input dataframe.")
    
    # Extract list of features, excluding "time"
    X_features = [col for col in dfX.columns if col != 'time']

    X = []
    max_index = len(dfX) - n_timesteps - lead_time - num_outputs + 1

    for i in tqdm(range(max_index), desc="Progress:", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        total_block = dfX.iloc[i : i + n_timesteps + lead_time + num_outputs]
        
        time_diffs = total_block['time'].diff()[1:].dt.total_seconds()
        if not all(time_diffs == timestep_size):
            continue

        X_block = dfX.iloc[i : i + n_timesteps][X_features]
        X.append(X_block.values)

    X = np.array(X, dtype=float)

    # Replacing any nan or None values with 0
    X = np.nan_to_num(X, nan=0.0)

    return X

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def get_future_time(dfx, lead_time):
    """
    Calculate the datetime value that is lead_time hours ahead of the latest time in dfx.

    Args:
    dfx (pandas.DataFrame): DataFrame containing a 'time' column with datetime values.
    lead_time (int): Number of hours to add to the latest time in the dataframe.

    Returns:
    datetime.datetime: The calculated future time.
    """
    # Check if 'time' column exists
    if 'time' not in dfx.columns:
        raise KeyError("'time' column is required in the dataframe.")

    # Find the latest time in the dataframe
    latest_time = dfx['time'].max()

    # Calculate the future time
    future_time = latest_time + timedelta(hours=lead_time)

    return future_time

if __name__ == "__main__":
    # Apply the function to your dataframe
    
    dfs = get_buoys(buoy_ids)
    
    prepped_dfs = prep_dfs(dfs, cols_to_keep)
    
    # for df in prepped_dfs:
    #     print(df.head(), df.info())
    dfx = get_merged(prepped_dfs)
    time_of_y = get_future_time(dfx, lead_time)
    # print(dfx.info())
    dfx = convert_columns_to_float(dfx)
    Xscaler, yscaler = load_scalers()
    
    dfx.columns = scaler_compatible_columns        
    dfx_processed = encodeNscale(dfx, Xscaler)
    dfx_processed_aligned = align_dataframes(dfx_processed, final_X_columns)
    dfx_processed__aligned_smoothed = smooth_dataframe(dfx_processed_aligned)
    X = get_data_contiguous_X_only(dfx_processed__aligned_smoothed, n_timesteps, lead_time=lead_time, num_outputs = num_outputs, timestep_size=3600)
    print(X)
    print(f'X is of shape {X.shape}')
    model = load_model(model_name)
    y_scaled = model.predict(X)
    y = yscaler.inverse_transform(y_scaled)
    
    print(f'model predicts buoy reading of {y}ft at {time_of_y}')