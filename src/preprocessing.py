import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline

def load_data():
    zhaw_data = pd.read_csv('../data/datazhawfull.csv', skiprows=[1])
    zhaw_metadata = pd.read_csv('../data/datazhawfull_meta.csv', skiprows=[1])
    agroscope_data = pd.read_csv('../data/dataagro24_run1.csv', skiprows=[1])
    agroscope_metadata = pd.read_csv('../data/dataagro24_run1_meta.csv', skiprows=[1])
    return zhaw_data, zhaw_metadata, agroscope_data, agroscope_metadata

def merge_data(data, metadata, key='timestring'):
    data = pd.merge(data, metadata, on=key, how='outer')
    data[key] = pd.to_datetime(data[key], dayfirst=True, format="%d.%m.%Y %H:%M:%S.%f")
    #data[key] = pd.to_datetime(data[key])
    data = data.sort_values(key).reset_index(drop=True)
    return data
    
def add_c02_col(data):
    # Find the pH column (case-insensitive)
    ph_col = next((col for col in data.columns if col.strip().lower() == 'ph'), None)

    # Initialize CO2 column
    co2_values = [0]

    for i in range(1, len(data)):
        co2 = 0
        curr_ph = data.iloc[i][ph_col]
        prev_ph = data.iloc[i - 1][ph_col]
        curr_time = data.iloc[i]['timestring']
        prev_time = data.iloc[i - 1]['timestring']
        # Check for valid pH values
        if pd.notnull(curr_ph) and pd.notnull(prev_ph):
            try:
                curr_ph = float(curr_ph)
                prev_ph = float(prev_ph)
                if curr_ph < prev_ph:
                    minutes = abs((curr_time - prev_time).total_seconds() / 60)
                    co2 = 2 * minutes if minutes < 5 else 0
            except ValueError:
                pass
        co2_values.append(co2)

    data['CO2'] = co2_values
    return data

def interpolate_dw(data, key):
    dw_mask = data[key].notnull()
    dw_times = data.loc[dw_mask, 'timestring'].astype(np.int64) // 10**9  # seconds
    dw_values = data.loc[dw_mask, key]

    if len(dw_times) > 1:
        cs_dw = CubicSpline(dw_times, dw_values)
        data['Dryweight'] = cs_dw(data['timestring'].astype(np.int64) // 10**9)
    else:
        data['Dryweight'] = np.nan
    
    data = data.drop(columns=[key])
    
    return data

def preprocess_data(interpolate_targets=False):
    zhaw_data, zhaw_metadata, agroscope_data, agroscope_metadata = load_data()
    
    zhaw_merged = merge_data(zhaw_data, zhaw_metadata)
    agroscope_merged = merge_data(agroscope_data, agroscope_metadata)
    
    agroscope_with_co2 = add_c02_col(agroscope_merged)
    
    zhaw_merged = merge_data(zhaw_data, zhaw_metadata)
    
    if interpolate_targets:
        zhaw_complete = interpolate_dw(zhaw_merged, 'Trockenmasse')
    else:
        zhaw_complete = zhaw_merged.rename(columns={'Trockenmasse':'Dryweight'})

    # zhaw_complete = interpolate_dw(zhaw_merged, 'Trockenmasse')
    # agroscope_complete = interpolate_dw(agroscope_with_co2, 'DW')
    
    zhaw_complete = zhaw_merged
    agroscope_complete = agroscope_with_co2
    
    #zhaw_complete = zhaw_complete.drop(['timestring', 'Person','Note', 'TM: Probevolumen', 'TM: Einwaage', 'End Result', 'SUM.OF.WATER', 'TURBIDITY'], axis=1)
    #agroscope_complete = agroscope_complete.drop(['timestring', 'Comments'], axis=1)
    
    zhaw_complete = zhaw_complete.drop(['Person','Note', 'TM: Probevolumen', 'TM: Einwaage', 'End Result', 'SUM.OF.WATER', 'TURBIDITY', 'THICKNESS.OF.ALGAE'], axis=1)
    agroscope_complete = agroscope_complete.drop(['Comments'], axis=1)
        
    return zhaw_complete, agroscope_complete