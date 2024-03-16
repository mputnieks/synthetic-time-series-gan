import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from processing_pipeline_healthgan import Encoder, Decoder
from plotter import plot_glucose, plot_glucose_steps, plot_covariate_weekday
import numpy as np
import pandas as pd
import os

def drop_na(data, percentage):
    """Drops rows with more than given percentage NaN values"""
    return data.dropna(thresh=len(data.columns) * (1-percentage)) # drop rows

def add_week_day(data):
    """Adds day of the week to the dataframe based on the date"""
    data['date'] = pd.to_datetime(data['date'])
    data = data.assign(week_day = data['date'].dt.day_name()) # add day of the week
    return data

def interpolate_row(data, row_index):
    """Interpolates a row in the dataframe. Results only make sense if the row has no NaN values as first and last elements."""
    
    row = data.loc[row_index].to_numpy()
    mask = ~np.isnan(row)

    # Extract numeric values and their indices
    x_values = np.arange(len(row))[mask]
    y_values = row[mask]

    cs = CubicSpline(x_values, y_values, extrapolate='values')
    
    # Generate indices for the entire array
    all_indices = np.arange(len(row))

    # Perform interpolation for all indices
    interpolated_values = cs(all_indices)
    
    # Replace the original row with the interpolated values
    data.loc[row_index+1] = interpolated_values # TODO: remove + 1
    
    return data

def get_glucose(path):
    """Returns a dataframe with daily glucose values from the folder"""
    print("Processing glucose file: ", path)
    data = pd.read_csv(path, delim_whitespace=True, header=None, skiprows=3, usecols=[0, 1, 2, 3, 4])
    data = data.drop(columns=[0, 3]) # remove first and fourth column
    data.columns = ["date", "time", "gl"] # add headers
    
    # process time and gl
    data = data.assign(
        gl = pd.to_numeric(data['gl'].str.replace(',', '.'), errors='coerce'), # Convert the 'glucemia' column to float
        hour = pd.to_datetime(data['time'], format='%H:%M').dt.hour, # Create a new column for the hour
        half_hour = (pd.to_datetime(data['time'], format='%H:%M').dt.hour * 2) + (pd.to_datetime(data['time'], format='%H:%M').dt.minute // 30) # Create a new column for the half-hour
    )
    
    # Group by 'hour' or 'half_hour' and calculate the mean glucemia
    data = data.groupby(['date', 'half_hour'])['gl'].mean().unstack().round(2)
    data = data.reset_index()
    
    # date format is either 2017-05-19 or 19-5-2017, convert to a consistent format
    data['date'] = pd.to_datetime(data['date'], format='mixed').dt.date
    
    return data
    
def get_steps(path):
    """Returns a dataframe with daily steps from the folder"""
    data = pd.read_excel(path)
    
    # code to deal with all kinds of horrible inconsistencies in file structure
    if data.columns[0] != 'time':
        data = data.rename(columns={data.columns[0]: 'time'})
        for col in data.columns[1:]:
            try:
                new_col = pd.to_datetime(col, errors='raise').date()
                data.rename(columns={col: new_col}, inplace=True)
            except ValueError:
                data = data.drop(columns=[col])
                print("ValueError")
    
    # each date is represented by a column from 2nd to last. The first column is the time.
    # we need to transform this dataframe to a long format
    data = pd.melt(data, id_vars=['time'], var_name='date', value_name='steps')
    
    # process time
    data = data.assign(
        hour = pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour, # Create a new column for the hour
        half_hour = (pd.to_datetime(data['time'], format='%H:%M:%S').dt.hour * 2) + (pd.to_datetime(data['time'], format='%H:%M:%S').dt.minute // 30) # Create a new column for the half-hour
    )
    
    # Group by 'hour' or 'half_hour' and calculate the sum of steps
    data = data.groupby(['date', 'half_hour'])['steps'].sum().unstack()
    
    # reset index
    data = data.reset_index()
    
    # date format is either 2017-05-19 or 19-5-2017, convert to a consistent format
    data['date'] = pd.to_datetime(data['date'], format='mixed').dt.date
    
    return data

def merge_glucose_steps(glucose, steps):
    """Merges glucose and steps dataframes based on date, removes dates that are not in both dataframes,
    and adds a column with the day of the week for each date. Then removes the date."""
    data = glucose.merge(steps, on='date')
    data = add_week_day(data)
    data = data.drop(columns=['date']) # remove dates
    return data

def extract_data(compiled_cgm, compiled_steps, compiled_full, path):
    """Extracts the DIALECT data from the given path and compiles it into 3 CSV files: compiled_cgm, compiled_steps, and compiled_full."""
    # Remove previous compiled files if they exist
    if os.path.exists(compiled_cgm):
        os.remove(compiled_cgm)
    if os.path.exists(compiled_steps):
        os.remove(compiled_steps)
    if os.path.exists(compiled_full):
        os.remove(compiled_full)
    
    # Loop through each folder
    for dir in os.listdir(path):
        folder = os.path.join(path, dir)
        # get glucose file if its there
        file_name_gl = [file for file in os.listdir(folder) if file.endswith("glucose.txt")]
        if file_name_gl:
            glucose = get_glucose(os.path.join(folder, file_name_gl[0]))
            glucose = drop_na(glucose, 0.15)
            # Append to the compiled CSV
            glucose.to_csv(compiled_cgm, mode='a', index=False, header=not os.path.exists(compiled_cgm))
            # now get steps file if its there
            file_name_steps = [file for file in os.listdir(folder) if file.endswith("steps.xlsx")]
            if file_name_steps:
                steps = get_steps(os.path.join(folder, file_name_steps[0]))
                # Append to the compiled CSV
                steps.to_csv(compiled_steps, mode='a', index=False, header=not os.path.exists(compiled_steps))
                # Merge glucose and steps dataframes based on date
                # steps = drop_na(steps, 0)
                merged = merge_glucose_steps(glucose, steps)
                # Append to the compiled CSV
                merged.to_csv(compiled_full, mode='a', index=False, header=not os.path.exists(compiled_full))
            else:
                print("No steps file found in folder ", folder)
        else:
            print("No glucose file found in folder ", folder)
    print("\n--- --- FINISHED EXTRACTING --- ---\n")


if __name__ == '__main__':
    
    path = "data\dialect"
    path_processed = "data\training"
    
    compiled_cgm = os.path.join(path_processed, "compiled_cgm.csv")
    compiled_steps = os.path.join(path_processed, "compiled_steps.csv")
    compiled_full = os.path.join(path_processed, "compiled.csv")
    
    # !! uncomment when using for first time to extract DIALECT data !!
    #extract_data(compiled_cgm, compiled_steps, compiled_full, path)
    
    compiled_cgm_weekday = os.path.join(path_processed, "compiled_cgm_weekday.csv")
    compiled_full_weekday = os.path.join(path_processed, "compiled_full_weekday.csv")
    
    # data = pd.read_csv(compiled_full_weekday)
    # # move week_day from the being the last column to being the one after the glucose data
    # cols = list(data.columns)
    # cols = cols[:48] + [cols[-1]] + cols[48:-1]
    # data = data[cols]
    # print(data.head())
    
    # get glucose data and week_day
    # glucose = data.iloc[:, :49]
    # print(glucose.head())
    
    # # save glucose data
    # ccw_glucose = os.path.join(path_processed, "c_full_w_glucose.csv")
    # glucose.to_csv(ccw_glucose, index=False)
    
    # # run through the encoder
    # en = Encoder()
    # en.encode_train(ccw_glucose, fix_na_values=True)
    
    # # decode back
    # file = "c_full_w_glucose_sdv"
    # sd_file = os.path.join(path_processed, file+".csv")
    # sd_file_decoded = os.path.join(path_processed, file+"_decoded.csv")
    # de = Decoder()
    # de.decode(ccw_glucose, sd_file)
    
    # # now take the decoded file and replace the cgm data in the original dataframe
    # decoded = pd.read_csv(sd_file_decoded)
    # data.iloc[:, :49] = decoded
    
    # # save the new dataframe
    # data.to_csv(compiled_full_weekday, index=False)
    
    # data = add_week_day(data)
    # data = data.drop(columns=['date']) # remove dates
    # data.to_csv(compiled_full_weekday, index=False)
    
    # UNCOMMENT TO ENCODE
    # en = Encoder()
    # en.encode_train(compiled_full_weekday, fix_na_values=True)
    # ERROR - if every column is decimal, no limits file get created.
    # decoder requires a limits file to decode the data

    # UNCOMMENT TO DECODE
    file = "sd_8000_156_10_10"
    sd_file = "src/data/"+file+".csv"
    sd_file_decoded = "src/data/"+file+"_decoded.csv"
    # de = Decoder()
    # de.decode(compiled_full_weekday, sd_file)
    # if sd file has no covariate or other columns, but original one does, the decoder will add them empty to the decoded file
    
    # UNCOMMENT TO PLOT
    decoded = pd.read_csv(sd_file_decoded)
    # real = pd.read_csv(os.path.join(path_processed, "compiled_cgm_weekday_sdv_decoded.csv"))
    # plot_glucose(real, [1, 2, 42], "Decoded Real glucose time-series")
    # plot_glucose(decoded, [1, 2, 42], "Decoded SD glucose time-series")
    plot_covariate_weekday(decoded, "Decoded glucose time-series covariate", 7, 11)
    plot_covariate_weekday(decoded, "Decoded glucose time-series covariate", 0, 30)
    
    # UNCOMMENT TO PLOT GLUCOSE & STEPS
    rd = pd.read_csv(compiled_full_weekday)
    # plot_glucose_steps(decoded, 2, "Synthetic Data")
    # plot_glucose_steps(decoded, 42, "Synthetic Data")
    # plot_glucose_steps(rd, 42, "Real Data")
    # plot_glucose_steps(rd, 642, "Real Data")
    
    # UNCOMMENT TO PLOT ALL GLUCOSE & STEPS
    # for i in range(0, 100):
    #     plot_glucose_steps(decoded, i, "Synthetic Data")
    
    # create a new dataframe from real row number 321, and decoded row number 306
    # real_row = rd.iloc[321]
    # decoded_row = decoded.iloc[306]
    # print(real_row)
    # print(decoded_row)
    # new_df = pd.DataFrame([real_row, decoded_row])
    # print(new_df)
    # plot_glucose(new_df, [0, 1], "Most similar glucose time-series from real and synthetic data.", ["Real row 321", "Synthetic row 306"])