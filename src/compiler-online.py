"""Compiles the online Colas 2019 data into a single CSV file."""
import pandas as pd
import os

os.chdir("data\online") # Set the working directory to the data folder
# if previous compiled files exist, delete them
if os.path.exists("compiled.csv"):
    os.remove("compiled.csv")
clinical_data = pd.read_csv("clinical_data.txt", sep=" ")
files = [file for file in os.listdir() if file.endswith(".csv")] # get all CSV files in this folder

# Loop through each file
for file in files:
    print(file)                             # Track progress
    curr = pd.read_csv(file)                # Read CSV file
    id_val = file.split(".")[0].split()[1]  # Extract id from CSV filename
    
    # Extract the patient data based on id from clinical_data
    # clinal_data.txt format:
    # "id" "gender" "age"   "BMI"   "glycaemia" "HbA1c" "follow.up" "T2DM"
    # "1"   1       77      25.4    106         6.3     413         FALSE
    # ...
    patient_data = clinical_data[clinical_data['id'] == int(id_val)]
    
    # Select relevant columns and process time
    curr = curr.assign(
        id = id_val,
        gl = curr['glucemia'].astype(float),
        hour = pd.to_datetime(curr['hora'], format='%H:%M:%S').dt.hour, # Create a new column for the hour
        half_hour = (pd.to_datetime(curr['hora'], format='%H:%M:%S').dt.hour * 2) + (pd.to_datetime(curr['hora'], format='%H:%M:%S').dt.minute // 30) # Create a new column for the half-hour
    )
    
    # Detect day change, split data into multiple days
    curr['day_change'] = curr['hour'].diff() < 0
    curr['day'] = curr['day_change'].cumsum()
    curr = curr.drop(columns=['day_change'])
    days = [group for _, group in curr.groupby('day')]  # Creates a list of single day dataframes
    
    for day in days:
        
        curr = day
        curr = curr.iloc[:288] # for now crop to a single day
        
        if curr.shape[0] < 288:
            print("data < 288, skipping...")
            continue
        
        # TODO: Interpolate missing values
        # For now: Check if the data has any missing values, and if so, move to the next file
        if curr.isnull().values.any():
            print("missing values, skipping...")
            continue
        
        # Group by 'id' and 'hour' and calculate the mean glucemia for each group
        #curr = curr.groupby(['id', 'hour'])['gl'].mean().unstack().round(2)
        curr = curr.groupby(['id', 'half_hour'])['gl'].mean().unstack().round(2)

        # Rename the columns to represent the hours, then add the 3 covariates
        curr.columns = [f'half_h_{half_hour}' for half_hour in curr.columns]
        #curr.columns = [f'h{hour}' for hour in curr.columns]
        curr = curr.assign(
            age = patient_data['age'].values[0],
            BMI = patient_data['BMI'].values[0]
        )
        
        # Append to the compiled CSV
        output_file = "compiled.csv"
        curr.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

# Split the data into test and train sets (80/20), and save them as separate files
data = pd.read_csv("compiled.csv")
os.chdir("..\training")
# if previous compiled files exist, delete them
if os.path.exists("compiled-online.csv"):
    os.remove("compiled-online.csv")
# save the data
print("\n Saving data...")
print(data.shape)
print(data.head())
data.to_csv("compiled-online.csv", index=False)