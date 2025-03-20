import os
import pandas as pd

# Path to folders containing the CSV files
building_details_path = 'Tasks_Eval_Final/ICASSP_Test_Data/Test_Building_Details'
positions_path = 'Tasks_Eval_Final/ICASSP_Test_Data/Test_Data_Positions'

# Dictionary to hold building dimension data
building_dimensions = {}

# Load Building Details CSV files
for filename in os.listdir(building_details_path):
    if filename.endswith('.csv'):
        building_id = filename.split('_')[1]  # Extract building ID (e.g., 'B1')
        df = pd.read_csv(os.path.join(building_details_path, filename))
        building_dimensions[building_id] = df.iloc[0].to_dict()  # Assuming 1 row per file

# Function to get the dimension data for a building
def get_building_dimensions(building_id):
    dims = building_dimensions.get(building_id, {})
    return dims.get('W', 0), dims.get('H', 0)  # Return width and height

# Initialize an empty list to store the final rows for the new CSV
output_data = []

# Load Positions CSV files and generate rows based on the image naming convention
for filename in os.listdir(positions_path):
    if filename.endswith('.csv'):
        parts = filename.replace('.csv', '').split('_')
        building_id = parts[1]
        antenna = parts[2]
        frequency = parts[3]

        # Load the position data for this building/antenna/frequency combination
        df_positions = pd.read_csv(os.path.join(positions_path, filename))
        
        # Determine sample range based on antenna type
        sample_range = range(50) if antenna == 'Ant1' else range(80)
        
        for sample_num in sample_range:
            # Get position and azimuth for this sample
            position = df_positions.iloc[sample_num]
            X, Y, Azimuth = position['X'], position['Y'], position['Azimuth']
            
            # Generate image name based on the pattern
            image_name = f"{building_id}_{antenna}_{frequency}_S{sample_num}.png"
            
            # Get building width and height
            BW, BH = get_building_dimensions(building_id)
            
            # Append data to output list
            output_data.append([image_name, BW, BH, frequency, Azimuth, X, Y])


import re

# Helper functions to extract components from the image name
def get_building_number(image_name):
    match = re.search(r'B(\d+)', image_name)
    return int(match.group(1)) if match else 0

def get_antenna_number(image_name):
    match = re.search(r'Ant(\d+)', image_name)
    return int(match.group(1)) if match else 0

def get_frequency_number(image_name):
    match = re.search(r'f(\d+)', image_name)
    return int(match.group(1)) if match else 0

def get_sample_number(image_name):
    match = re.search(r'S(\d+)', image_name)
    return int(match.group(1)) if match else 0

# Create DataFrame from output data
output_df = pd.DataFrame(output_data, columns=['Image', 'BW', 'BH', 'Frequency', 'Azimuth', 'X', 'Y'])

# Add temporary columns for sorting
output_df['Building_Number'] = output_df['Image'].apply(get_building_number)
output_df['Antenna_Number'] = output_df['Image'].apply(get_antenna_number)
output_df['Frequency_Number'] = output_df['Image'].apply(get_frequency_number)
output_df['Sample_Number'] = output_df['Image'].apply(get_sample_number)

# Sort the DataFrame by building, antenna, frequency, and sample number
output_df = output_df.sort_values(
    by=['Building_Number', 'Antenna_Number', 'Frequency_Number', 'Sample_Number']
).reset_index(drop=True)

# Drop the temporary columns used for sorting
output_df = output_df.drop(columns=['Building_Number', 'Antenna_Number', 'Frequency_Number', 'Sample_Number'])



# Save sorted DataFrame to CSV
output_df.to_csv('Task3_output_file_eval.csv', index=False)

print("CSV file created and sorted by Image column successfully.")