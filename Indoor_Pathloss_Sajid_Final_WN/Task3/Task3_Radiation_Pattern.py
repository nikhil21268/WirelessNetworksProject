import pandas as pd
import numpy as np
from PIL import Image
import os

# Load main CSV file
data_df = pd.read_csv('Task3_output_file_eval.csv')

# Load each antenna's radiation pattern into a dictionary
radiation_patterns = {}
for i in range(1, 4):
    pattern_file = f'Tasks_Eval_Final/ICASSP_Test_Data/Test_Radiation_Patterns/Ant{i}_Pattern.csv'
    radiation_patterns[i] = pd.read_csv(pattern_file, header=None).iloc[:, 0].values  # 360 gain values for 360 degrees

import timeit

# Directory to save generated images
output_dir = 'Task3_New_Channel_changed_opt_eval_2'
os.makedirs(output_dir, exist_ok=True)


# Generate images
for idx, row in data_df.iterrows():
    t0 = timeit.default_timer()
    # Extract image information from the row
    image_name = row['Image']
    bw, bh = int(row['BW']), int(row['BH'])
    tx_y, tx_x = int(row['X']), int(row['Y'])
    azimuth = float(row['Azimuth'])  # Use the azimuth value from the row
    
    # Determine which antenna pattern to use based on the image name
    antenna_num = int(image_name.split("_")[1][3:])  # Extract antenna number from name, e.g., "Ant1"
    gain_pattern = radiation_patterns[antenna_num]  # 360-degree gain pattern for this antenna

    # Define standard deviation based on image size (adjust for desired spread)
    std_dev = max(bw, bh)/4

    x = np.linspace(0, bw, bw)
    y = np.linspace(0, bh, bh)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten(); yv = yv.flatten()
    distance = np.sqrt((xv - tx_x)**2 + (yv - tx_y)**2)     # Calculate distance from transmitter location
    angle = (np.degrees(np.arctan2(yv - tx_y, xv - tx_x)) + 360) % 360      # Calculate angle relative to transmitter
    adjusted_angle = np.int8((angle + azimuth + 360) % 360)     # Adjust angle by azimuth to account for antenna orientation, Ensure it remains within [0, 360)
    gain_linear = 10 ** (gain_pattern[adjusted_angle] / 10)     # Convert log-scale gain to linear
    intensity = np.exp(-(distance ** 2) / (2 * std_dev ** 2)) * gain_linear     # Calculate Gaussian decay centered at the transmitter, adjusted by gain
    image_array = np.reshape(intensity, (bh, bw))
    
    # print("Before scaling:")
    # print(f"Min: {image_array.min()}, Max: {image_array.max()}")

    # Scale the pixel values between 0 and 16
    image_array = 255 * (image_array / image_array.max())  # Normalize and scale
    # print("After scaling to 0-256 range:")
    # print(f"Min: {image_array.min()}, Max: {image_array.max()}")

  
    # Invert the grayscale image
    inverted_image_array = ((255 - image_array)/15).astype(np.uint8)# Invert pixel values
    # print("After inverting grayscale values and scaling to 0-16:")
    # print(f"Min: {inverted_image_array.min()}, Max: {inverted_image_array.max()}")


    image_name_without_extension = image_name.replace('.png', '')
    new_file_name = f"{image_name_without_extension}"
    
    # Save as a NumPy array
    npy_path = os.path.join(output_dir, f"{new_file_name}")
    np.save(npy_path, inverted_image_array)
    
    print(timeit.default_timer() - t0)