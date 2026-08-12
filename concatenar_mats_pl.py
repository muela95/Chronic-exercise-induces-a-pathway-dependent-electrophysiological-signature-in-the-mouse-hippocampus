import os
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

# Define the folder path containing the .mat files
folder_path = 'D:/Laboratorio/Registros/Experimental-ejercicio/para_perforante_lateral/'
output_file = os.path.join(folder_path, 'pl6-sed.mat')

# Define which row to extract from each file using a dictionary
# Key: filename, Value: row index (0-based)

row_mapping = {
    #'Sed1.mat': 0,  
    #'Sed2.mat': 3,  
    'Sed3.mat': 1,    
    'Sed4_ndim_auto.mat': 3,
    'Sed6_ndim_auto.mat': 2,
    'Sed7.mat': 2,
    'sed1-mio.mat': 2,
    #'sed2-mio.mat': 0,
    'sed5-mio.mat': 5   
    
}

"""
row_mapping = {
    'Run1_ndim_auto.mat': 2,  # Parece tener estados, desde el ~360 al 680 parece estar mucho más potente LM
    'Run2.mat': 0,  # un poco de theta al principio, nada al final, debería dar igual si solo se coge del centro
    'Run4.mat': 2, 
    'Run5.mat': 0,  # Tiene algo de theta, no muchísimo, pero en todo el archivo, tendría que estar fuera de todo
    'Run6.mat': 2,
    'Run2-mio-invertido_perforante.mat': 3
    #'run6-mio.mat': 1 # Electrodo muy superficial, no sé si debería importar   
    
}
"""

# Number of data points to extract from the middle
num_points = 450250

# Initialize a list to store the extracted rows and windows count
extracted_rows = []
windows_info = {}
# Store scalers for potential later use
scalers = []

# Iterate through the files specified in the row_mapping dictionary
for file_name, row_index in row_mapping.items():
    file_path = os.path.join(folder_path, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: File {file_name} not found. Skipping.")
        continue
    
    try:
        # Load the .mat file
        data = scipy.io.loadmat(file_path)
        
        # Extract G.s
        g_s = data['G']['s'][0, 0]
        
        # Store the shape information for this file
        num_rows, num_cols = g_s.shape
        windows_info[file_name] = {'total_rows': num_rows, 'original_data_points': num_cols}
        
        # Check if the requested row exists
        if row_index < num_rows:
            # Extract the specified row
            row_data = g_s[row_index, :]
            
            # Calculate start and end indices to get the middle num_points
            if num_cols >= num_points:
                start_idx = (num_cols - num_points) // 2
                end_idx = start_idx + num_points
                middle_data = row_data[start_idx:end_idx]
                windows_info[file_name]['extracted_data_points'] = num_points
                windows_info[file_name]['start_index'] = start_idx
                windows_info[file_name]['end_index'] = end_idx
            else:
                # If not enough data points, take all
                middle_data = row_data
                windows_info[file_name]['extracted_data_points'] = len(middle_data)
                windows_info[file_name]['start_index'] = 0
                windows_info[file_name]['end_index'] = num_cols
                print(f"Warning: File {file_name} has fewer than {num_points} data points. Taking all {num_cols} available data points.")
            
            # Normalize the data in this row
            scaler = StandardScaler()
            # Reshape for StandardScaler which expects 2D input
            normalized_row = scaler.fit_transform(middle_data.reshape(-1, 1)).flatten()
            
            # Add a small random variation to prevent optimization into a 3D array
            # Trim a random small number of elements (1-5) from the end
            trim = random.randint(1, 5)
            if len(normalized_row) > trim:
                normalized_row = normalized_row[:-trim]
                
            scalers.append(scaler)
            
            # Append the normalized row
            extracted_rows.append(normalized_row)
            print(f"Processed file: {file_name}, extracted and normalized {len(normalized_row)} data points from row {row_index}")
        else:
            print(f"Error: Row {row_index} does not exist in {file_name}. File has {num_rows} rows.")
    
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Check if we have any extracted rows
if extracted_rows:
    # Create a dictionary to store our data
    G = {}
    
    # Convert each array to object type before creating the cell array
    extracted_rows = [np.array(row, dtype=object) for row in extracted_rows]
    
    # Convert the list of arrays to a cell array in MATLAB format
    # Explicitly force dtype=object to preserve cell array structure
    G['s'] = np.array([extracted_rows], dtype=object)
    
    # Print data types before saving for verification
    print("\nVerifying data structure before saving:")
    print(f"Type of G['s']: {type(G['s'])}")
    print(f"Shape of G['s']: {G['s'].shape}")
    for i, row in enumerate(extracted_rows):
        print(f"  Row {i} dtype: {row.dtype}, shape: {row.shape}")
    
    # Save the data to a .mat file using v7.3 format to preserve structure
    scipy.io.savemat(output_file, {'G': G}, do_compression=True)
    
    print(f"\nConcatenated data saved to {output_file}")
    print(f"Shape of G['s']: {G['s'].shape}")
    print(f"Number of rows in extracted data: {len(extracted_rows)}")
    
    # Print the length of each row for verification
    for i, row in enumerate(extracted_rows):
        print(f"Length of row {i}: {len(row)}")
    
    # Print the data point information for each animal
    print("\nData point information for each animal:")
    for file_name, info in windows_info.items():
        if 'extracted_data_points' in info:
            print(f"{file_name}: {info['extracted_data_points']} data points extracted from {info['original_data_points']} total (rows: {info['total_rows']})")
            print(f"  Extraction range: [{info['start_index']} to {info['end_index']}]")
        else:
            print(f"{file_name}: {info['original_data_points']} total data points (rows: {info['total_rows']})")
    
    # Calculate total number of data points (original and extracted)
    total_original = sum(info['original_data_points'] for info in windows_info.values())
    total_extracted = sum(info.get('extracted_data_points', 0) for info in windows_info.values())
    
    # Print total number of data points
    print(f"\nTotal number of data points across all files (original): {total_original}")
    print(f"Total number of data points extracted: {total_extracted}")
    
    # Print relative percentage of each file's data points to the total
    print("\nRelative percentage of original data points for each file:")
    for file_name, info in windows_info.items():
        percentage = (info['original_data_points'] / total_original) * 100
        print(f"{file_name}: {percentage:.2f}%")
    
    # Print relative percentage of each file's extracted data to the total extracted
    print("\nRelative percentage of extracted data points for each file:")
    for file_name, info in windows_info.items():
        if 'extracted_data_points' in info and total_extracted > 0:
            percentage = (info['extracted_data_points'] / total_extracted) * 100
            print(f"{file_name}: {percentage:.2f}%")
else:
    print("No rows were extracted. Cannot create the output file.")
    
    # Even if no rows were extracted, calculate statistics on found files
    if windows_info:
        # Calculate total number of data points
        total_original = sum(info['original_data_points'] for info in windows_info.values())
        
        # Print total number of data points
        print(f"\nTotal number of data points across all files (original): {total_original}")
        
        # Print relative percentage of each file's data points to the total
        if total_original > 0:
            print("\nRelative percentage of original data points for each file:")
            for file_name, info in windows_info.items():
                percentage = (info['original_data_points'] / total_original) * 100
                print(f"{file_name}: {percentage:.2f}%")
    else:
        print("\nTotal number of data points across all files: 0")
        print("\nRelative percentage of data points for each file: N/A (no files processed)")
