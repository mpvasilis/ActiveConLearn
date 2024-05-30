import os
import pandas as pd

# Define the directory containing the CSV files
directory = r"./results"

# Initialize an empty list to store DataFrames
dfs = []

# Loop through all the files in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file and does not end with 'constraints.txt'
    if not filename.endswith("constraints.txt") and not filename.endswith(".html"):
        try:
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            problem_name = filename.split('_solution')[0]
            df.insert(0, 'benchmark', problem_name)
            df['isOnlyActive'] = df['init_bias'] == 0
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Concatenate all DataFrames in the list into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Define the path for the output merged CSV file
output_path = os.path.join(directory, "merged_results.html")

# Save the merged DataFrame to a new CSV file
merged_df.to_html(output_path, index=False)
