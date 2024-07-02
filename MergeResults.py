import os
import pandas as pd

# Directory containing the CSV files
directory = "./results"

# List to store dataframes
dfs = []

# Read each file in the directory
for filename in os.listdir(directory):
    if not filename.endswith("constraints.txt") and not filename.endswith(".html"):
        try:
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            problem_name = filename.split('_solution')[0]
            df.insert(0, 'benchmark', problem_name)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Remove instances where type is 'pl_al_genacq' and genacq_q is 0
# filtered_df = merged_df[~((merged_df['type'] == 'pl_al_genacq') & (merged_df['genacq_q'] == 0))]

# Remove duplicate types on the same benchmark
final_df = merged_df.drop_duplicates(subset=['benchmark', 'type'])

# Remove the specified columns
columns_to_remove = ['gen_q', 'fs_q', 'fc_q', 'avg|q|', 'conv', 'CL', 'top_lvl_q']
final_df = final_df.drop(columns=columns_to_remove, errors='ignore')
final_df = final_df.sort_values(by=['benchmark', 'Tot_q'])

output_path = os.path.join(directory, "merged_results.html")
final_df.to_html(output_path, index=False)
output_path = os.path.join(directory, "merged_results.csv")
final_df.to_csv(output_path, index=False)