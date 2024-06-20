import os
import pandas as pd

directory = r"./results"

dfs = []

for filename in os.listdir(directory):
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

merged_df = pd.concat(dfs, ignore_index=True)

output_path = os.path.join(directory, "merged_results.html")

merged_df.to_html(output_path, index=False)