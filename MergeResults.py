import os
import pandas as pd
from tabulate import tabulate

# Directory containing the CSV files
directory = "./results"
directory_tex = "./results/tex"

if not os.path.exists(directory_tex):
    os.makedirs(directory_tex)
else:
    for filename in os.listdir(directory_tex):
        file_path = os.path.join(directory_tex, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# List to store dataframes
dfs = []

def shorten_problem_name(name):
    parts = name.split('_')
    if len(parts) > 3:
        return '_'.join(parts[:3])
    return name

def read_first_and_last_row(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
            header = lines[0]
            last_row = lines[-1]
            combined = header + last_row
            from io import StringIO
            df = pd.read_csv(StringIO(combined))
        else:
            df = pd.read_csv(filepath)  # In case there's only one line
    return df


# Read each file in the directory
for filename in os.listdir(directory):
    if not filename.endswith("constraints.txt") and not filename.endswith(".html"):
        try:
            filepath = os.path.join(directory, filename)
            df = read_first_and_last_row(filepath)
            problem_name = filename.split('_solution')[0]
            problem_name = shorten_problem_name(problem_name)  # Shorten the problem name
            df.insert(0, 'benchmark', problem_name)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicate types on the same benchmark
final_df = merged_df.drop_duplicates(subset=['benchmark', 'type'])
final_df = final_df.rename(columns={'learned_global_cstrs': 'CL_g'})


# Remove the specified columns
columns_to_remove = ['gen_q', 'fs_q', 'fc_q', 'avg|q|', 'conv', 'CL', 'top_lvl_q', 'avg_t', 'max_t']
final_df = final_df.drop(columns=columns_to_remove, errors='ignore')
final_df = final_df.sort_values(by=['benchmark', 'Tot_q'])

# Add the new column 'verified_global_constraints'
final_df['verified_gc'] = 0

# Set the number of global constraints based on the problem name
for idx, row in final_df.iterrows():
    if row['benchmark'] == '4sudoku' and row['genacq_q'] > 0:
        final_df.at[idx, 'verified_gc'] = 12
    elif row['benchmark'] in ['9sudoku', 'greaterThansudoku', 'jsudoku', 'sudoku_9x9'] and row['genacq_q'] > 0:
        final_df.at[idx, 'verified_gc'] = 27

# Create a separate LaTeX table for each unique problem name (benchmark)
for problem_name in final_df['benchmark'].unique():
    problem_df = final_df[final_df['benchmark'] == problem_name].copy()

    # Remove the benchmark column
    problem_df = problem_df.drop(columns=['benchmark'])

    # Ensure 'type' is the first column
    cols = problem_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('type')))
    problem_df = problem_df[cols]

    latex_table = tabulate(problem_df, headers='keys', tablefmt='latex', showindex=False)

    # Add a caption to the LaTeX table
    latex_caption = f"\\caption{{Results for {problem_name}}}\n"
    latex_table_with_caption = f"\\begin{{table}}[ht]\n{latex_caption}{latex_table}\n\\end{{table}}"

    latex_filename = f"{problem_name}_results.tex"
    latex_filepath = os.path.join(directory_tex, latex_filename)

    with open(latex_filepath, 'w') as f:
        f.write(latex_table_with_caption)

    print(f"LaTeX table for '{problem_name}' saved as '{latex_filename}'")
