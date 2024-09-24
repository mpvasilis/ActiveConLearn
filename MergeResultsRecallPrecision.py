import os
import pandas as pd
from tabulate import tabulate

# Directory containing the CSV files
directory = "./results"
directory_tex = "./results/tex"
accuracy_file = "results/merged_accuracy_recall.csv"  # Path to the new accuracy file

# Ensure output directory exists
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
            df = pd.read_csv(filepath)
    return df

# Read each file in the directory
for filename in os.listdir(directory):
    if not filename.endswith("constraints.txt") and not filename.endswith("target.txt")  and not filename.endswith("recall.txt") and not filename.endswith(".csv") and not filename.endswith(".html"):
        try:
            filepath = os.path.join(directory, filename)
            df = read_first_and_last_row(filepath)
            problem_name = filename.split('_solution')[0]
            problem_name = shorten_problem_name(problem_name)
            df.insert(0, 'benchmark', problem_name)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")

# Concatenate all dataframes
merged_df = pd.concat(dfs, ignore_index=True)

# Remove duplicate types on the same benchmark
final_df = merged_df.drop_duplicates(subset=['benchmark', 'type'])
final_df = final_df.rename(columns={
    'learned_global_cstrs': 'C_L',
    'init_bias': 'Bias_i',
    'init_cl': 'CL_i',
    'Tot_q': 'Q_total',
    'genacq_q': 'Q_gen',
    'max_t': 'T_learn'
})


final_df = final_df.rename(columns={'type': 'Method'})


def extract_method(constraint_detail):
    """
    Extracts the method name from the Constraints_Detail string.
    Example:
        '4sudoku_solution_countcp_al_countcp_al_accuracy' -> 'countcp_al'
    """
    if isinstance(constraint_detail, str):
        if '_solution_' in constraint_detail and constraint_detail.endswith('_accuracy'):
            try:
                # Remove '_accuracy' suffix
                method_part = constraint_detail.replace('_accuracy', '')
                # Split by '_solution_' and take the second part
                method_part = method_part.split('_solution_')[1]
                # Split into parts
                parts = method_part.split('_')
                # Check for duplication (e.g., 'countcp_al_countcp_al')
                mid = len(parts) // 2
                if len(parts) % 2 == 0 and parts[:mid] == parts[mid:]:
                    return '_'.join(parts[:mid])
                if method_part == 'countcp_countcp_al_genacq':
                    method_part = 'countcp_al_genacq'
                if method_part == 'vgc_pl_al_genacq':
                    method_part = 'pl_al_genacq'
                method_part = method_part.replace('custom_', '')
                method_part = method_part.replace('vgc_', '')

                return method_part
            except (IndexError, ValueError):
                # In case the expected pattern is not found
                return 'Unknown_Method_Pattern'
    # Return 'Unknown' for non-string or unmatched patterns
    return 'Unknown'
accuracy_df = pd.read_csv(accuracy_file)
accuracy_df['benchmark'] = accuracy_df['Benchmark'].str.split('_solution').str[0]
accuracy_df['Method'] = accuracy_df['Benchmark'].apply(extract_method)
accuracy_df['benchmark'] = accuracy_df['benchmark'].apply(shorten_problem_name)
accuracy_df = accuracy_df.drop(columns=['Benchmark', 'Constraints_Detail'])
final_df['benchmark'] = final_df['benchmark'].apply(shorten_problem_name)
accuracy_df['benchmark'] = accuracy_df['benchmark'].apply(shorten_problem_name)
final_df = final_df.merge(
    accuracy_df[['benchmark', 'Method', 'Precision (%)', 'Recall (%)']],
    on=['benchmark', 'Method'],
    how='left'
)

# Save the DataFrame as HTML
final_df.to_html(os.path.join(directory, "merged_results.html"))

# Select only the columns that are needed for the LaTeX table
final_df = final_df[['benchmark', 'Method', 'Bias_i', 'CL_i', 'C_L', 'Q_total', 'Q_gen', 'T_learn', 'Precision (%)', 'Recall (%)']]

# Create a separate LaTeX table for each unique problem name (benchmark)
for problem_name in final_df['benchmark'].unique():
    problem_df = final_df[final_df['benchmark'] == problem_name].copy()
    problem_df = problem_df.drop(columns=['benchmark'])
    cols = problem_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Method')))
    problem_df = problem_df[cols]

    # Format the LaTeX table using tabulate
    latex_table = tabulate(problem_df, headers='keys', tablefmt='latex', showindex=False)

    # Add a caption to the LaTeX table
    latex_caption = f"\\caption{{Results for {problem_name}}}\n"
    latex_table_with_caption = f"\\begin{{table}}[ht]\n{latex_caption}{latex_table}\n\\end{{table}}"

    # Write the table to a .tex file
    latex_filename = f"{problem_name}_results.tex"
    latex_filepath = os.path.join(directory_tex, latex_filename)

    with open(latex_filepath, 'w') as f:
        f.write(latex_table_with_caption)

    print(f"LaTeX table for '{problem_name}' saved as '{latex_filename}'")
