import csv
import os
import shutil
import subprocess
import yaml
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from cpmpy import *
from cpmpy import intvar
from sympy import Or


# --- Verification Functions ---

def parse_constraints(constraints_file):
    """
    Parses the constraints from a file.

    Args:
        constraints_file (str): Path to the constraints file.

    Returns:
        List of tuples: Each tuple contains (var1, operator, var2).
    """
    constraint_pattern = re.compile(r'\(var(\d+)\)\s*(==|!=|<=|>=|<|>)\s*\(var(\d+)\)')
    constraints = []
    with open(constraints_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments
            match = constraint_pattern.match(line)
            if match:
                var1, operator, var2 = match.groups()
                constraints.append( (f"var{var1}", operator, f"var{var2}") )
            else:
                raise ValueError(f"Invalid constraint format: {line}")
    return constraints

def read_json_solutions(json_file):
    """
    Reads the JSON solutions file.

    Args:
        json_file (str): Path to the JSON file.

    Returns:
        Tuple: (format_template, list of solutions)
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    format_template = data.get("formatTemplate", {})
    solutions = data.get("solutions", [])
    return format_template, solutions

def map_variables(format_template):
    """
    Maps variable names to grid positions and infers variable domains.

    Args:
        format_template (dict): The format template from JSON.

    Returns:
        Tuple: (variable_mapping, variable_domains, grid_shape)
    """
    array = format_template.get("array", [])
    num_rows = len(array)
    num_cols = len(array[0]) if num_rows > 0 else 0
    grid_shape = (num_rows, num_cols)

    variable_mapping = {}
    variable_domains = {}
    var_counter = 0

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            var_name = f"var{var_counter}"
            var_info = array[row_idx][col_idx]
            low = var_info.get("low", 0)
            high = var_info.get("high", 10)  # Default domain if not specified
            variable_mapping[var_name] = (row_idx, col_idx)
            variable_domains[var_name] = (low, high)
            var_counter += 1

    return variable_mapping, variable_domains, grid_shape

def create_cpmpy_model(variable_domains, constraints):
    """
    Creates a CPMpy model based on variable domains and constraints.

    Args:
        variable_domains (dict): Mapping of variable names to their (low, high) domains.
        constraints (list): List of constraints as tuples (var1, operator, var2).

    Returns:
        Tuple: (model, variables_dict)
    """
    variables = {}
    for var, (low, high) in variable_domains.items():
        variables[var] = intvar(low, high, name=var)

    model = Model()

    # Add constraints to the model
    for var1, operator, var2 in constraints:
        if operator == "!=":
            model += variables[var1] != variables[var2]
        elif operator == "==":
            model += variables[var1] == variables[var2]
        elif operator == "<=":
            model += variables[var1] <= variables[var2]
        elif operator == ">=":
            model += variables[var1] >= variables[var2]
        elif operator == "<":
            model += variables[var1] < variables[var2]
        elif operator == ">":
            model += variables[var1] > variables[var2]
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    return model, variables

def verify_solution(model, variables, solution):
    """
    Verifies if a given solution satisfies the model constraints.

    Args:
        model (cpmpy.Model): The CPMpy model with constraints.
        variables (dict): Mapping of variable names to CPMpy variables.
        solution (dict): Mapping of variable names to their assigned values.

    Returns:
        bool: True if the solution satisfies the constraints, False otherwise.
    """
    # Create a new model that includes the original constraints and assignments
    verification_model = Model(model.constraints)

    # Add constraints that variables must equal the solution values
    for var, value in solution.items():
        verification_model += (variables[var] == value)

    # Check satisfiability
    if verification_model.solve():
        return True
    else:
        return False

def extract_solution_variables(solution_array):
    """
    Extracts variable assignments from the solution array.

    Args:
        solution_array (list): Nested list representing the grid.

    Returns:
        dict: Mapping of variable names to their assigned values.
    """
    flat_list = [item for sublist in solution_array for item in sublist]
    solution_dict = {}
    for idx, value in enumerate(flat_list):
        solution_dict[f"var{idx}"] = value
    return solution_dict


def verify_constraints(constraints_file_target, json_solutions_learned, constraints_file_learned,
                       verification_output_path):
    """
    Verifies all solutions in the learned JSON file against the target constraints (Precision)
    and samples solutions from the target model to verify against the learned constraints (Recall).

    Args:
        constraints_file_target (str): Path to the target constraints file.
        json_solutions_learned (str): Path to the learned model's JSON solutions file.
        constraints_file_learned (str): Path to the learned constraints file.
        verification_output_path (str): Path to save the verification results.

    Returns:
        None
    """
    # --- Precision Calculation ---

    # Step 1: Parse Target Constraints
    try:
        constraints_target = parse_constraints(constraints_file_target)
        print(f"Parsed {len(constraints_target)} constraints from {constraints_file_target}")
    except Exception as e:
        print(f"Failed to parse target constraints from {constraints_file_target}: {e}")
        return

    # Step 2: Read Learned Solutions
    try:
        format_template_learned, solutions_learned = read_json_solutions(json_solutions_learned)
        print(f"Loaded {len(solutions_learned)} learned solutions from {json_solutions_learned}")
    except Exception as e:
        print(f"Failed to read learned solutions from {json_solutions_learned}: {e}")
        return

    if not solutions_learned:
        print("No learned solutions to verify.")
        return

    # Step 3: Map Variables for Target Model
    try:
        variable_mapping_target, variable_domains_target, grid_shape_target = map_variables(format_template_learned)
        print(f"Target variable mapping complete. Grid shape: {grid_shape_target}")
    except Exception as e:
        print(f"Failed to map target model variables: {e}")
        return

    # Step 4: Create CPMpy Model for Target Constraints
    try:
        model_target, variables_target = create_cpmpy_model(variable_domains_target, constraints_target)
        print("Target CPMpy model created.")
    except Exception as e:
        print(f"Failed to create target CPMpy model: {e}")
        return

    # Step 5: Verify Solutions for Precision
    verification_results_precision = []
    for idx, sol in enumerate(solutions_learned, 1):
        try:
            solution_array = sol.get("array", [])
            solution_dict = extract_solution_variables(solution_array)
            is_valid = verify_solution(model_target, variables_target, solution_dict)
            verification_results_precision.append({"solution_index": idx, "is_valid": is_valid})
            status = "Valid" if is_valid else "Invalid"
            print(f"Solution {idx} (Precision): {status}")
        except Exception as e:
            print(f"Failed to verify solution {idx} (Precision): {e}")
            verification_results_precision.append({"solution_index": idx, "is_valid": False, "error": str(e)})

    # Calculate Precision Metrics
    total_learned = len(verification_results_precision)
    valid_learned = sum(1 for result in verification_results_precision if result["is_valid"])
    precision = (valid_learned / total_learned) * 100 if total_learned > 0 else 0
    print(f"Precision: {precision:.2f}% ({valid_learned}/{total_learned})")

    # --- Recall Calculation ---

    # Step 6: Parse Learned Constraints
    try:
        constraints_learned = parse_constraints(constraints_file_learned)
        print(f"Parsed {len(constraints_learned)} constraints from {constraints_file_learned}")
    except Exception as e:
        print(f"Failed to parse learned constraints from {constraints_file_learned}: {e}")
        return

    # Step 7: Create CPMpy Model for Learned Constraints
    try:
        model_learned, variables_learned = create_cpmpy_model(variable_domains_target, constraints_learned)
        print("Learned CPMpy model created.")
    except Exception as e:
        print(f"Failed to create learned CPMpy model: {e}")
        return

    # Step 8: Generate 100 Solutions from Target Model (Recall)
    target_solutions_generated = []
    try:
        while len(target_solutions_generated) < 100:
            found = model_target.solve()
            if not found:
                print("No more solutions found in target model for Recall.")
                break
            # Extract variable assignments
            solution_dict = {var: variables_target[var].value() for var in variables_target}
            target_solutions_generated.append(solution_dict)
            print(f"Solution {len(target_solutions_generated)}: {solution_dict}")
        print(f"Generated {len(target_solutions_generated)} solutions from target model for Recall.")
    except Exception as e:
        print(f"Error during solution generation for Recall: {e}")

    # Step 9: Verify Generated Target Solutions against Learned Constraints (Recall)
    verification_results_recall = []
    for idx, sol in enumerate(target_solutions_generated, 1):
        try:
            is_valid = verify_solution(model_learned, variables_learned, sol)
            verification_results_recall.append({"solution_index": idx, "is_valid": is_valid})
            status = "Valid" if is_valid else "Invalid"
            print(f"Solution {idx} (Recall): {status}")
        except Exception as e:
            print(f"Failed to verify solution {idx} (Recall): {e}")
            verification_results_recall.append({"solution_index": idx, "is_valid": False, "error": str(e)})

    # Calculate Recall Metrics
    total_target = len(verification_results_recall)
    valid_target = sum(1 for result in verification_results_recall if result["is_valid"])
    recall = (valid_target / total_target) * 100 if total_target > 0 else 0
    print(f"Recall: {recall:.2f}% ({valid_target}/{total_target})")

    # --- Save Verification Results ---
    try:
        with open(verification_output_path, 'w') as f:
            f.write(f"Precision:\n")
            f.write(f"Total Learned Solutions: {total_learned}\n")
            f.write(f"Valid Learned Solutions: {valid_learned}\n")
            f.write(f"Accuracy (Precision): {precision:.2f}%\n\n")
            f.write(f"Recall:\n")
            f.write(f"Total Target Solutions: {total_target}\n")
            f.write(f"Valid Target Solutions: {valid_target}\n")
            f.write(f"Recall: {recall:.2f}%\n")
        print(f"Verification metrics saved to {verification_output_path}")
    except Exception as e:
        print(f"Failed to save verification results to {verification_output_path}: {e}")


# --- Existing Benchmark Runner Functions ---

def run_count_cp_and_get_results(exp, output, name, input_file):
    """
    Runs the COUNT-CP experiments and handles output files.

    Args:
        exp (str): Experiment identifier.
        output (str): Output directory path.
        name (str): Name of the experiment.
        input_file (str): Path to the input file.

    Returns:
        str: Path to the results directory.
    """
    # Set Count-CP directory
    input_file = os.path.abspath(os.path.join(input_directory, exp))
    if not os.path.exists(os.path.join(output, name)):
        os.makedirs(os.path.join(output, name))
    results_dir = os.path.join(output, name)
    count_cp_command = [
        'C:/Users/Balafas/Documents/GitHub/cp-diverse-solutions/venv/Scripts/python.exe',
        r'C:\Users\Balafas\Documents\GitHub\COUNT-CP\cp2022_experiments.py',
        '--output', results_dir,
        '--name', name,
        '--input', input_file
    ]
    result = subprocess.run(count_cp_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running count-cp with command: {' '.join(count_cp_command)}\n{result.stderr}")
    else:
        print(f"Successfully ran count-cp with command: {' '.join(count_cp_command)}\nOutput:\n{result.stdout}")
        source_con_file = os.path.join('modules', 'benchmarks', name, name+'_con')
        if os.path.exists(source_con_file):
            shutil.copy(source_con_file, os.path.join(results_dir, '_constraints.txt'))
            print(f"Successfully copied {source_con_file} to {results_dir}")
        else:
            print(f"Source _con file does not exist: {source_con_file}")
    return results_dir

def calculate_constraint_percentage(model_file_path):
    """
    Calculates the percentage of each constraint type in the model file.

    Args:
        model_file_path (str): Path to the model file.

    Returns:
        dict: Dictionary mapping constraint types to their percentage.
    """
    constraint_counts = {}
    total_constraints = 0

    with open(model_file_path, 'r') as file:
        for line in file:
            if line.strip():
                constraint_type = line.split()[0]
                constraint_counts[constraint_type] = constraint_counts.get(constraint_type, 0) + 1
                total_constraints += 1

    percentages = {k: (v / total_constraints) * 100 for k, v in constraint_counts.items()}
    return percentages

def write_percentages_to_csv(percentages, output_file):
    """
    Writes the constraint percentages to a CSV file.

    Args:
        percentages (dict): Dictionary of constraint percentages.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Constraint', 'Percentage'])
        for constraint, percentage in percentages.items():
            writer.writerow([constraint, percentage])
    print(f"Percentages written to {output_file}")

def run_jar_with_config(jar_path, config_path):
    """
    Runs a Java JAR with the specified configuration.

    Args:
        jar_path (str): Path to the JAR file.
        config_path (str): Path to the configuration YAML file.

    Returns:
        None
    """
    java_command = [
        r"C:\Program Files\Eclipse Adoptium\jdk-21.0.2.13-hotspot\bin\java.exe",
        '-Xmx30g',
        '-jar', jar_path,
        config_path
    ]
    print("Running Java command:", " ".join(java_command))
    result = subprocess.run(java_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running jar with config {config_path}: {result.stderr}")
    else:
        print(f"Successfully ran jar with config {config_path}\nOutput:\n{result.stdout}")

def generate_config_file(solution_set_path, output_directory):
    """
    Generates a YAML configuration file for the experiment.

    Args:
        solution_set_path (str): Path to the solution set JSON file.
        output_directory (str): Directory to save the configuration file.

    Returns:
        tuple: (base_name, config_file_path)
    """
    base_name = os.path.splitext(os.path.basename(solution_set_path))[0]
    config_data = {
        'problem': solution_set_path,
        'problemType': base_name,
        'runName': base_name,
        'activeLearning': True,
        'constraintsToCheck': [
            "allDifferent",
            "count",
            "sum",
            "arithm"
        ],
        'decreasingLearning': False,
        'numberOfSolutionsForDecreasingLearning': 2,
        'enableSolutionGeneratorForActiveLearning': True,
        'plotChart': False,
        'validateConstraints': False,
        'mQuack2MaxIterations': 1,
        'mQuack2SatisfyWithChoco': False,
        'runTestCases': False,
        'testCasesFile': "testcases/gts-testcases.json"
    }

    config_file_path = os.path.join(output_directory, f"{base_name}_config.yaml")
    with open(config_file_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

    print(f"Config file for {base_name} has been written to {config_file_path}")
    return base_name, config_file_path

def run_passive_learning_with_jar(jar_path, solution_set_path, output_directory):
    """
    Runs passive learning using the Java JAR with the generated configuration.

    Args:
        jar_path (str): Path to the JAR file.
        solution_set_path (str): Path to the solution set JSON file.
        output_directory (str): Directory to save the configuration file.

    Returns:
        str: Base name of the experiment.
    """
    basename, config_path = generate_config_file(solution_set_path, output_directory)
    run_jar_with_config(jar_path, config_path)
    return basename

def run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints, use_count_cp, testsets_directory):
    """
    Runs an individual experiment, including verification of solutions.

    Args:
        config (dict): Configuration dictionary for the experiment.
        benchmark (str): Benchmark file name.
        jar_path (str): Path to the Java JAR file.
        input_directory (str): Directory containing input solution files.
        output_directory (str): Directory to save experiment results.
        use_constraints (bool): Flag to determine if constraints should be used for verification.
        use_count_cp (bool): Flag to determine if COUNT-CP should be used.
        testsets_directory (str): Directory containing test set solutions.

    Returns:
        None
    """
    solution_set_path = os.path.join(input_directory, benchmark)
    experiment_name = os.path.splitext(os.path.basename(solution_set_path))[0]

    if use_count_cp:
        experiment_path = run_count_cp_and_get_results(experiment_name, output_directory, experiment_name, solution_set_path)
    else:
        benchmark_dir = os.path.join("modules", "benchmarks", experiment_name)
        if os.path.exists(benchmark_dir):
            print(f"Skipping {experiment_name} as it has already been run")
            experiment_path = benchmark_dir
        else:
            experiment_name = run_passive_learning_with_jar(jar_path, solution_set_path, output_directory)
            experiment_path = os.path.join("modules", "benchmarks", experiment_name)
        model_file_path = os.path.join("modules", "benchmarks", experiment_name, f"{experiment_name}_model")
        if os.path.exists(model_file_path):
            percentages = calculate_constraint_percentage(model_file_path)
            output_csv_path = os.path.join("modules", "benchmarks", experiment_name, f"{experiment_name}_percentages.csv")
            write_percentages_to_csv(percentages, output_csv_path)
        else:
            print(f"Model file not found: {model_file_path}")

    # Verification Step
    if use_constraints:
        # Locate constraints files ending with "_constraints.txt" in experiment_path
        constraints_files = [f for f in os.listdir("results") if f.startswith(experiment_name) and f.endswith("_constraints.txt")]

        print(experiment_name)
        try:
            constraints_files = [f for f in os.listdir("results") if
                                 f.startswith(experiment_name) and f.endswith("_constraints.txt")]
            if not constraints_files:
                print(f"No constraints file ending with '_constraints.txt' found in {"results"}")
                constraints_files = []
            else:
                print(f"Found {len(constraints_files)} constraints file(s) in {"results"}: {constraints_files}")
        except Exception as e:
            print(f"Error accessing {"results"}: {e}")
            constraints_files = []

        json_solution_file = os.path.join(testsets_directory, benchmark)  # Adjusted to read from testsets
        print(f"JSON solutions file: {json_solution_file}")
        if not os.path.exists(json_solution_file):
            print(f"Solution file does not exist: {json_solution_file}")
            constraints_files = []  # Prevent further processing

        if constraints_files and os.path.exists(json_solution_file):
            for idx, constraints_filename in enumerate(constraints_files, 1):
                constraints_file = os.path.join("results", constraints_filename)
                print(f"\nVerifying with constraints file: {constraints_file}")
                # Define a unique output file for each constraints file
                base_constraints_name = constraints_filename.replace('_constraints.txt', '')

                # Define a unique output file name with 'accuracy_recall.txt'
                verification_output_file = os.path.join(
                    "results",
                    f"{base_constraints_name}_accuracy_recall.txt"
                )

                target = benchmark.replace('_solution.json', '_target.txt')
                target = os.path.join("results", target)
                print(target)
                verify_constraints(target, json_solution_file,constraints_file, verification_output_file)
        else:
            print(
                f"Constraints or solutions file missing for {experiment_name}. Constraints found: {constraints_files}, Solutions: {json_solution_file}")

    # Existing Command Execution
    # command = base_command.format(
    #     config["algo"],
    #     config["bench"],
    #     experiment_name,
    #     experiment_path,
    #     output_directory,
    #     str(use_constraints),
    #     str(config.get("onlyActive", False)),
    #     str(config.get("emptyCL", False)),
    #     str(config.get("type", ""))
    # )
    # print("Running command:", command)
    # result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # if result.returncode != 0:
    #     print(f"Error running command: {command}\n{result.stderr}")
    # else:
    #     print(f"Successfully ran command: {command}\nOutput:\n{result.stdout}")


def merge_verification_results(results_directory, merged_output_file):

    merged_data = []

    # Regular expressions to match the lines
    total_learned_re = re.compile(r'Total Learned Solutions:\s*(\d+)')
    valid_learned_re = re.compile(r'Valid Learned Solutions:\s*(\d+)')
    precision_re = re.compile(r'Accuracy \(Precision\):\s*([\d\.]+)%')

    total_target_re = re.compile(r'Total Target Solutions:\s*(\d+)')
    valid_target_re = re.compile(r'Valid Target Solutions:\s*(\d+)')
    recall_re = re.compile(r'Recall:\s*([\d\.]+)%')

    # Traverse the results directory
    for root, dirs, files in os.walk(results_directory):
        for file in files:
            if file.endswith("_recall.txt"):
                file_path = os.path.join(root, file)

                # Extract Benchmark and Constraints Detail from filename
                base_name = file[:-len("_recall.txt")]
                if '_constraints_' in base_name:
                    parts = base_name.split('_constraints_', 1)
                    benchmark_name = parts[0]
                    constraints_detail = parts[1]
                else:
                    benchmark_name = base_name
                    constraints_detail = "N/A"

                # Initialize variables to store extracted metrics
                total_learned = valid_learned = precision = 0.0
                total_target = valid_target = recall = 0.0

                # Read and parse the verification file
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            print(f"Reading line: {line}")  # Debug: Print the line being read

                            # Match Total Learned Solutions
                            match = total_learned_re.match(line)
                            if match:
                                total_learned = int(match.group(1))
                                continue

                            # Match Valid Learned Solutions
                            match = valid_learned_re.match(line)
                            if match:
                                valid_learned = int(match.group(1))
                                continue

                            # Match Accuracy (Precision)
                            match = precision_re.match(line)
                            if match:
                                precision = float(match.group(1))
                                continue

                            # Match Total Target Solutions
                            match = total_target_re.match(line)
                            if match:
                                total_target = int(match.group(1))
                                continue

                            # Match Valid Target Solutions
                            match = valid_target_re.match(line)
                            if match:
                                valid_target = int(match.group(1))
                                continue

                            # Match Recall
                            match = recall_re.match(line)
                            if match:
                                recall = float(match.group(1))
                                continue

                    # Append the extracted data to merged_data if any metrics were captured
                    if total_learned > 0 or total_target > 0:
                        merged_data.append({
                            "Benchmark": benchmark_name,
                            "Constraints_Detail": constraints_detail,
                            "Total_Learned_Solutions": total_learned,
                            "Valid_Learned_Solutions": valid_learned,
                            "Precision (%)": f"{precision:.2f}",
                            "Total_Target_Solutions": total_target,
                            "Valid_Target_Solutions": valid_target,
                            "Recall (%)": f"{recall:.2f}"
                        })
                        print(f"Parsed verification results from {file_path}")  # Debug: File parsed successfully
                    else:
                        print(f"No valid metrics found in {file_path}")  # Debug: No metrics found

                except Exception as e:
                    print(f"Failed to parse {file_path}: {e}")

    # Write merged data to CSV
    try:
        with open(merged_output_file, 'w', newline='') as csvfile:
            fieldnames = [
                "Benchmark",
                "Constraints_Detail",
                "Total_Learned_Solutions",
                "Valid_Learned_Solutions",
                "Precision (%)",
                "Total_Target_Solutions",
                "Valid_Target_Solutions",
                "Recall (%)"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in merged_data:
                writer.writerow(data)
        print(f"Merged verification results saved to {merged_output_file}")
    except Exception as e:
        print(f"Failed to write merged results to {merged_output_file}: {e}")
# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in parallel or serial mode with constraint verification")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel mode")
    parser.add_argument("--use_count_cp", action="store_true", help="Use count-cp instead of the JAR")
    parser.add_argument("--testsets_directory", type=str, default="testsets", help="Directory containing test set solutions")
    args = parser.parse_args()

    benchmarks = [
        "4sudoku_solution.json",
        "9sudoku_solution.json",
        "examtt_advanced_solution.json",
        "examtt_simple_solution.json",
        "greaterThansudoku_9x9_16b_diverse.json",
        "greaterThansudoku_9x9_24b_diverse.json",
        "greaterThansudoku_9x9_8b_diverse.json",
        "greaterThansudoku_9x9_8b_nodiverse.json",
        "jsudoku_solution.json",
        "murder_problem_solution.json",
        "nurse_rostering_solution.json",
        "sudoku_9x9_diverse.json",
        "sudoku_9x9_nodiverse.json"
    ]

    input_directory = "exps/instances/gts/"
    output_directory = "results"  # Updated to absolute path
    use_constraints = True


    merged_output_file = os.path.join(output_directory, "merged_accuracy_recall.csv")
    merge_verification_results(output_directory, merged_output_file)
    exit()


    jar_path = './phD.jar'

    base_command = "python main.py -a {} -b {} -qg pqgen -exp {} -i {} --output {} --useCon {} --onlyActive {} --emptyCL {} --type {}"

    configs = [
        # {"algo": "mquacq2-a", "bench": "countcp_only", "onlyActive": False, "emptyCL": False, "type": "countcp_only"}, # countcp only
        # {"algo": "mquacq2-a", "bench": "countcp_al", "onlyActive": False, "emptyCL": False, "type": "countcp_al"},# countcp + al
        #{"algo": "mquacq2-a", "bench": "countcp", "onlyActive": False, "emptyCL": True, "type": "countcp_al_genacq"},# countcp + al + genacq
        #  {"algo": "mquacq2-a", "bench": "vgc", "onlyActive": False, "emptyCL": True, "type": "pl_al_genacq"},# pl + al + genacq
        #   {"algo": "mquacq2-a", "bench": "custom", "onlyActive": False, "emptyCL": False, "type": "pl_al"},#pl + al
        #  {"algo": "mquacq2-a", "bench": "custom", "onlyActive": True, "emptyCL": False, "type": "al"},# al
        {"algo": "mquacq2-a", "bench": "genacq", "onlyActive": True, "emptyCL": False, "type": "genacq"}, #genacq
        # {"algo": "mquacq2-a", "bench": "mineask", "onlyActive": True, "emptyCL": False, "type": "mineask"} #mineask
    ]

    if args.parallel:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for benchmark in benchmarks:
                for config in configs:
                    futures.append(
                        executor.submit(run_experiment, config, benchmark, jar_path, input_directory, output_directory,
                                        use_constraints, args.use_count_cp, args.testsets_directory))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Experiment failed with exception: {e}")
    else:
        for benchmark in benchmarks:
            for config in configs:
                try:
                    run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints, args.use_count_cp, args.testsets_directory)
                except Exception as e:
                    print(f"Experiment {benchmark} with config {config} failed with exception: {e}")

    merged_output_file = os.path.join(output_directory, "merged_accuracy_recall.csv")
    merge_verification_results(output_directory, merged_output_file)

    # exit()

