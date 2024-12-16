import csv
import os
import shutil
import subprocess
import yaml
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def calculate_constraint_percentage(model_file_path):
    constraint_counts = {}
    total_constraints = 0

    try:
        with open(model_file_path, 'r') as file:
            for line in file:
                if line.strip():
                    constraint_type = line.split()[0]
                    constraint_counts[constraint_type] = constraint_counts.get(constraint_type, 0) + 1
                    total_constraints += 1

        percentages = {k: (v / total_constraints) * 100 for k, v in constraint_counts.items()}
        return percentages
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_file_path}")
        return {}
    except Exception as e:
        logging.error(f"Error processing model file {model_file_path}: {e}")
        return {}

def write_percentages_to_csv(percentages, output_file):
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Constraint', 'Percentage'])
            for constraint, percentage in percentages.items():
                writer.writerow([constraint, percentage])
        logging.info(f"Percentages written to {output_file}")
    except Exception as e:
        logging.error(f"Failed to write percentages to CSV {output_file}: {e}")

def run_jar_with_config(jar_path, config_path, experiment_subdir):
    java_command = [
        r"C:\Program Files\Eclipse Adoptium\jdk-21.0.2.13-hotspot\bin\java.exe",
        '-Xmx10g',  # Adjusted memory allocation as per user modification
        '-jar',
        jar_path,
        config_path
    ]
    logging.info(f"Running JAR with command: {' '.join(java_command)}")

    try:
        # Initialize the subprocess with Popen
        with subprocess.Popen(java_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1) as process:
            # Stream stdout
            for line in process.stdout:
                print(line, end='')  # Print to console
                logging.info(line.strip())  # Log the output

            # Stream stderr
            for line in process.stderr:
                print(line, end='')  # Print to console
                logging.error(line.strip())  # Log the error

            # Wait for the subprocess to finish and get the return code
            return_code = process.wait()

        if return_code != 0:
            logging.error(f"Error running JAR with config {config_path}. Return code: {return_code}")
        else:
            logging.info(f"Successfully ran JAR with config {config_path}. Return code: {return_code}")

    except FileNotFoundError:
        logging.error(f"Java executable not found. Please check the path: {java_command[0]}")
    except Exception as e:
        logging.error(f"Unexpected error running JAR with config {config_path}: {e}")

def copy_txt_to_txt(base_name, base_name_with_sol, binary_cons_dir="binary_cons"):
    source_file = os.path.join(binary_cons_dir, f"{base_name}.txt")
    dest_file = os.path.join(binary_cons_dir, f"{base_name_with_sol}.txt")

    try:
        shutil.copyfile(source_file, dest_file)
        logging.info(f"Copied {source_file} to {dest_file}")
    except FileNotFoundError:
        logging.error(f"Source file not found: {source_file}")
    except Exception as e:
        logging.error(f"Failed to copy {source_file} to {dest_file}: {e}")


def generate_config_file(solution_set_path, output_directory, num_solutions):
    base_name = os.path.splitext(os.path.basename(solution_set_path))[0]
    base_name_with_sol = f"{base_name}_sol{num_solutions}"  # Include sols number in base_name

    config_data = {
        'problem': solution_set_path,
        'problemType': base_name_with_sol,  # Updated to include sols number
        'runName': base_name_with_sol,      # Updated to include sols number
        'activeLearning': True,
        'constraintsToCheck': [
            "allDifferent",
            "arithm"
        ],
        'decreasingLearning': False,
        'numberOfSolutionsForDecreasingLearning': num_solutions,
        'enableSolutionGeneratorForActiveLearning': True,
        'plotChart': False,
        'validateConstraints': False,
        'mQuack2MaxIterations': 1,
        'mQuack2SatisfyWithChoco': False,
        'runTestCases': False,
        'testCasesFile': "testcases/gts-testcases.json"
    }

    config_file_name = f"{base_name_with_sol}_config.yaml"  # Include sols number in config file name
    config_file_path = os.path.join(output_directory, config_file_name)
    try:
        with open(config_file_path, 'w') as file:
            yaml.dump(config_data, file, default_flow_style=False)
        logging.info(f"Config file for {base_name_with_sol} written to {config_file_path}")
        return base_name, base_name_with_sol, config_file_path
    except Exception as e:
        logging.error(f"Failed to write config file {config_file_path}: {e}")
        return base_name, base_name_with_sol, None

def run_passive_learning_with_jar(jar_path, solution_set_path, output_directory, num_solutions):
    basename, base_name_with_sol, config_path = generate_config_file(solution_set_path, output_directory, num_solutions)
    if config_path:
        copy_txt_to_txt(basename, base_name_with_sol)
        run_jar_with_config(jar_path, config_path, output_directory)
    else:
        logging.error(f"Skipping JAR execution for {basename} due to config file generation failure.")
    return basename

def run_experiment(benchmark, jar_path, input_directory, output_directory, num_solutions):
    solution_set_path = os.path.join(input_directory, benchmark)
    experiment_name = os.path.splitext(os.path.basename(solution_set_path))[0]

    # Create a subdirectory for each experiment based on the number of solutions
    experiment_subdir = os.path.join(output_directory, experiment_name, f"sol{num_solutions}")
    os.makedirs(experiment_subdir, exist_ok=True)

    logging.info(f"Starting experiment: Benchmark={benchmark}, Solutions={num_solutions}")

    run_passive_learning_with_jar(jar_path, solution_set_path, experiment_subdir, num_solutions)

    # Assuming the JAR outputs a model file named "{experiment_name}_model"
    # Since base_name now includes sols number, update model file naming accordingly
    model_file_path = os.path.join(experiment_subdir, f"{experiment_name}_model")
    if os.path.exists(model_file_path):
        percentages = calculate_constraint_percentage(model_file_path)
        if percentages:
            output_csv_path = os.path.join(experiment_subdir, f"{experiment_name}_percentages.csv")
            write_percentages_to_csv(percentages, output_csv_path)
    else:
        logging.warning(f"Model file not found: {model_file_path}")

if __name__ == "__main__":
    logging.info(f"Current working directory: {os.getcwd()}")

    parser = argparse.ArgumentParser(description="Run JAR experiments with varying numberOfSolutionsForDecreasingLearning")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel mode")
    parser.add_argument("--max-workers", type=int, default=2, help="Maximum number of parallel workers")
    args = parser.parse_args()

    benchmarks = [
        # "9sudoku_solution.json",
        # "greaterThansudoku_solution.json",
        # "4sudoku_solution.json",
        # "examtt_advanced_solution.json",
        #"examtt_simple_solution.json",
        # "jsudoku_solution.json",
         "murder_problem_solution.json",
        "nurse_rostering_solution.json",
    ]

    input_directory = "./"
    output_directory = "results"
    os.makedirs(output_directory, exist_ok=True)

    jar_path = './phD.jar'

    num_solutions_list = [2, 5, 10, 20, 50, 100, 200, 500]

    if args.parallel:
        max_workers = min(16, os.cpu_count() or 1)
        logging.info(f"Running experiments in parallel mode with {max_workers} workers.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for benchmark in benchmarks:
                for num_solutions in num_solutions_list:
                    futures.append(
                        executor.submit(
                            run_experiment,
                            benchmark,
                            jar_path,
                            input_directory,
                            output_directory,
                            num_solutions
                        )
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Experiment failed with exception: {e}")
    else:
        logging.info("Running experiments in serial mode.")
        for benchmark in benchmarks:
            for num_solutions in num_solutions_list:
                run_experiment(
                    benchmark,
                    jar_path,
                    input_directory,
                    output_directory,
                    num_solutions
                )
