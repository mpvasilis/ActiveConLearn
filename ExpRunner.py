import os
import subprocess
import yaml


def run_jar_with_config(jar_path, config_path):
    java_command = ['java', '-Xmx30g', '-jar', jar_path, config_path]
    result = subprocess.run(java_command, capture_output=True, text=True)
    print(" ".join(['java', '-jar', jar_path, config_path]))
    if result.returncode != 0:
        print(f"Error running jar with config {config_path}: {result.stderr}")
    else:
        print(f"Successfully ran jar with config {config_path}\nOutput:\n{result.stdout}")


def generate_config_file(solution_set_path, output_directory):
    base_name = os.path.normpath(os.path.basename(solution_set_path).replace('.json', ''))
    config_data = {
        'problem': solution_set_path,
        'problemType': base_name,
        'runName': base_name,
        'activeLearning': True,
        'constraintsToCheck': [
            "allDifferent",
            "arithm"
        ],
        'decreasingLearning': False,
        'numberOfSolutionsForDecreasingLearning': 0,
        'enableSolutionGeneratorForActiveLearning': True,
        'plotChart': False,
        'validateConstraints': True,
        'mQuack2MaxIterations': 1,
        'mQuack2SatisfyWithChoco': False,
        'runTestCases': False,
        'testCasesFile': "testcases/gts-testcases.json"
    }

    config_file_path = os.path.join(output_directory, f"{base_name}_config.yaml")
    with open(config_file_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False)

    print(f"Config file for {base_name} has been written to {config_file_path}")
    return config_file_path


def run_passive_learning_with_jar(jar_path, solution_set_path, output_directory):
    config_path = generate_config_file(solution_set_path, output_directory)
    run_jar_with_config(jar_path, config_path)


# List of benchmark problems
benchmarks = [
    "greaterThansudoku_9x9_16b_diverse.json",
   # "greaterThansudoku_9x9_24b_diverse.json",
    #"greaterThansudoku_9x9_8b_diverse.json"
]

# Experiment configuration parameters
experiment_name = "b03_21_00_40_27_greaterThanSudoku_b10__diverse_diversity_Hamming_10_1_sols_100"
input_directory = "exps/instances/gts/"
output_directory = "results"
use_constraints = True

# Path to the jar file
jar_path = './phD.jar'

# Base command template
base_command = "python main.py -a {} -b {} -qg pqgen -exp {} -i {} --output {} --useCon {}"

# Run configurations
configs = [
    {"algo": "mquacq2-a", "bench": "vgc"},
    {"algo": "mquacq2-a", "bench": "custom"}
]

for benchmark in benchmarks:
    # Run passive learning with the jar
    solution_set_path = os.path.join(input_directory, benchmark)
    run_passive_learning_with_jar(jar_path, solution_set_path, output_directory)

    for config in configs:
        command = base_command.format(
            config["algo"],
            config["bench"],
            experiment_name,
            solution_set_path,
            output_directory,
            str(use_constraints)
        )
        print("Running command:", command)
        subprocess.run(command, shell=True)
