import os
import subprocess
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    basename, config_path = generate_config_file(solution_set_path, output_directory)
    run_jar_with_config(jar_path, config_path)
    return basename


def run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints):
    solution_set_path = os.path.join(input_directory, benchmark)
    experiment_name = os.path.normpath(os.path.basename(solution_set_path).replace('.json', ''))

    if os.path.exists(f"./modules/benchmarks/{experiment_name}"):
        print(f"Skipping {experiment_name} as it has already been run")
    else:
        experiment_name = run_passive_learning_with_jar(jar_path, solution_set_path, output_directory)

    experiment_path = "./modules/benchmarks/" + experiment_name

    command = base_command.format(
        config["algo"],
        config["bench"],
        experiment_name,
        experiment_path,
        output_directory,
        str(use_constraints),
        str(config["onlyActive"]),
        str(config["emptyCL"])
    )
    print("Running command:", command)
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}\n{result.stderr}")
    else:
        print(f"Successfully ran command: {command}\nOutput:\n{result.stdout}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments in parallel or serial mode")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel mode")
    args = parser.parse_args()

    benchmarks = [
        #"4sudoku_solution.json",
        "9sudoku_solution.json",
        "greaterThansudoku_9x9_16b_diverse.json",
        "greaterThansudoku_9x9_24b_diverse.json",
        "greaterThansudoku_9x9_8b_diverse.json",
        "greaterThansudoku_9x9_8b_nodiverse.json",
        "sudoku_9x9_diverse.json",
        "sudoku_9x9_nodiverse.json"
    ]

    input_directory = "exps/instances/gts/"
    output_directory = "results"
    use_constraints = True

    jar_path = './phD.jar'

    base_command = "python main.py -a {} -b {} -qg pqgen -exp {} -i {} --output {} --useCon {} --onlyActive {} --emptyCL {}"

    configs = [
        {"algo": "mquacq2-a", "bench": "vgc", "onlyActive": True, "emptyCL": True}
    ]

    if args.parallel:
        with ThreadPoolExecutor() as executor:
            futures = []
            for benchmark in benchmarks:
                for config in configs:
                    futures.append(
                        executor.submit(run_experiment, config, benchmark, jar_path, input_directory, output_directory,
                                        use_constraints))

            for future in as_completed(futures):
                future.result()
    else:
        for benchmark in benchmarks:
            for config in configs:
                run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints)
