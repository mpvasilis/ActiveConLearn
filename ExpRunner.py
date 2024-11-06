import csv
import os
import shutil
import subprocess
import yaml
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_count_cp_and_get_results(exp, output, name, input_file):
    # Set Count-CP directory
    input_file = os.path.abspath(os.path.join(input_directory, benchmark))
    #os.chdir(r"C:\Users\Balafas\Documents\GitHub\COUNT-CP")
    if not os.path.exists(os.path.join(output, name)):
        os.makedirs(os.path.join(output, name))
    results_dir = os.path.join(output, name)
    count_cp_command = [
        'C:/Users/Balafas/Documents/GitHub/cp-diverse-solutions/venv/Scripts/python.exe', r'C:\Users\Balafas\Documents\GitHub\COUNT-CP\cp2022_experiments.py',
        '--output', results_dir,
        '--name', name,
        '--input', input_file
    ]
    print(f"Running count-cp with command: {' '.join(count_cp_command)}")
    result = subprocess.run(count_cp_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running count-cp with command: {' '.join(count_cp_command)}\n{result.stderr}")
    else:
        print(f"Successfully ran count-cp with command: {' '.join(count_cp_command)}\nOutput:\n{result.stdout}")
        source_con_file = os.path.join('modules', 'benchmarks', name, name+'_con')
        if os.path.exists(source_con_file):
            shutil.copy(source_con_file, results_dir+'/_con')
            print(f"Successfully copied {source_con_file} to {results_dir}")
        else:
            print(f"Source _con file does not exist: {source_con_file}")
    return results_dir

def calculate_constraint_percentage(model_file_path):
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
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Constraint', 'Percentage'])
        for constraint, percentage in percentages.items():
            writer.writerow([constraint, percentage])
    print(f"Percentages written to {output_file}")


def run_jar_with_config(jar_path, config_path):
    java_command = [r"C:\Program Files\Eclipse Adoptium\jdk-21.0.2.13-hotspot\bin\java.exe", '-Xmx30g', '-jar', jar_path, config_path]
    print(java_command)
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


def run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints, use_count_cp):
    solution_set_path = os.path.join(input_directory, benchmark)
    experiment_name = os.path.normpath(os.path.basename(solution_set_path).replace('.json', ''))

    if "countcp" in config["type"]:
        print(f"Running {config["type"]} for", experiment_name)
        experiment_path = run_count_cp_and_get_results(experiment_name, output_directory, experiment_name, solution_set_path)
    else:
        print(f"Running {config["type"]} for", experiment_name)
        if os.path.exists(f"./modules/benchmarks/{experiment_name}"):
            print(f"Skipping {experiment_name} as it has already been run")
            experiment_path = "./modules/benchmarks/" + experiment_name
        else:
            experiment_name = run_passive_learning_with_jar(jar_path, solution_set_path, output_directory)
            experiment_path = "./modules/benchmarks/" + experiment_name
        model_file_path = os.path.join("./modules/benchmarks/", experiment_name, f"{experiment_name}_model")
        if os.path.exists(model_file_path):
            percentages = calculate_constraint_percentage(model_file_path)
            output_csv_path = os.path.join("./modules/benchmarks/", experiment_name,
                                           f"{experiment_name}_percentages.csv")
            write_percentages_to_csv(percentages, output_csv_path)
        else:
            print(f"Model file not found: {model_file_path}")

    if True:
        command = base_command.format(
            config["algo"],
            config["bench"],
            experiment_name,
            experiment_path,
            output_directory,
            str(use_constraints),
            str(config["onlyActive"]),
            str(config["emptyCL"]),
            str(config["type"])
        )
        print("Running command:", command)

        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Error running command: {command}\n{result.stderr}")
        else:
            print(f"Successfully ran command: {command}\nOutput:\n{result.stdout}")


if __name__ == "__main__":
    #print current working dir
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Run experiments in parallel or serial mode")
    parser.add_argument("--parallel", action="store_true", help="Run experiments in parallel mode")
    parser.add_argument("--use_count_cp", action="store_true", help="Use count-cp instead of the JAR")
    args = parser.parse_args()
    benchmarks = [
         # "4sudoku_solution.json",
              "9sudoku_solution.json",
        #     "jsudoku_solution.json",
        #       "murder_problem_solution.json",
        #       "nurse_rostering_solution.json",
        #  "examtt_simple_solution.json",
        #  "greaterThansudoku_solution.json",
        # "BIBD.json",
        # "job.json",
        # "Golomb.json",
        ]

    # benchmarks = [
    #    # "BIBD.json",
    #   # "job.json",
    #    #   "Golomb.json",
    #    #  "GraphColoring.json",
    #      #   "Interval.json",
    #      #   "Latin.json",
    #      # "Magic.json",
    #      #  "Nqueens2.json",
    #      #  "Nqueens3.json",
    #     #  "Schur.json",
    #     # "Schur2.json",
    #     #  "Warehouse.json",
    #     # "Warehouse2.json"
    #      "greaterThansudoku_solution.json",
    #         "4sudoku_solution.json",
    #          "9sudoku_solution.json",
    #        "examtt_advanced_solution.json",
    #         "examtt_simple_solution.json",
    #     # "greaterThansudoku_9x9_16b_diverse.json",
    #        #"greaterThansudoku_9x9_8b_diverse.json",
    #       #"greaterThansudoku_9x9_8b_nodiverse.json",
    #        "jsudoku_solution.json",
    #         "murder_problem_solution.json",
    #         "nurse_rostering_solution.json",
    #     #    "sudoku_9x9_diverse.json",
    #     #   "sudoku_9x9_nodiverse.json"
    # ]

    input_directory = "exps/instances/gts/"
    output_directory = "results"
    use_constraints = True

    jar_path = './phD.jar'

    base_command = "python main.py -a {} -b {} -qg pqgen -exp {} -i {} --output {} --useCon {} --onlyActive {} --emptyCL {} --type {}"

    configs = [
        # {"algo": "mquacq2-a", "bench": "countcp_al", "onlyActive": False, "emptyCL": False, "type": "countcp_al"},# countcp + al
       #  {"algo": "mquacq2-a", "bench": "countcp", "onlyActive": False, "emptyCL": True, "type": "countcp_al_genacq"},# countcp + al + genacq
           {"algo": "mquacq2-a", "bench": "vgc", "onlyActive": False, "emptyCL": True, "type": "pl_al_genacq"},# pl + al + genacq
          #{"algo": "mquacq2-a", "bench": "custom", "onlyActive": False, "emptyCL": False, "type": "pl_al"},#pl + al
         # {"algo": "mquacq2-a", "bench": "custom", "onlyActive": True, "emptyCL": False, "type": "al"},# al
       #  {"algo": "mquacq2-a", "bench": "genacq", "onlyActive": True, "emptyCL": False, "type": "genacq"}, #genacq
      # {"algo": "mquacq2-a", "bench": "mineask", "onlyActive": True, "emptyCL": False, "type": "mineask"} #mineask
    ]

    if args.parallel:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for benchmark in benchmarks:
                for config in configs:
                    futures.append(
                        executor.submit(run_experiment, config, benchmark, jar_path, input_directory, output_directory,
                                        use_constraints, args.use_count_cp))

            for future in as_completed(futures):
                future.result()
    else:
        for benchmark in benchmarks:
            for config in configs:
                run_experiment(config, benchmark, jar_path, input_directory, output_directory, use_constraints, args.use_count_cp)
