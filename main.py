import json
import os
import argparse
import subprocess
import random
from itertools import combinations

import cpmpy as cp
from cpmpy import *
from cpmpy.transformations.normalize import toplevel_list
import pandas as pd
import yaml


from QuAcq import QuAcq
from MQuAcq import MQuAcq
from MQuAcq2 import MQuAcq2
from GrowAcq import GrowAcq
from utils import *
from BenchmarksGenerator import (
    _construct_magic_square,
    _construct_schurs_lemma,
    _construct_golomb_ruler,
    _construct_BIBD,
    _construct_greaterThanSudoku,
    construct_job_shop_scheduling_problem
)

jar_path = './phD.jar'
output_directory = './results'
def parse_args():
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # Parsing algorithm
    parser.add_argument("-a", "--algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a", "growacq", "genacq", "mineask"],
                        required=True,
                        help="The name of the algorithm to use")
    # Parsing specific to GrowAcq
    parser.add_argument("-ia", "--inner-algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a"],
                        required=False,
                        help="Only relevant when the chosen algorithm is GrowAcq - "
                             "the name of the inner algorithm to use")

    parser.add_argument( "--type", type=str,
                        required=False, default="",
                        help="Type of the experiment")

    # Parsing query generation method
    parser.add_argument("-qg", "--query-generation", type=str, choices=["baseline", "base", "tqgen", "pqgen"],
                        help="The version of the query generation method to use", default="pqgen")
    parser.add_argument("-obj", "--objective", type=str, choices=["max", "sol", "p", "prob", "proba"],
                        help="The objective function used in query generation", default="max")
    # Parsing findscope method
    parser.add_argument("-fs", "--findscope", type=int, choices=[1, 2], required=False,
                        help="The version of the findscope method to use", default=2)
    # Parsing findc method
    parser.add_argument("-fc", "--findc", type=int, choices=[1, 2], required=False,
                        help="The version of the findc method to use", default=1)

    # Parsing time limit - will default to None if none is provided
    parser.add_argument("-t", "--time-limit", type=float, help="An optional time limit")

    # Parsing benchmark
    parser.add_argument("-b", "--benchmark", type=str, required=True,
                        choices=["9sudoku", "4sudoku", "jsudoku", "random122", "random495", "new_random",
                                 "golomb8", "murder", "job_shop_scheduling",
                                 "exam_timetabling", "exam_timetabling_simple", "exam_timetabling_adv",
                                 "exam_timetabling_advanced", "nurse_rostering", "nurse_rostering_simple",
                                 "nurse_rostering_advanced", "nurse_rostering_adv", "custom", "vgc", "genacq", "mineask", "countcp", "countcp_al", "countcp_only"],
                        help="The name of the benchmark to use")

    parser.add_argument("-exp", "--experiment", type=str, required=False,
                    help="Experiment name for custom benchmark")
    parser.add_argument("-i", "--input", type=str, required=False,
                        help="File path of input files (_var, _model, _con, _cl, _bias) for custom problems")
    parser.add_argument("-o", "--output", type=str, required=False,
                        help="Output directory")
    parser.add_argument("-ulm", "--use_learned_model", type=bool, required=False,
                        help="Use the Passive Learning model as CT")
    parser.add_argument("-con", "--useCon", type=bool, required=False,
                        help="Use _con (fixed arity constraints) file as target model")
    parser.add_argument("-oa", "--onlyActive", type=str2bool, required=False,
                        help="Run a custom model with only active learning - don't use the Passive Learning CL and bias")
    parser.add_argument("-ecl", "--emptyCL", type=str2bool, required=False,
                        help="Run using empty CL")
    # Parsing specific to job-shop scheduling benchmark
    parser.add_argument("-nj", "--num-jobs", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the number of jobs")
    parser.add_argument("-nm", "--num-machines", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the number of machines")
    parser.add_argument("-hor", "--horizon", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the horizon")
    parser.add_argument("-s", "--seed", type=int, required=False,
                        help="Only relevant when the chosen benchmark is job-shop scheduling - the seed")

    # Parsing specific to nurse rostering benchmark
    parser.add_argument("-nspd", "--num-shifts-per-day", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of shifts per day")
    parser.add_argument("-ndfs", "--num-days-for-schedule", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of days for the schedule")
    parser.add_argument("-nn", "--num-nurses", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering - the number of nurses")
    parser.add_argument("-nps", "--nurses-per-shift", type=int, required=False,
                        help="Only relevant when the chosen benchmark is nurse rostering (advanced) - "
                             "the number of nurses per shift")

    # Parsing specific to exam timetabling benchmark
    parser.add_argument("-ns", "--num-semesters", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of semesters")
    parser.add_argument("-ncps", "--num-courses-per-semester", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of courses per semester")
    parser.add_argument("-nr", "--num-rooms", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of rooms")
    parser.add_argument("-ntpd", "--num-timeslots-per-day", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of timeslots per day")
    parser.add_argument("-ndfe", "--num-days-for-exams", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of days for exams")
    parser.add_argument("-np", "--num-professors", type=int, required=False,
                        help="Only relevant when the chosen benchmark is exam timetabling - "
                             "the number of professors")
    parser.add_argument("-pl", "--run-passive-learning", required=False,type=bool,  help="Run passive learning")
    parser.add_argument("-sols", "--solution-set-path", type=str, required=False, help="Path to the solution set JSON file")


    args = parser.parse_args()

    # Additional validity checks
    if args.algorithm == "growacq" and args.inner_algorithm is None:
        parser.error("When GrowAcq is chosen as main algorithm, an inner algorithm must be specified")
    if args.query_generation in ["baseline", "base"]:
        args.query_generation = "base"
    if args.objective in ["p", "prob", "proba"]:
        args.objective = "proba"
    if args.benchmark == "job_shop_scheduling" and \
            (args.num_jobs is None or args.num_machines is None or args.horizon is None or args.seed is None):
        parser.error("When job-shop-scheduling is chosen as benchmark, a number of jobs, a number of machines,"
                     "a horizon and a seed must be specified")
    if (args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple") and \
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day and a number of days for exams"
                     " must be specified")
    if (args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced") and \
            (args.num_semesters is None or args.num_courses_per_semester is None or args.num_rooms is None or
             args.num_timeslots_per_day is None or args.num_days_for_exams is None or args.num_professors is None):
        parser.error("When exam-timetabling is chosen as benchmark, a number of semesters, a number of courses per"
                     "semester, a number of rooms, a number of timeslots per day, a number of days for exams"
                     " and a number of professors must be specified")

    return args

def map_expression_to_constraints(expr_str, var1, var2, bounds):
    """
    Maps an expression string to CPMpy constraints based on the expression type.

    Args:
        expr_str (str): The expression string (e.g., 'x - y', 'x + y', 'Abs(x - y)').
        var1 (cpmpy.intvar): The first variable in the expression.
        var2 (cpmpy.intvar): The second variable in the expression.
        bounds (tuple): The lower and upper bounds for the expression.

    Returns:
        list: A list of CPMpy constraints.
    """
    constraints = []
    lb, ub = bounds

    if expr_str == "x - y":
        # Without specific bounds, default to x == y
        # Adjust as needed based on bounds
        if lb is None and ub is None:
            constraints.append(var1 == var2)
        else:
            if lb is not None:
                constraints.append(var1 - var2 >= lb)
            if ub is not None:
                constraints.append(var1 - var2 <= ub)

    elif expr_str == "x + y":
        # Without specific bounds, default to x + y >= 0
        if lb is None and ub is None:
            constraints.append(var1 + var2 >= 0)
        else:
            if lb is not None:
                constraints.append(var1 + var2 >= lb)
            if ub is not None:
                constraints.append(var1 + var2 <= ub)

    elif expr_str == "Abs(x - y)":
        # Without specific bounds, default to |x - y| <= 1
        # Implemented as two constraints: (x - y <= c) and (y - x <= c)
        # Adjust 'c' based on bounds
        if lb is None and ub is None:
            c = 1  # Default bound
            constraints.append(var1 - var2 <= c)
            constraints.append(var2 - var1 <= c)
        else:
            if lb is not None:
                # |x - y| >= lb translates to (x - y >= lb) OR (x - y <= -lb)
                # CPMpy does not support OR directly, so handle accordingly if needed
                # For simplicity, skip OR constraints or use auxiliary variables
                # Here, we'll just implement |x - y| >= lb as separate constraints
                constraints.append(var1 - var2 >= lb)
                constraints.append(var1 - var2 <= -lb)
            if ub is not None:
                # |x - y| <= ub translates to (x - y <= ub) AND (y - x <= ub)
                constraints.append(var1 - var2 <= ub)
                constraints.append(var2 - var1 <= ub)

    else:
        print(f"Unsupported expression: {expr_str}. No constraints applied.")

    return constraints


def apply_partitions_to_model(model, partitions, list_vars):
    for partition in partitions:
        expr = partition["Expression"]
        partition_indices = partition["Partition Indices"]
        bounds = partition["Bounds"]

        # Extract variable indices
        var_indices = [idx for tensor, idx in partition_indices]
        vars_in_partition = [list_vars[idx] for idx in var_indices]

        # Generate all unique unordered pairs (ALL_PAIRS)
        for var1, var2 in combinations(vars_in_partition, 2):
            constraints = map_expression_to_constraints(expr, var1, var2, bounds)
            if constraints:
                for constr in constraints:
                    model += constr

def construct_benchmark_from_name(experiment_name):
    if experiment_name == 'magic_square' or 'Magic' in experiment_name:
        return _construct_magic_square(N=3)
    elif experiment_name == 'schurs_lemma' or 'Schur' in experiment_name:
        return _construct_schurs_lemma(N=4)
    elif experiment_name == 'golomb_ruler' or 'Golomb' in experiment_name:
        return _construct_golomb_ruler(num_marks=6)
    elif experiment_name == 'greater_than_sudoku':
        return _construct_greaterThanSudoku()
    elif experiment_name == 'BIBD':
        return _construct_BIBD(v=7, b=7, r=3, k=3, l=1)
    elif experiment_name == 'job_shop_scheduling' or 'job' in experiment_name:
        return construct_job_shop_scheduling_problem(n_jobs=10, machines=3, horizon=40, seed=42)
    else:
        raise ValueError(f"Unknown experiment name: {experiment_name}")

def construct_custom(experiment, data_dir="data/exp", use_learned_model=False):
    """
    Constructs a custom model based on the given experiment.
    
    Args:
        experiment (str): The name of the experiment.
        data_dir (str): The directory where experiment data is stored.
        use_learned_model (bool): Flag to use a learned model or not.

    Returns:
        Tuple: Contains grid, constraints, model, variables, biases, and cls.
    """
    def parse_and_apply_constraints(file_path, variables, model=None):
        parsed_data = parse_con_file(file_path)
        constraints = []
        for con_type, var1, var2 in parsed_data:
            con_str = constraint_type_to_string(con_type)
            try:
                constraint = eval(f"variables[var1] {con_str} variables[var2]")
                constraints.append(constraint)
                if model is not None:
                    model += constraint
            except:
                print(f"Error in {con_type} {var1} {var2}")
        return constraints

    model = Model()
    total_global_constraints = 0
    vars_file = f"{data_dir}/{experiment}_var"
    vars = parse_vars_file(vars_file)
    dom_file = f"{data_dir}/{experiment}_dom"
    domain_constraints = parse_dom_file(dom_file)
    variables = [intvar(domain_constraints[0][0], domain_constraints[0][1], name=f"var{var}") for var in vars]
    grid = intvar(domain_constraints[0][0], domain_constraints[0][1], shape=(1, len(variables)), name="grid")
    for i, var in enumerate(variables):
        grid[1:i] = var

    model_file = f"{data_dir}/{experiment}_model"
    parsed_constraints, max_index = parse_model_file(model_file)
    total_global_constraints = len(parsed_constraints)
    if use_learned_model:
        for constraint_type, indices in parsed_constraints:
            if constraint_type == 'ALLDIFFERENT':
                model += AllDifferent([variables[i] for i in indices])


    if args.useCon:
        con_file = f"{data_dir}/{experiment}_con"
        if os.path.isfile(con_file):
            fixed_arity_ct = parse_and_apply_constraints(con_file, variables, model)
        else:  #TODO: Implement this
            # Use models from BenchMarkGenerator as oracle
            print(f"{con_file} does not exist. Using model from BenchMarkGenerator as oracle.")
            grid, C_T, model_bench, format_template = construct_benchmark_from_name(experiment)
            fixed_arity_ct = list(C_T)
            total_global_constraints += len(fixed_arity_ct)
            variables = list(grid.flatten())
            model += model_bench.constraints

    if args.onlyActive:
        biases = []
        cls = []
    else:
        bias_file = f"{data_dir}/{experiment}_bias"
        if os.path.isfile(bias_file):
            biases = parse_and_apply_constraints(bias_file, variables)
        else:
            biases = []

        cl_file = f"{data_dir}/{experiment}_cl"
        cls = parse_and_apply_constraints(cl_file, variables)

    grid = cp.cpm_array(np.expand_dims(variables, 0))

    if use_learned_model:
        C = list(model.constraints)
        C_T = set(toplevel_list(C))
        print(len(C_T))
    else:
        C_T = set(fixed_arity_ct)

    return grid, C_T, model, variables, biases, cls, total_global_constraints

def verify_global_constraints(experiment, data_dir="data/exp", use_learned_model=False):
    biasg = []
    def parse_and_apply_constraints(file_path, variables, model=None):
        parsed_data = parse_con_file(file_path)
        constraints = []
        for con_type, var1, var2 in parsed_data:
            con_str = constraint_type_to_string(con_type)
            try:
                constraint = eval(f"variables[var1] {con_str} variables[var2]")
                constraints.append(constraint)
                if model is not None:
                    model += constraint
            except:
                print(f"Error in {con_type} {var1} {var2}")
        return constraints

    model = Model()
    vars_file = f"{data_dir}/{experiment}_var"
    dom_file = f"{data_dir}/{experiment}_dom"
    if os.path.isfile(vars_file) and os.path.isfile(dom_file):
        vars = parse_vars_file(vars_file)
        domain_constraints = parse_dom_file(dom_file)
        variables = [intvar(domain_constraints[0][0], domain_constraints[0][1], name=f"var{var}") for var in vars]
        grid = intvar(domain_constraints[0][0], domain_constraints[0][1], shape=(1, len(variables)), name="grid")
        for i, var in enumerate(variables):
            grid[1:i] = var
        variables = [intvar(domain_constraints[0][0], domain_constraints[0][1], name=f"var{var}") for var in vars]
    else:
        # Use variables from the model
        print(f"{vars_file} does not exist. Using variables from BenchMarkGenerator model.")
        grid, C_T, model_bench, format_template = construct_benchmark_from_name(experiment)
        variables = list(grid.flatten())
        model += model_bench.constraints

    model_file = f"{data_dir}/{experiment}_model"
    if os.path.isfile(model_file):
        parsed_constraints, max_index = parse_model_file(model_file)
        total_global_constraints = len(parsed_constraints)
        for constraint_type, indices in parsed_constraints:
            if constraint_type == 'ALLDIFFERENT':
                if use_learned_model:
                    try:
                        model += AllDifferent([variables[i] for i in indices])
                    except:
                        print("Error in AllDifferent")
                try:
                    biasg.append(AllDifferent([variables[i] for i in indices]).decompose()[0])
                except:
                    print("Error in AllDifferent")

    if args.useCon:
        con_file = f"{data_dir}/{experiment}_con"
        if os.path.isfile(con_file):
            fixed_arity_ct = parse_and_apply_constraints(con_file, variables, model)
        else:
            # Use models from BenchMarkGenerator as oracle
            print(f"{con_file} does not exist. Using model from BenchMarkGenerator as oracle.")
            grid, C_T, model_bench, format_template = construct_benchmark_from_name(experiment)
            fixed_arity_ct = list(C_T)
            variables = list(grid.flatten())
            model += model_bench.constraints
            total_global_constraints = len(fixed_arity_ct)


    if args.onlyActive:
        biases = []
        cls = []
    else:
        bias_file = f"{data_dir}/{experiment}_bias"
        if os.path.isfile(bias_file):
            biases = parse_and_apply_constraints(bias_file, variables)
        else:
            biases = []

        cl_file = f"{data_dir}/{experiment}_cl"
        cls = parse_and_apply_constraints(cl_file, variables)


    grid = cp.cpm_array(np.expand_dims(variables, 0))

    if use_learned_model:
        C = list(model.constraints)
        C_T = set(toplevel_list(C))
        print(len(C_T))
    else:
        C_T = set(fixed_arity_ct)

    return grid, C_T, model, variables, biases, biasg, cls, total_global_constraints

def simplify_partition_type(partition_description):
    """
    Simplify the partition description by keeping only the partition type.

    Args:
    partition_description (str): The partition description string.

    Returns:
    str: Simplified partition description string.
    """
    prefix = 'Partition('
    if partition_description.startswith(prefix):
        parts = partition_description[len(prefix):].split(',')
        if len(parts) > 1:
            return parts[1].strip(' )')
    return partition_description
def count_cp(experiment, data_dir="data/exp", use_learned_model=False,benchmark=None):
    biasg = []
    C_L = []
    constraint_set = set()
    total_global_constraints = 0
    oracle =None

    def read_constraints_file(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        return lines

    def parse_constraints_by_partition(lines):
        constraints_by_partition = defaultdict(list)
        for line in lines:
            parts = line.strip().split()
            constraint = {
                "constraint_type": int(parts[0]),
                "var1_index": int(parts[1]),
                "var2_index": int(parts[2]),
                "partition_type": parts[3],
                "partition_number": parts[4],
                "sequence": parts[5]
            }
            partition_key = constraint["partition_type"] + "_" + str(constraint["partition_number"])
            constraints_by_partition[partition_key].append(constraint)
        return constraints_by_partition

    def parse_and_apply_constraints(file_path, variables, model=None):
        parsed_data = parse_con_file(file_path)
        constraints = []
        for con_type, var1, var2 in parsed_data:
            con_str = constraint_type_to_string(con_type)
            try:
                constraint = eval(f"variables[var1] {con_str} variables[var2]")
                constraints.append(constraint)
                if model is not None:
                    model += constraint
            except:
                print(f"Error in {con_type} {var1} {var2}")
            return constraints

    def read_all_partitions_constraints(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
        return lines

    def parse_line(line):
        import re
        pattern = r"var(\d+)\s*(==|!=|<=|>=|<|>)\s*var(\d+)"
        match = re.match(pattern, line.strip())
        if match:
            var1_index = int(match.group(1))
            operator = match.group(2)
            var2_index = int(match.group(3))
            return var1_index, var2_index, operator
        else:
            print(f"Line does not match pattern: {line}")
            return None, None, None

    model = Model()
    vars_file = f"{data_dir}/output_vars.txt"
    dom_file = f"{data_dir}/output_dom.txt"

    if os.path.isfile(vars_file) and os.path.isfile(dom_file):
        vars = parse_vars_file(vars_file)
        domain_constraints = parse_dom_file(dom_file)
        variables = [intvar(domain_constraints[0][0], domain_constraints[0][1], name=f"var{var}") for var in vars]
        grid = intvar(domain_constraints[0][0], domain_constraints[0][1], shape=(1, len(variables)), name="grid")
        for i, var in enumerate(variables):
            grid[0, i] = var
    else:
        # Use variables from the model
        print(f"{vars_file} does not exist. Using variables from BenchMarkGenerator model.")
        grid, C_T, model_bench, format_template = construct_benchmark_from_name(experiment)
        variables = list(grid.flatten())
        model += model_bench.constraints

    # Read and parse the all_partitions_constraints.txt file
    constraints_lines = read_all_partitions_constraints(f"{data_dir}/all_partitions_constraints.txt")
    for line in constraints_lines:
        var1_index, var2_index, operator = parse_line(line)
        if var1_index is not None:
            print(var1_index, var2_index, operator)
            try:
                var1 = variables[var1_index]
                var2 = variables[var2_index]
                if operator == '!=':
                    constraint = var1 != var2
                elif operator == '==':
                    constraint = var1 == var2
                elif operator == '<':
                    constraint = var1 < var2
                elif operator == '>':
                    constraint = var1 > var2
                elif operator == '<=':
                    constraint = var1 <= var2
                elif operator == '>=':
                    constraint = var1 >= var2
                else:
                    print(f"Unknown operator {operator}")
                    continue
                constraint_str = str(constraint)
                if constraint_str not in constraint_set:
                    constraint_set.add(constraint_str)
                    C_L.append(constraint)
                else:
                    #print(f"Duplicate constraint detected and skipped: {constraint_str}")
                    pass
            except:
                print(f"Error in {var1_index} {var2_index} {operator}")


    print(f"Number of constraints in C_L: {len(C_L)}")
    for c in C_L:
        print(c)
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{experiment}_{benchmark}_constraints.txt")

    with open(output_file, 'w') as f:
        for constraint in C_L:
            # Extract variables and operator from constraint
            # Assuming constraint is a Comparison object
            if isinstance(constraint, (cpmpy.expressions.core.Operator, cpmpy.expressions.core.Comparison)):
                op = constraint.name
                args = constraint.args
                var1 = args[0].name
                var2 = args[1].name
                # Map CPMPy's operator names to symbols
                op_symbol = {
                    '==': '==',
                    '!=': '!=',
                    '<=': '<=',
                    '>=': '>=',
                    '<': '<',
                    '>': '>'
                }.get(op, op)
                f.write(f"({var1}) {op_symbol} ({var2})\n")
            else:
                # If constraint is not a Comparison, convert to string
                f.write(f"{constraint}\n")
    constraints_lines = read_constraints_file(f"{data_dir}/output_necf.txt")
    constraints_by_partition = parse_constraints_by_partition(constraints_lines)
    total_global_constraints = 0
    for partition_key, constraints in constraints_by_partition.items():
        print(partition_key)
        total_global_constraints +=1
    # Append constraints_by_partition to biasg
    for partition_key, constraints in constraints_by_partition.items():
        temp_biasg = []
        for constraint in constraints:
            var1 = variables[constraint["var1_index"]]
            var2 = variables[constraint["var2_index"]]
            constraint_type = constraint["constraint_type"]
            if constraint_type == 1:
                temp_biasg.append(var1 == var2)
            else:
                temp_biasg.append(var1 != var2)
        biasg.append(temp_biasg)

    if True:
        con_file = f"{data_dir}/_con"
        if os.path.isfile(con_file):
            fixed_arity_ct = parse_and_apply_constraints(con_file, variables, model)
        else:
            # Use models from BenchMarkGenerator as oracle
            print(f"{con_file} does not exist. Using model from BenchMarkGenerator as oracle.")
            grid, C_T, model_bench, format_template = construct_benchmark_from_name(experiment)
            fixed_arity_ct = list(C_T)
            variables = list(grid.flatten())
            model += model_bench.constraints

    bias_file = f"{data_dir}/bc.txt"
    if os.path.isfile(bias_file):
        biases = parse_and_apply_constraints(bias_file, variables)
    else:
        biases = []

    for b in biasg:
        print(b)

    grid = cp.cpm_array(np.expand_dims(variables, 0))

    if use_learned_model:
        C = list(model.constraints)
        C_T = set(toplevel_list(C))
        print(len(C_T))
    else:
        C_T = set(fixed_arity_ct)

    return grid, C_T, model, variables, biases, biasg, C_L, total_global_constraints



def save_results(alg=None, inner_alg=None, qg=None, tl=None, t=None, blimit=None, fs=None, fc=None, bench=None, start_time=None, conacq=None, init_bias=None, init_cl=None, learned_global_cstrs=None):

    try:
        if conacq is None: conacq = ca_system
    except:
        pass
    if alg is None: alg = args.algorithm
    if qg is None: qg = args.query_generation
    if fs is None: fs = args.findscope
    if fc is None: fc = args.findc
    if bench is None: bench = benchmark_name
    if start_time is None: start_time = start

    end = time.time()  # to measure the total time of the acquisition process
    total_time = end - start_time

    print("\n\nConverged ------------------------")

    print("Total number of queries: ", conacq.metrics.queries_count)
    print("Number of generalization queries: ", conacq.metrics.gen_queries_count)
    print("Number of top-level queries: ", conacq.metrics.top_lvl_queries)
    print("Number of generated queries: ", conacq.metrics.generated_queries)
    print("Number of findscope queries: ", conacq.metrics.findscope_queries)

    avg_size = conacq.metrics.average_size_queries / conacq.metrics.queries_count if conacq.metrics.queries_count > 0 else 0
    print("Average size of queries: ", avg_size)

    print("Total time: ", total_time)
    average_waiting_time = total_time / conacq.metrics.queries_count if conacq.metrics.queries_count > 0 else 0

    print("Average waiting time for a query: ", average_waiting_time)
    print("Maximum waiting time for a query: ", conacq.metrics.max_waiting_time)
    print("Size of B: ", len(conacq.B)+len(toplevel_list(conacq.Bg)))
    print("C_L size: ", len(toplevel_list(conacq.C_l.constraints)))
    #print((toplevel_list(conacq.C_l.constraints)))
    res_name = ["results"]
    res_name.append(alg)

    if alg == "growacq":
        if inner_alg is None: inner_alg = args.inner_algorithm
        res_name.append(inner_alg)

    res_name.append(f"{str(qg)}")

    if qg == "tqgen":
        if tl is None: tl = args.time_limit
        if t is None: t = 0.1
        res_name.append(f"tl{str(tl)}")
        res_name.append(f"t{str(t)}")
    elif qg == "pqgen":
        if blimit is None: blimit = 5000
        res_name.append(f"bl{str(blimit)}")

    res_name.append(f"fs{str(fs)}")
    if fc != None:
        res_name.append(f"fc{str(fc)}")

    res_name.append(str(conacq.obj))

    if bench:
        res_name.append(bench)
    else:
        res_name.append("custom")

    if args.output:
        results_file = args.output+"/"+args.experiment+"_"+args.benchmark+"_"+args.type
    else:
        results_file = "_".join(res_name)

    constraints_file = results_file + "_constraints.txt"

    file_exists = os.path.isfile(results_file)

    # Create a DataFrame to store results
    results_df = pd.DataFrame(columns=["CL", "Tot_q", "top_lvl_q", "genacq_q", "gen_q", "fs_q", "fc_q", "avg|q|", "avg_t", "max_t", "tot_t", "conv", "init_bias", "init_cl", "learned_global_cstrs"])

    if file_exists:
        results_df = pd.read_csv(results_file)

    if len(init_bias) == 0 and len(init_cl) == 0:
        learned_global_cstrs = "-"

    new_result = {
        "type": args.type,
        "CL": len(toplevel_list(conacq.C_l.constraints)),
        "Tot_q": conacq.metrics.queries_count,
        "top_lvl_q": conacq.metrics.top_lvl_queries,
        "genacq_q":  conacq.metrics.gen_queries_count,
        "gen_q": conacq.metrics.generated_queries,
        "fs_q": conacq.metrics.findscope_queries,
        "fc_q": conacq.metrics.findc_queries,
        "avg|q|": round(conacq.metrics.average_size_queries / conacq.metrics.queries_count, 4) if conacq.metrics.queries_count > 0 else 0,
        "avg_t": round(average_waiting_time, 4),
        "max_t": round(conacq.metrics.max_waiting_time, 4),
        "tot_t": round(total_time, 4),
        "conv": conacq.metrics.converged,
        "init_bias": len(init_bias),
        "init_cl": len(init_cl),
        "learned_global_cstrs": learned_global_cstrs
    }

    new_result_df = pd.DataFrame([new_result])
    results_df = pd.concat([results_df, new_result_df], ignore_index=True)
    results_df.to_csv(results_file, index=False)

    constraints = toplevel_list(conacq.C_l.constraints)
    with open(constraints_file, 'w') as f:
        for constraint in constraints:
            f.write(str(constraint) + "\n")


def run_jar_with_config(jar_path, config_path):
    result = subprocess.run(['java', '-jar', jar_path, config_path], capture_output=True, text=True)
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


if __name__ == "__main__":

    args = parse_args()
    if args.findscope is None:
        fs_version = 2
    else:
        fs_version = args.findscope
    if args.findc is None:
        fc_version = 1
    else:
        fc_version = args.findc
    start = time.time()
    gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 <= var2", "var1 >= var2"]
    # gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 <= var2", "var1 >= var2",
    #          "abs(var1 - var2) != abs(var3 - var4)"] + \
    #             [f"var1 + {i} == var2" for i in range(1, 4 + 1)] + \
    #             [f"var2 + {i} == var1" for i in range(1, 4 + 1)] # 4 = max duration

    if args.benchmark == "mineask":
        print("Running Mine&Ask")
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = count_cp(benchmark_name,
                                                                                                     path,
                                                                                         False,benchmark_name)
        print("Size of bias: ", len(set(bias)))
        print("Size of biasg: ", len(toplevel_list(biasg)), len(biasg))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        bias = []
        biasg = []
        C_l = []
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=bias, Bg=biasg, C_l=C_l, benchmark=args.benchmark)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)
        exit()

    if args.benchmark == "genacq":
        print("Running GenAcq")

        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = verify_global_constraints(benchmark_name,
                                                                                                     path,
                                                                                                     False)
        print("Size of bias: ", len(set(bias)))
        print("Size of biasg: ", len(toplevel_list(biasg)), len(biasg))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        bias = []
        biasg = []
        C_l = []
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=bias, Bg=biasg, C_l=C_l, benchmark=args.benchmark)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)
        exit()

    if args.benchmark == "vgc": #verify global constraints - genacq
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = verify_global_constraints(benchmark_name, path, False)
        print("Size of bias: ", len(set(bias)))
        print("Size of biasg: ",len(toplevel_list(biasg)), len(biasg))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        #C_l = [constraint for constraint in C_l if constraint not in biasg]
        if args.onlyActive:
            bias = []
            C_l = []
        else:
           # _bias = C_T - set(bias) - set(C_l)
            #bias.extend(_bias)
            pass


        if args.emptyCL or True:
            C_l = []
        print("-------------------")
        print("Size of bias: ", len(set(bias)))
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=bias, Bg=biasg, C_l=C_l, benchmark=args.benchmark)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)
        exit()

    if args.benchmark == "countcp": #count cp
        #run_count_cp_and_get_results(args.countcp_exp, args.countcp_output_dir, args.countcp_name, args.countcp_input)
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = count_cp(benchmark_name, path, False,benchmark_name)
        print("Size of bias: ", len(set(bias)))
        print("Size of biasg: ",len(toplevel_list(biasg)), len(biasg))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        #C_l = [constraint for constraint in C_l if constraint not in biasg]

        _bias = C_T - set(bias) - set(C_l)
        biasg_set = set(toplevel_list(biasg))
        _bias = [b for b in _bias if b not in biasg_set]

        bias.extend(_bias)
        print(bias)
        C_l = []
        print("-------------------")
        print("Size of bias: ", len(set(bias)))
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=bias, Bg=biasg, C_l=C_l, benchmark=args.benchmark)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints,conacq=ca_system)
        exit()

    if args.benchmark == "countcp_al": #count cp al
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = count_cp(benchmark_name, path, False,benchmark_name)
        print("Size of bias: ", len(set(bias)))

        print("Size of biasg: ",len(toplevel_list(biasg)), len(biasg))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        #C_l = [constraint for constraint in C_l if constraint not in biasg]

        #_bias = set(bias)
        _bias = C_T - set(bias) - set(C_l)
        biasg_set = set(toplevel_list(biasg))
        _bias = [b for b in _bias if b not in biasg_set]

        bias.extend(_bias)
        print(bias)
        print("-------------------")
        print("Size of bias: ", len(set(bias)))
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=_bias, Bg=[], C_l=C_l, benchmark=args.benchmark)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)
        exit()

    if args.benchmark == "countcp_only":  # count cp al
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l, total_global_constraints = count_cp(benchmark_name, path, False,benchmark_name)
        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)
        exit()

    if args.benchmark == "custom": #mquack2 custom problem
        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, C_l, total_global_constraints = construct_custom(benchmark_name, path, False)
        print("Size of bias: ", len(set(bias)))
        print("Size of C_l: ", len(C_l))
        print("Size of C_T: ", len(C_T))
        if args.onlyActive:
            bias = []
            C_l = []
        else:
            _bias = (C_T - set(bias))
            bias.extend(_bias)

        # if args.emptyCL:
        #     C_l = []
        print("Size of bias: ", len(bias))
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, B=bias, Bg=[], C_l=C_l, X=X)
        ca_system.learn()

        save_results(init_bias=bias, init_cl=C_l, learned_global_cstrs=total_global_constraints)

        exit()
    # else:
    #     benchmark_name, grid, C_T, oracle, gamma = construct_benchmark()
    # grid.clear()

    # print("Size of C_T: ", len(C_T))
    #
    # all_cons = []
    # X = list(list(grid.flatten()))
    # for relation in gamma:
    #     if relation.count("var") == 2:
    #         for v1, v2 in all_pairs(X):
    #             print(v1,v2)
    #             constraint = relation.replace("var1", "v1")
    #             constraint = constraint.replace("var2", "v2")
    #             constraint = eval(constraint)
    #             all_cons.append(constraint)
    #             all_cons.append(constraint)
    #
    # bias = all_cons
    # bias.pop()
    # C_l=[all_cons[i] for i in range(0, len(all_cons), 2)]
    # C_l=[all_cons[-1]]
    #
    # bias_filtered = []
    # for item_cl in C_l:
    #     for item_bias in bias:
    #         if not are_comparisons_equal(item_cl, item_bias):
    #             bias_filtered.append(item_cl)
    # bias = bias_filtered



    #
    #
    # if args.algorithm == "quacq":
    #     ca_system = QuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
    #                       time_limit=args.time_limit, findscope_version=fs_version,
    #                       findc_version=fc_version)
    # elif args.algorithm == "mquacq":
    #     ca_system = MQuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
    #                        time_limit=args.time_limit, findscope_version=fs_version,
    #                        findc_version=fc_version)
    # elif args.algorithm == "mquacq2":
    #     ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
    #                         time_limit=args.time_limit, findscope_version=fs_version,
    #                         findc_version=fc_version)
    # elif args.algorithm == "mquacq2-a":
    #     ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
    #                         time_limit=args.time_limit, findscope_version=fs_version,
    #                         findc_version=fc_version, perform_analyzeAndLearn=True)
    # elif args.algorithm == "growacq":
    #     ca_system = GrowAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
    #                         time_limit=args.time_limit, findscope_version=fs_version,
    #                         findc_version=fc_version)
    #
    # ca_system.learn()

    save_results()
