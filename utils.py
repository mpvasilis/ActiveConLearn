import math
import time

from cpmpy import intvar, boolvar, Model, all, sum, SolverLookup
from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.variables import _NumVarImpl
from sklearn.utils import class_weight
import numpy as np

import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain

from ConAcq import SOLVER
import json


def parse_dom_file(file_path):
    domain_constraints = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 3:
                var_index = int(parts[0])
                lower_bound = int(parts[2])
                upper_bound = int(parts[-1])
                domain_constraints[var_index] = (lower_bound, upper_bound)
    return domain_constraints

def parse_con_file(file_path):
    biases = []

    with open(file_path, 'r') as file:
        for line in file:
            con_type, var1, var2 = map(int, line.strip().split())
            biases.append((con_type, var1, var2))

    return biases


def constraint_type_to_string(con_type):
    return {
        0: "!=",
        1: "==",
        2: ">",
        3: "<",
        4: ">=",
        5: "<="
    }.get(con_type, "Unknown")


def parse_vars_file(file_path):
    with open(file_path, 'r') as file:
        total_vars = int(file.readline().strip())
        vars_values = [0] * total_vars

        for i, line in enumerate(file):
            value, _ = map(int, line.split())
            vars_values[i] = value

    return vars_values


def parse_model_file(file_path):
    max_index = -1
    constraints = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue

            constraint_type, indices_part = parts[0], parts[1]
            indices = re.findall(r'\d+', indices_part)
            indices = [int(i) for i in indices]

            max_index_in_line = max(indices)
            if max_index_in_line > max_index:
                max_index = max_index_in_line

            if constraint_type == 'ALLDIFFERENT':
                constraints.append((constraint_type, indices))

    return constraints, max_index


def are_comparisons_equal(comp1, comp2):
    """
    Checks if two Comparison objects are equal.

    :param comp1: The first Comparison object.
    :param comp2: The second Comparison object.
    :return: True if the Comparisons are equal, False otherwise.
    """
    if comp1.name != comp2.name:
        return False

    if comp1.args[0] != comp2.args[0]:
        return False

    if comp1.args[1] != comp2.args[1]:
        return False

    return True


def check_value(c):
    return bool(c.value())


def get_con_subset(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y)]


def get_kappa(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is False]


def get_lambda(B, Y):
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_scope(c)).issubset(Y) and c.value() is True]


def gen_pairwise(v1, v2):
    return [v1 == v2, v1 != v2, v1 < v2, v1 > v2]


# to create the binary oracle
def gen_pairwise_ineq(v1, v2):
    return [v1 != v2]


def alldiff_binary(grid):
    for v1, v2 in all_pairs(grid):
        for c in gen_pairwise_ineq(v1, v2):
            yield c


def gen_scoped_cons(grid):
    # rows
    for row in grid:
        for v1, v2 in all_pairs(row):
            for c in gen_pairwise_ineq(v1, v2):
                yield c
    # columns
    for col in grid.T:
        for v1, v2 in all_pairs(col):
            for c in gen_pairwise_ineq(v1, v2):
                yield c

    # DT: Some constraints are not added here, I will check it and fix  TODO
    # subsquares
    for i1 in range(0, 4, 2):
        for i2 in range(i1, i1 + 2):
            for j1 in range(0, 4, 2):
                for j2 in range(j1, j1 + 2):
                    if (i1 != i2 or j1 != j2):
                        for c in gen_pairwise_ineq(grid[i1, j1], grid[i2, j2]):
                            yield c


def gen_all_cons(grid):
    # all pairs...
    for v1, v2 in all_pairs(grid.flat):
        for c in gen_pairwise(v1, v2):
            yield c


def construct_bias(X, gamma):
    all_cons = []

    X = list(X)

    for relation in gamma:

        if relation.count("var") == 2:

            for v1, v2 in all_pairs(X):
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)

                all_cons.append(constraint)

        elif relation.count("var") == 4:

            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    for x in range(j + 1, len(X) - 1):
                        for y in range(x + 1, len(X)):
                            if (y != i and x != j and x != i and y != j):
                                #            for v1, v2 in all_pairs(X):
                                #                for v3, v4 in all_pairs(X):
                                constraint = relation.replace("var1", "X[i]")
                                constraint = constraint.replace("var2", "X[j]")
                                constraint = constraint.replace("var3", "X[x]")
                                constraint = constraint.replace("var4", "X[y]")
                                constraint = eval(constraint)

                                all_cons.append(constraint)

    return all_cons


def construct_bias_for_var(X, gamma, v1):
    all_cons = []

    for relation in gamma:
        if relation.count("var") == 2:
            for v2 in X:
                if not (v1 is v2):
                    constraint = relation.replace("var1", "v1")
                    constraint = constraint.replace("var2", "v2")
                    constraint = eval(constraint)

                    all_cons.append(constraint)

        elif relation.count("var") == 4:
            X = X.copy()
            X.reverse()
            print(X)
            for j in range(0, len(X)):
                for x in range(j + 1, len(X) - 1):
                    for y in range(x + 1, len(X)):
                        # if (y != i and x != j and x != i and y != j):
                        #            for v1, v2 in all_pairs(X):
                        #                for v3, v4 in all_pairs(X):
                        constraint = relation.replace("var1", "v1")
                        constraint = constraint.replace("var2", "X[j]")
                        constraint = constraint.replace("var3", "X[x]")
                        constraint = constraint.replace("var4", "X[y]")
                        constraint = eval(constraint)

                        all_cons.append(constraint)

    return all_cons


def get_scopes_vars(C):
    return set([x for scope in [get_scope(c) for c in C] for x in scope])


def get_scopes(C):
    return list(set([tuple(get_scope(c)) for c in C]))


from cpmpy.expressions.variables import _IntVarImpl

def get_scope(constraint):
    """
    Utility function to extract the scope of variables from a constraint.
    """
    if isinstance(constraint, Comparison):
        return [var for var in constraint.args if isinstance(var, _IntVarImpl)]
    elif isinstance(constraint, Operator):
        return [var for var in constraint.args if isinstance(var, _IntVarImpl)]
    return []


import networkx as nx
from collections import defaultdict


def calculate_modularity(G, communities):
    """
    Calculate the modularity of a given partition.
    """
    m = G.size(weight='weight')
    degrees = dict(G.degree(weight='weight'))
    Q = 0
    for community in communities:
        Lc = 0
        Dc = 0
        for u in community:
            Dc += degrees[u]
            for v in community:
                if G.has_edge(u, v):
                    Lc += G[u][v].get('weight', 1)
        Q += (Lc / (2 * m)) - (Dc / (2 * m)) ** 2
    return Q


def get_communities(partition):
    """
    Get the communities from the partition.
    """
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    return list(communities.values())


def optimize_modularity(G, max_iterations=1000, min_modularity_improvement=0.0001):
    partition = {node: i for i, node in enumerate(G.nodes())}
    best_modularity = -1
    best_partition = partition.copy()
    improvement = True
    iteration = 0

    while improvement and iteration < max_iterations:
        improvement = False
        current_modularity = calculate_modularity(G, get_communities(partition))

        for node in G.nodes():
            best_community = partition[node]
            best_increase = 0
            current_community = partition[node]

            # remove node from its current community
            partition[node] = -1

            for neighbor in G.neighbors(node):
                if partition[neighbor] != -1:
                    community = partition[neighbor]
                    partition[node] = community
                    new_modularity = calculate_modularity(G, get_communities(partition))
                    increase = new_modularity - current_modularity

                    if increase > best_increase:
                        best_increase = increase
                        best_community = community

                    partition[node] = -1

            partition[node] = best_community

            if best_increase > min_modularity_improvement:
                improvement = True

        new_modularity = calculate_modularity(G, get_communities(partition))

        if new_modularity > best_modularity:
            best_modularity = new_modularity
            best_partition = partition.copy()

        # aggregate the graph
        communities = get_communities(partition)
        new_G = nx.Graph()

        for i, community in enumerate(communities):
            new_node = i
            new_G.add_node(new_node)

            for node in community:
                for neighbor in G.neighbors(node):
                    if partition[neighbor] == partition[node]:
                        if new_G.has_edge(new_node, new_node):
                            new_G[new_node][new_node]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, new_node, weight=G[node][neighbor].get('weight', 1))
                    else:
                        neighbor_comm = partition[neighbor]

                        if new_G.has_edge(new_node, neighbor_comm):
                            new_G[new_node][neighbor_comm]['weight'] += G[node][neighbor].get('weight', 1)
                        else:
                            new_G.add_edge(new_node, neighbor_comm, weight=G[node][neighbor].get('weight', 1))

        G = new_G
        partition = {node: i for i, node in enumerate(G.nodes())}
        iteration += 1

    return get_communities(best_partition)


def get_arity(constraint):
    return len(get_scope(constraint))


def get_min_arity(C):
    if len(C) > 0:
        return min([get_arity(c) for c in C])
    return 0


def get_max_arity(C):
    if len(C) > 0:
        return max([get_arity(c) for c in C])
    return 0


def get_relation(c, gamma):
    scope = get_scope(c)

    for i in range(len(gamma)):
        relation = gamma[i]

        if relation.count("var") != len(scope):
            continue

        constraint = relation.replace("var1", "scope[0]")
        for j in range(1, len(scope)):
            constraint = constraint.replace("var" + str(j + 1), "scope[" + str(j) + "]")

        constraint = eval(constraint)

        if hash(constraint) == hash(c):
            return i

    return -1


def get_var_name(var):
    name = re.findall("\[\d+[,\d+]*\]", var.name)
    if not name:
        name = var.name.replace('var','')
    else:
        name = var.name.replace(name[0], '')
    return name


def get_var_ndims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    dims = re.findall("\[\d+[,\d+]*\]", var.name)
    if dims:
        dims_str = "".join(dims)
        dims = re.split("[\[\]]", dims_str)[1]
        dims = [int(dim) for dim in re.split(",", dims)]
    else:
        dims = var.name.replace("var","")
        dims=[int(dims)]

    return dims


def get_divisors(n):
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def join_con_net(C1, C2):
    C3 = [[c1 & c2 if c1 is not c2 else c1 for c2 in C2] for c1 in C1]
    C3 = list(chain.from_iterable(C3))
    C3 = remove_redundant_conj(C3)
    return C3


def remove_redundant_conj(C1):
    C2 = list()

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        flag_eq = False
        flag_neq = False
        flag_geq = False
        flag_leq = False
        flag_ge = False
        flag_le = False

        for c1 in conj_args:
            print(c1.name)
            # Tias is on 3.9, no 'match' please!
            if c1.name == "==":
                flag_eq = True
            elif c1.name == "!=":
                flag_neq = True
            elif c1.name == "<=":
                flag_leq = True
            elif c1.name == ">=":
                flag_geq = True
            elif c1.name == "<":
                flag_le = True
            elif c1.name == ">":
                flag_ge = True
            else:
                raise Exception("constraint name is not recognised")

            if not ((flag_eq and (flag_neq or flag_le or flag_ge)) or (
                    (flag_leq or flag_le) and ((flag_geq or flag_ge)))):
                C2.append(c)
    return C2


def get_max_conjunction_size(C1):
    max_conj_size = 0

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        max_conj_size = max(len(conj_args), max_conj_size)

    return max_conj_size


def get_delta_p(C1):
    max_conj_size = get_max_conjunction_size(C1)

    Delta_p = [[] for _ in range(max_conj_size)]

    for c in C1:

        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        Delta_p[len(conj_args) - 1].append(c)

    return Delta_p


def compute_sample_weights(Y):
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


class Metrics:

    def __init__(self):
        self.gen_queries_count = 0
        self.queries_count = 0
        self.top_lvl_queries = 0
        self.generated_queries = 0
        self.findscope_queries = 0
        self.findc_queries = 0

        self.average_size_queries = 0

        self.start_time_query = time.time()
        self.max_waiting_time = 0
        self.generation_time = 0

        self.converged = 1
        self.N_egativeQ = set()
        self.gen_no_answers = 0
        self.gen_yes_answers = 0

    def increase_gen_queries_count(self, amount=1):
        self.gen_queries_count += amount

    def increase_queries_count(self, amount=1):
        self.queries_count += amount

    def increase_top_queries(self, amount=1):
        self.top_lvl_queries += amount

    def increase_generated_queries(self, amount=1):
        self.generated_queries += amount

    def increase_findscope_queries(self, amount=1):
        self.findscope_queries += amount

    def increase_findc_queries(self, amount=1):
        self.findc_queries += amount

    def increase_generation_time(self, amount):
        self.generation_time += self.generation_time

    def increase_queries_size(self, amount):
        self.average_size_queries += 1

    def aggreagate_max_waiting_time(self, max2):
        if self.max_waiting_time < max2:
            self.max_waiting_time = max2

    def aggregate_convergence(self, converged2):
        if self.converged + converged2 < 2:
            self.converged = 0

    def __add__(self, other):

        new = self

        new.increase_queries_count(other.queries_count)
        new.increase_top_queries(other.top_lvl_queries)
        new.increase_generated_queries(other.generated_queries)
        new.increase_findscope_queries(other.findscope_queries)
        new.increase_findc_queries(other.findc_queries)
        new.increase_generation_time(other.generation_time)
        new.increase_queries_size(other.average_size_queries)

        new.aggreagate_max_waiting_time(other.max_waiting_time)
        new.aggregate_convergence(other.converged)

        return new


def find_suitable_vars_subset2(l, B, Y):
    if len(Y) <= get_min_arity(B) or len(B) < 1:
        return Y

    scope = get_scope(B[0])
    Y_prime = list(set(Y) - set(scope))

    l2 = int(l) - len(scope)

    if l2 > 0:
        Y1 = Y_prime[:l2]
    else:
        Y1 = []

    [Y1.append(y) for y in scope]

    return Y1


def generate_findc_query(L, delta):
    # constraints from  B are taken into account as soft constraints who we do not want to satisfy (i.e. we want to violate)
    # This is the version of query generation for the FindC function that was presented in "Constraint acquisition via Partial Queries", IJCAI 2013

    tmp = Model(L)

    objective = sum([c for c in delta])  # get the amount of satisfied constraints from B

    # at least 1 violated and at least 1 satisfied
    # we want this to assure that each answer of the user will reduce
    # the set of candidates
    # Difference with normal query generation: if all are violated, we already know that the example will
    # be a non-solution due to previous answers

    tmp += objective < len(delta)
    tmp += objective > 0

    # Try first without objective

    s = SolverLookup.get(SOLVER, tmp)
    flag = s.solve()

    if not flag:
        # UNSAT, stop here
        return flag

    Y = get_scope(delta[0])
    Y = list(dict.fromkeys(Y))  # remove duplicates

    # Next solve will change the values of the variables in the lY2
    # so we need to return them to the original ones to continue if we dont find a solution next
    values = [x.value() for x in Y]

    # so a solution was found, try to find a better one now
    s.solution_hint(Y, values)

    # run with the objective
    s.minimize(abs(objective - round(len(delta) / 2)))  # we want to try and do it like a dichotomic search
    # s.minimize(objective)  # we want to minimize the violated cons

    flag2 = s.solve(time_limit=0.2)

    if not flag2:
        tmp = Model()
        i = 0
        for x in Y:
            tmp += x == values[i]
            i = i + 1

        tmp.solve()

        return flag

    else:
        return flag2


def generate_findc2_query(L, delta):
    # This is the version of query generation for the FindC function that was presented in "Constraint acquisition through Partial Queries", AIJ 2023

    tmp = Model(L)

    max_conj_size = get_max_conjunction_size(delta)
    delta_p = get_delta_p(delta)

    p = intvar(0, max_conj_size)
    kappa_delta_p = intvar(0, len(delta), shape=(max_conj_size,))
    p_soft_con = boolvar(shape=(max_conj_size,))

    for i in range(max_conj_size):
        tmp += kappa_delta_p[i] == sum([c for c in delta_p[i]])
        p_soft_con[i] = (kappa_delta_p[i] > 0)

    tmp += p == min([i for i in range(max_conj_size) if (kappa_delta_p[i] < len(delta_p[i]))])

    objective = sum([c for c in delta])  # get the amount of satisfied constraints from B

    # at least 1 violated and at least 1 satisfied
    # we want this to assure that each answer of the user will reduce
    # the set of candidates
    tmp += objective < len(delta)
    tmp += objective > 0

    # Try first without objective
    s = SolverLookup.get(SOLVER, tmp)

    print("p: ", p)

    # run with the objective
    s.minimize(100 * p - p_soft_con[p])

    flag = s.solve()
    #        print("OPT solve", s.status())

    return flag


def write_solutions_to_json(instance, size, format_template, solutions, non_solutions, problem_type, output_file):
    data = {
        "instance": instance,
        "size": size,
        "formatTemplate": format_template,
        "solutions": [{"array": sol} for sol in solutions],
        "nonSolutions": [{"array": non_sol} for non_sol in non_solutions],
        "problemType": problem_type
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)