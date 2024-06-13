import itertools
import random
import time
from statistics import mean, stdev

from cpmpy.expressions.core import Comparison
from cpmpy.expressions.variables import _IntVarImpl

SOLVER = "ortools"

from cpmpy import *
from cpmpy.transformations.get_variables import get_variables
from cpmpy.expressions.utils import is_any_list
from cpmpy.transformations.normalize import toplevel_list
from utils import *
import math
from ortools.sat.python import cp_model as ort

from utils import find_suitable_vars_subset2

partial = False

class ConAcq:

    def __init__(self, gamma, grid, ct=list(), B=None, Bg=None, X=set(), C_l=None, qg="pqgen", obj="proba",
                 time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000):

        self.debug_mode = True

        # Target network
        self.C_T = ct

        self.grid = grid
        self.gamma = gamma

        if Bg is None:
            Bg = []
        self.Bg = Bg

        if B is None:
            B = []

        self.B = [c for c in B if c not in frozenset(toplevel_list(Bg))]

        # Guery generation, FindScope and FindC versions
        self.qg = qg
        self.fs = findscope_version
        self.fc = findc_version

        # Objective
        self.obj = obj

        # For the counts
        self.counts = [0] * len(gamma)
        self.countsB = [0] * len(gamma)

        # Initialize learned network
        if C_l is None:
            C_l = []
        if len(C_l) > 0:
            self.C_l = Model(C_l)
        else:
            self.C_l = Model()

        # Initialize variables
        if len(X) > 0:
            self.X = list(X)
        else:
            self.X = list(grid.flatten())

        # Hash the variables
        self.hashX = [hash(x) for x in self.X]

        # For TQ-Gen
        self.alpha = 0.8
        self.l = len(self.X)

        # Query generation time limit
        if time_limit is None:
            time_limit = 1
        self.time_limit = time_limit

        # TQGen's time limit tau
        if tqgen_t is None:
            tqgen_t = 0.20
        self.tqgen_t = tqgen_t

        # Bias size limit for determining type of query generation (denoted with 'l' in paper)
        self.qgen_blimit = qgen_blimit

        # Time limit for during FindScope with objective function
        self.fs_limit = 0.5

        # To be used in the constraint features
        # -------------------------------------

        # Length of dimensions per variable name
        self.var_names = list(set([get_var_name(x) for x in self.X]))
        var_dims = [[get_var_dims(x) for x in self.X if get_var_name(x) == self.var_names[i]] for i in
                    range(len(self.var_names))]
        self.dim_lengths = [
            [np.max([var_dims[i][k][j] for k in range(len(var_dims[i]))]) + 1 for j in range(len(var_dims[i][0]))] for i
            in range(len(var_dims))]

        self.dim_divisors = list()

        for i in range(len(self.dim_lengths)):
            dim_divisors = list()
            for j in range(len(self.dim_lengths[i])):
                divisors = get_divisors(self.dim_lengths[i][j])
                dim_divisors.append(divisors)

            self.dim_divisors.append(dim_divisors)

        self.metrics = Metrics()

    def flatten_blists(self, C):
        if not is_any_list(C):
            C = [C]
        i=0
        while i < len(self.Bg):
            bl = self.Bg[i]
            if any(c in frozenset(bl) for c in C):
                self.B.extend(bl)
                self.Bg.pop(i)
                i = i-1
            i +=1

        self.B = list(dict.fromkeys(self.B))

    def remove_from_bias(self, C):

        #Flatten that list adding it to B (removal of such constraints from B will happen in next step)
        self.flatten_blists(C)

        # Remove all the constraints from network C from B
        prev_B_length = len(self.B)
        self.B = list(set(self.B) - set(C))

        if self.debug_mode:
            print(f"Removed from bias: {C}")
#        if not (prev_B_length - len(C) == len(self.B)):
#            print("B: ", self.B)
#            print("C:", C)
#            raise Exception(f'Something was not removed properly: {prev_B_length} - {len(C)} = {len(self.B)}')

#        for bl in self.Bg:
#            if any(c in bl for c in C):
#                self.Bg -= bl

    def add_to_cl(self, c):

        # Add a constraint c to the learned network
        if self.debug_mode:
            print(f"adding {c} to C_L")
        self.C_l += c
        self.genAcq(c)



    def remove(self, B, C):

        # Remove all constraints in network C from B
        lenB = len(B)
        lenC = len(C)
        lenB_init = len(B)

        i = 0

        while i < lenB:
            if any(B[i] is c2 for c2 in C):
                # B[i] in set(C) in condition is slower

                B.pop(i)
                i -= 1
                lenB -= 1
            i += 1

        if lenB_init - len(B) != lenC:
            raise Exception("Removing constraints from Bias did not result in reducing its size")


    def remove_scope_from_bias(self, scope):

        # Remove all constraints with the given scope from B
        scope_set = set(scope)
        learned_con_rel = get_relation(self.C_l.constraints[-1], self.gamma)
        B_scopes_set = [set(get_scope(c)) for c in self.B + toplevel_list(self.Bg)]

        removing = [c for i, c in enumerate(self.B + toplevel_list(self.Bg)) if B_scopes_set[i] == scope_set
                    and get_relation(c, self.gamma) != learned_con_rel]
        self.flatten_blists(removing)

#        prev_B_length = len(self.B)
        self.B = [c for i, c in enumerate(self.B) if not (B_scopes_set[i] == scope_set)]

#        if len(self.B) == prev_B_length:
#            if self.debug_mode:
#                print(self.B)
#                print(scope)
#            raise Exception("Removing constraints from Bias did not result in reducing its size")

    def set_cl(self, C_l):

        if isinstance(C_l, Model):
            self.C_l = C_l
        elif isinstance(C_l, set) or isinstance(C_l, list):
            self.C_l = Model(C_l)

    def call_findc(self, scope):

        # Call the specified findC function

        if self.fc == 1:

            # Initialize delta
            delta = get_con_subset(self.B + toplevel_list(self.Bg), scope)
            delta = [c for c in delta if check_value(c) is False]

            c = self.findC(scope, delta)

        else:
            c = self.findC2(scope)

        if c is None:
            raise Exception("findC did not find any, collapse")
        else:
            self.add_to_cl(c)
            self.remove_scope_from_bias(scope)
            self.genAcq(c)


    def call_findscope(self, Y, kappa):

        # Call the specified findScope function

        if self.fs == 1:
            scope = self.findScope(self.grid.value(), set(), Y, do_ask=False)
        else:
            scope = self.findScope2(self.grid.value(), set(), Y, kappa)

        return scope

    def adjust_tq_gen(self, l, a, answer):

        # Adjust the number of variables taken into account in the next iteration of TQGen

        if answer:
            l = min([int(math.ceil(l / a)), len(self.X)])
        else:
            l = int((a * l) // 1)  # //1 to round down
            if l < get_min_arity(self.B):
                l = 2

        return l

    def tq_gen(self, alpha, t, l):

        # Generate a query using TQGen
        # alpha: reduction factor
        # t: solving timeout
        # l: expected query size

        ttime = 0

        while ttime < self.time_limit and len(self.B) > 0:

            t = min([t, self.time_limit - ttime])
            l = max([l, get_min_arity(self.B)])

            Y = find_suitable_vars_subset2(l, self.B, self.X)

            B = get_con_subset(self.B, Y)
            Cl = get_con_subset(self.C_l.constraints, Y)

            m = Model(Cl)
            s = SolverLookup.get(SOLVER, m)

            # Create indicator variables upfront
            V = boolvar(shape=(len(B),))
            s += (V != B)

            # We want at least one constraint to be violated
            if self.debug_mode:
                print("length of B: ", len(B))
                print("l: ", l)
            s += sum(V) > 0

            t_start = time.time()
            flag = s.solve(time_limit=t)
            ttime = ttime + (time.time() - t_start)

            if flag:
                return flag, Y

            if s.ort_status == ort.INFEASIBLE:
                [self.add_to_cl(c) for c in B]
                self.remove_from_bias(B)
            else:
                l = int((alpha * l) // 1)  # //1 to round down

        if len(self.B) > 0:
            self.converged = 0

        return False, list()

    def generate_query(self):

        # A basic version of query generation for small problems. May lead
        # to premature convergence, so generally not used

        if len(self.B + toplevel_list(self.Bg)) == 0:
            return False

        # B are taken into account as soft constraints that we do not want to satisfy (i.e., that we want to violate)
        m = Model(self.C_l.constraints)  # could use to-be-implemented m.copy() here...

        # Get the amount of satisfied constraints from B
        objective = sum([c for c in self.B + toplevel_list(self.Bg)])

        # We want at least one constraint to be violated to assure that each answer of the
        # user will reduce the set of candidates
        m += objective < len(self.B + toplevel_list(self.Bg))

        s = SolverLookup.get(SOLVER, m)
        flag = s.solve(time_limit=600)

        if not flag:
            # If a solution is found, then continue optimizing it
            if s.ort_status == ort.UNKNOWN:
                self.converged = 0

        return flag

    def pqgen(self, time_limit=1):

        # Generate a query using PQGen

        # Start time (for the cutoff t)
        t0 = time.time()

        # Project down to only vars in scope of B
        Y = list(dict.fromkeys(get_variables(self.B + toplevel_list(self.Bg))))
        lY = list(Y)

        B = get_con_subset(self.B + toplevel_list(self.Bg), Y)
        Cl = get_con_subset(self.C_l.constraints, Y)

        global partial

        # If no constraints left in B, just return
        if len(B) == 0:
            return False, set()

        # If no constraints learned yet, start by just generating an example in all the variables in Y
        if len(Cl) == 0:
            Cl = [sum(Y) >= 1]

        if not partial and len(self.B + toplevel_list(self.Bg)) > self.qgen_blimit:

            m = Model(Cl)
            flag = m.solve()  # no time limit to ensure convergence

            if flag and not all([c.value() for c in B]):
                return flag, lY
            else:
                partial = True

        m = Model(Cl)
        s = SolverLookup.get(SOLVER, m)

        hybridB = self.Bg + self.B

        # Create indicator variables upfront
        V = boolvar(shape=(len(hybridB),))
        s += [V[i] == all(hybridB[i]) if is_any_list(hybridB[i]) else V[i] == hybridB[i] for i in range(len(V))]
        #s += (V == all(hybridB))

        # We want at least one constraint to be violated to assure that each answer of the user
        # will lead to new information
        s += ~all(V)

        # Solve first without objective (to find at least one solution)
        flag = s.solve(time_limit=600)
        t1 = time.time() - t0
        if not flag or (t1 > time_limit):
            # UNSAT or already above time_limit, stop here --- cannot maximize
            if self.debug_mode:
                print("RR1Time:", time.time() - t0, len(B), len(Cl))
            return flag, lY

        # Next solve will change the values of the variables in the lY2
        # so we need to return them to the original ones to continue if we dont find a solution next
        values = [x.value() for x in lY]

        # So a solution was found, try to find a better one now
        s.solution_hint(lY, values)

        if self.obj == "max":
            objective = sum([~v for v in V])

        else: # self.obj == "proba"

            # Use the counts to calculate the probability
            O_c = [1 if is_any_list(c) else 0 for c in hybridB]

            objective = sum(
                [~v * (1 - len(self.gamma) * O_c[c]) for
                 v, c in zip(V, range(len(B)))])

        # Run with the objective
        s.maximize(objective)

        flag2 = s.solve(time_limit=(time_limit - t1))

        if flag2:
            if self.debug_mode:
                print("RR2Time:", time.time() - t0, len(B), len(Cl))
            return flag2, lY
        else:
            tmp = Model()
            i = 0
            for x in lY:
                tmp += x == values[i]
                i = i + 1

            tmp.solve()
            if self.debug_mode:
                print("RR3Time:", time.time() - t0, len(B), len(Cl))
            return flag, lY

    def call_query_generation(self, answer=None):

        # Call the specified query generation method

        # Generate e in D^X accepted by C_l and rejected by B
        if self.qg == "base":
            gen_flag = self.generate_query()
            Y = self.X
        elif self.qg == "pqgen":
            gen_flag, Y = self.pqgen(time_limit=self.time_limit)
        elif self.qg == "tqgen":

            if self.metrics.queries_count > 0:
                self.l = self.adjust_tq_gen(self.l, self.alpha, answer)

            gen_flag, Y = self.tq_gen(self.alpha, self.tqgen_t, self.l)

        else:
            raise Exception("Error: No available query generator was selected!!")

        return gen_flag, Y

    def ask_query(self, value):

        if not (isinstance(value, list) or isinstance(value, set) or isinstance(value, frozenset) or
                isinstance(value, tuple)):
            Y = set()
            Y = Y.union(self.grid[value != 0])
        else:
            Y = value
            e = self.grid.value()

            # Project Y to those in kappa
            # Y = get_variables(get_kappa(self.B, Y))

            value = np.zeros(e.shape, dtype=int)

            # Create a truth table numpy array
            sel = np.array([item in set(Y) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            # Variables present in the partial query
            value[sel] = e[sel]

        # Post the query to the user/oracle
        if self.debug_mode:
            print("Y: ", Y)
            print(f"Query({self.metrics.queries_count}: is this a solution?")
            print(value)
            #print(f"Query: is this a solution?")
            #print(np.array([[v if v != 0 else -0 for v in row] for row in value]))

            print("B:", get_con_subset(self.B + toplevel_list(self.Bg),Y))
            print("violated from B: ", get_kappa(self.B + toplevel_list(self.Bg), Y))
            print("violated from C_T: ", get_kappa(self.C_T, Y))
            print("violated from C_L: ", get_kappa(self.C_l.constraints, Y))

        # Need the oracle to answer based only on the constraints with a scope that is a subset of Y
        suboracle = get_con_subset(self.C_T, Y)

        # Check if at least one constraint is violated or not
        ret = all([check_value(c) for c in suboracle])

        if self.debug_mode:
            print("Answer: ", ("Yes" if ret else "No"))

        # For the evaluation metrics

        # Increase the number of queries
        self.metrics.increase_queries_count()
        self.metrics.increase_queries_size(len(Y))

        # Measuring the waiting time of the user from the previous query
        end_time_query = time.time()
        # To measure the maximum waiting time for a query
        waiting_time = end_time_query - self.metrics.start_time_query
        self.metrics.aggreagate_max_waiting_time(waiting_time)
        self.metrics.start_time_query = time.time()  # to measure the maximum waiting time for a query

        return ret

    def genAsk(self, c, scope):
        self.metrics.increase_gen_queries_count()
        print(f"Query({self.metrics.gen_queries_count}): Can I generalize constraint {c} to all {scope}?")
        generalized = True
        for var_combination in itertools.combinations(scope, len(get_scope(c))):
            if not self.check_generalization(c, var_combination):
                generalized = False
                break

        print("Answer: ", ("Yes" if generalized else "No"))
        return generalized

    def check_constraint(self, constraint, example):
        model = Model(constraint)
        solver = SolverLookup.get(SOLVER, model)
        return solver.solve()

    def check_generalization(self, c, var_combination):
        scope = get_scope(c)
        left_vars = scope[:-1]
        right_var = scope[-1]

        if len(left_vars) != len(var_combination) - 1:
            return False

        new_left_vars = var_combination[:-1]
        new_right_var = var_combination[-1]

        if isinstance(c, Comparison):
            op = c.name
            left_expr = c.args[0]
            right_expr = c.args[1]

            if isinstance(left_expr, _IntVarImpl):
                new_left_expr = new_left_vars[0]
            else:
                new_left_expr = left_expr

            if isinstance(right_expr, _IntVarImpl):
                new_right_expr = new_right_var
            else:
                new_right_expr = right_expr

            if op == '==':
                new_constraint = (new_left_expr == new_right_expr)
            elif op == '!=':
                new_constraint = (new_left_expr != new_right_expr)
            elif op == '<':
                new_constraint = (new_left_expr < new_right_expr)
            elif op == '<=':
                new_constraint = (new_left_expr <= new_right_expr)
            elif op == '>':
                new_constraint = (new_left_expr > new_right_expr)
            elif op == '>=':
                new_constraint = (new_left_expr >= new_right_expr)
            else:
                raise ValueError(f"Unsupported comparison operator: {op}")
        else:
            raise ValueError("Unsupported constraint type")

        for example in self.C_T:
            if not self.check_constraint(new_constraint, example):
                return False

        return True
    def get_variables_from_constraint(constraint):
        if isinstance(constraint, Comparison):
            return set(get_scope(constraint))
        return set()


    def get_patterns(self):
        rows = [{f'var{i}' for i in range(j, j + 4)} for j in range(0, 16, 4)]
        columns = [{f'var{i}' for i in range(j, 16, 4)} for j in range(4)]
        blocks = [{f'var{i}' for i in [0, 1, 4, 5]},
                  {f'var{i}' for i in [2, 3, 6, 7]},
                  {f'var{i}' for i in [8, 9, 12, 13]},
                  {f'var{i}' for i in [10, 11, 14, 15]},
        return rows, columns, blocks


    def genacq(self, c, N_onTarget):
        T_able = [{var for var in self.var_names if var in get_scope(c)}]
        G = set()
        cutoffNo = float('inf')
        self.metrics.gen_no_answers = 0

        rows, columns, blocks = self.get_patterns()

        for pattern_type in [rows, columns, blocks]:
            for pattern in pattern_type:
                if self.genAsk(c, pattern):
                    G.add(frozenset(pattern))
                    for s in T_able:
                        if s.issubset(pattern):
                            T_able.remove(s)
                else:
                    for s in T_able:
                        if pattern.issubset(s):
                            T_able.remove(s)
                    self.metrics.N_egativeQ.add((frozenset(pattern), get_relation(c, self.gamma)))
                    self.metrics.gen_no_answers += 1

        return G

    def genAcq(self, con):
            cl = []
            i = 0
            while i < len(self.Bg):

                bl = self.Bg[i]

                if con not in frozenset(bl):
                    i += 1
                    continue

                self.Bg.pop(i)

                if self.genAsk(con, bl):
                    cl += bl
                    #                self.remove_from_bias(cl)
                    for c in cl:
                        self.remove_scope_from_bias(get_scope(c))
                    # self.B = list(frozenset(self.B) - frozenset(cl)) # remove only from normal B
                else:
                    self.B += bl

            [self.add_to_cl(c) for c in cl]

    # This is the version of the FindScope function that was presented in "Constraint acquisition via Partial Queries", IJCAI 2013
    def findScope(self, e, R, Y, do_ask):
        #if self.debug_mode:
            # print("\tDEBUG: findScope", e, R, Y, do_ask)
        if do_ask:
            # if ask(e_R) = yes: B \setminus K(e_R)
            # need to project 'e' down to vars in R,
            # will show '0' for None/na/" ", should create object nparray instead

            e_R = np.zeros(e.shape, dtype=int)
            sel = np.array([item in set(R) for item in list(self.grid.flatten())]).reshape(self.grid.shape)
            # if self.debug_mode:
            #    print(sel)
            if self.debug_mode and sum(sel) == 0:
                raise Exception("ERR, FindScope, Nothing to select, something went wrong...")

            self.metrics.increase_findscope_queries()

            e_R[sel] = e[sel]
            if self.ask_query(e_R):
                kappaB = get_kappa(self.B + toplevel_list(self.Bg), R)
                self.remove_from_bias(kappaB)

            else:
                return set()

        if len(Y) == 1:
            return set(Y)

        s = len(Y) // 2
        Y1, Y2 = Y[:s], Y[s:]

        S1 = self.findScope(e, R.union(Y1), Y2, True)
        S2 = self.findScope(e, R.union(S1), Y1, len(S1) > 0)

        return S1.union(S2)

    # This is the version of the FindScope function that was presented in "Constraint acquisition through Partial Queries", AIJ 2023
    def findScope2(self, e, R, Y, kappaB):

        if not frozenset(kappaB).issubset(self.B + toplevel_list(self.Bg)):
            raise Exception(f"kappaB given in findscope is not part of B: \nkappaB: {kappaB}, \nB: {self.B}")

        # if ask(e_R) = yes: B \setminus K(e_R)
        # need to project 'e' down to vars in R,
        # will show '0' for None/na/" ", should create object nparray instead
        kappaBR = get_con_subset(kappaB, list(R))
        if len(kappaBR) > 0:

            e_R = np.zeros(e.shape, dtype=int)

            sel = np.array([item in set(R) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            if self.debug_mode and sum(sel) == 0:
                raise Exception("ERR, FindScope, Nothing to select, something went wrong...")

            self.metrics.increase_findscope_queries()

            e_R[sel] = e[sel]
            if self.ask_query(e_R):
                self.remove_from_bias(kappaBR)
                self.remove(kappaB, kappaBR)
            else:
                return set()

        if len(Y) == 1:
            return set(Y)

        # Create Y1, Y2 -------------------------
        s = len(Y) // 2
        Y1, Y2 = Y[:s], Y[s:]

        S1 = set()
        S2 = set()

        # R U Y1
        RY1 = R.union(Y1)

        kappaBRY = kappaB.copy()
        kappaBRY_prev = kappaBRY.copy()
        kappaBRY1 = get_con_subset(kappaBRY, RY1)

        if len(kappaBRY1) < len(kappaBRY):
            S1 = self.findScope2(e, RY1, Y2, kappaBRY)

        # remove from original kappaB
        kappaBRY_removed = set(kappaBRY_prev) - set(kappaBRY)
        self.remove(kappaB, kappaBRY_removed)

        # R U S1
        RS1 = R.union(S1)

        kappaBRS1 = get_con_subset(kappaBRY, RS1)
        kappaBRS1Y1 = get_con_subset(kappaBRY, RS1.union(Y1))
        kappaBRS1Y1_prev = kappaBRS1Y1.copy()

        if len(kappaBRS1) < len(kappaBRY):
            S2 = self.findScope2(e, RS1, Y1, kappaBRS1Y1)

        # remove from original kappaB
        kappaBRS1Y1_removed = frozenset(kappaBRS1Y1_prev) - frozenset(kappaBRS1Y1)
        self.remove(kappaB, kappaBRS1Y1_removed)

        return S1.union(S2)

    # This is the version of the FindC function that was presented in
    # "Constraint acquisition via Partial Queries", IJCAI 2013
    def findC(self, scope, delta):
        # This function works only for normalised target networks!
        # A modification that can also learn conjunction of constraints in each scope is described in the
        # article "Partial Queries for Constraint Acquisition" that is published in AIJ !

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.C_l.constraints, scope)

        scope_values = [x.value() for x in scope]

        while True:
            # Try to generate a counter example to reduce the candidates
            flag = generate_findc_query(sub_cl, delta)

            if flag is False:
                # If no example could be generated
                # check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    print(self.B)
                    print("Collapse, the constraint we seek is not in B")
                    exit(-2)

                # FindC changes the values of the variables in the scope,
                # so we need to return them to the original ones to continue
                tmp = Model()
                i = 0
                for x in scope:
                    tmp += x == scope_values[i]
                    i = i + 1

                tmp.solve()

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            # Ask the partial counter example and update delta depending on the answer of the oracle
            e = self.grid.value()

            sel = np.array([item in set(scope) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            e_S = np.zeros(e.shape, dtype=int)
            e_S[sel] = e[sel]

            self.metrics.increase_findc_queries()

            if self.ask_query(e_S):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]

            else:  # user says UNSAT
                # delta <- K_{delta}(e)
                delta = [c for c in delta if check_value(c) is False]

    # This is the version of the FindC function that was presented in
    # "Constraint acquisition through Partial Queries", AIJ 2023
    def findC2(self, scope):
        # This function works also for non-normalised target networks!!!
        # TODO optimize to work better (probably only needs to make better the generate_find_query2)

        # Initialize delta
        delta = get_con_subset(self.B + toplevel_list(self.Bg), scope)
        delta = join_con_net(delta, [c for c in delta if check_value(c) is False])

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.C_l.constraints, scope)

        scope_values = [x.value() for x in scope]

        while True:

            # Try to generate a counter example to reduce the candidates
            if generate_findc2_query(sub_cl, delta) is False:

                # If no example could be generated
                # check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    print("Collapse, the constraint we seek is not in B")
                    exit(-2)

                # FindC changes the values of the variables in the scope,
                # so we need to return them to the original ones to continue
                tmp = Model()
                i = 0
                for x in scope:
                    tmp += x == scope_values[i]
                    i = i + 1

                tmp.solve()

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            # Ask the partial counter example and update delta depending on the answer of the oracle
            e = self.grid.value()
            sel = np.array([item in set(scope) for item in list(self.grid.flatten())]).reshape(self.grid.shape)

            e_S = np.zeros(e.shape, dtype=int)
            e_S[sel] = e[sel]

            self.metrics.findc_queries()

            if self.ask_query(e_S):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]
            else:  # user says UNSAT
                # delta <- joint(delta,K_{delta}(e))

                kappaD = [c for c in delta if check_value(c) is False]

                scope2 = self.call_findscope(list(scope), kappaD)

                if len(scope2) < len(scope):
                    self.call_findc(scope2)
                else:
                    delta = join_con_net(delta, kappaD)