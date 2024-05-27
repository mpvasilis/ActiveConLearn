import time
from statistics import mean, stdev

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

    def __init__(self, gamma, grid, ct=list(), B=list(), Bg=None, X=set(), C_l=None, qg="pqgen", obj="proba",
                 time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000):

        self.debug_mode = False

        # Target network
        self.C_T = ct

        self.grid = grid
        self.gamma = gamma
        if B is None:
            B = []
        self.B = B
        if Bg is None:
            Bg = []
        self.Bg = Bg

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
            if any(c in set(bl) for c in C):
                self.B.extend(bl)
                self.Bg.pop(i)
                i = i-1
            i +=1

        self.B = list(set(self.B))

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
#        learned_con_rel = get_relation(self.C_l.constraints[-1], self.gamma)
        B_scopes_set = [set(get_scope(c)) for c in self.B + toplevel_list(self.Bg)]

        removing = [c for i, c in enumerate(self.B + toplevel_list(self.Bg)) if B_scopes_set[i] == scope_set]
#                    and get_relation(c, self.gamma) != learned_con_rel]

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
        Y = frozenset(get_variables(self.B + toplevel_list(self.Bg)))
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
        flag = s.solve()

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

    def genAsk(self, c, bl):

        self.metrics.increase_gen_queries_count()
        print(f"Query({self.metrics.gen_queries_count}: Can I generalize constraint {c} to all {bl}?")

        ret = all(c in set(self.C_T) for c in bl)
        print("Answer: ", ("Yes" if ret else "No"))

        return ret

    def genAcq(self, con):

        cl = []
        i = 0
        while i < len(self.Bg):

            bl = self.Bg[i]

            # skip the lists not including the constraint on hand
            if con not in set(bl):
                i += 1
                continue

            # remove from Bg as we are going to ask a query for it!
            self.Bg.pop(i)

            if self.genAsk(con, bl):
                cl += bl
                self.B = list(set(self.B) - set(cl)) # remove only from normal B
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
        kappaBRS1Y1_removed = set(kappaBRS1Y1_prev) - set(kappaBRS1Y1)
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




import time

import numpy as np
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list

from ConAcq import ConAcq
from utils import construct_bias, get_kappa, get_scope, get_relation

cliques_cutoff = 0.5

class MQuAcq2(ConAcq):
    def __init__(self, gamma, grid, ct=list(), B=list(), Bg=list(), X=set(), C_l=set(), qg="pqgen", obj="proba",
                 time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000, perform_analyzeAndLearn=False):
        super().__init__(gamma, grid, ct, B, Bg, X, C_l, qg, obj, time_limit, findscope_version,
                    findc_version, tqgen_t, qgen_blimit)
        self.perform_analyzeAndLearn = perform_analyzeAndLearn

    def learn(self):

        answer = True

        if len(self.B + toplevel_list(self.Bg)) == 0:
            self.B = construct_bias(self.X, self.gamma)

        while True:
            if self.debug_mode:
                print("Size of CL: ", len(self.C_l.constraints))
                print("Size of B: ", len(self.B + toplevel_list(self.Bg)))
                print("Number of queries: ", self.metrics.queries_count)
                print("MQuAcq-2 Queries: ", self.metrics.top_lvl_queries)
                print("FindScope Queries: ", self.metrics.findscope_queries)
                print("FindC Queries: ", self.metrics.findc_queries)
            gen_start = time.time()

            gen_flag, Y = self.call_query_generation(answer)

            gen_end = time.time()

            if not gen_flag:
                # if no query can be generated it means we have converged to the target network -----
                break

            kappaB = get_kappa(self.B + toplevel_list(self.Bg), Y)
            Yprime = Y.copy()
            answer = True

            self.metrics.increase_generation_time(gen_end - gen_start)
            self.metrics.increase_generated_queries()

            while len(kappaB) > 0:

                self.metrics.increase_top_queries()

                if self.ask_query(Yprime):
                    # it is a solution, so all candidates violated must go
                    # B <- B \setminus K_B(e)
                    self.remove_from_bias(kappaB)
                    kappaB = set()

                else:  # user says UNSAT

                    answer = False

                    scope = self.call_findscope(Yprime, kappaB)
                    self.call_findc(scope)
                    NScopes = set()
                    NScopes.add(tuple(scope))

                    if self.perform_analyzeAndLearn:
                        NScopes = NScopes.union(self.analyze_and_learn(Y))

                    Yprime = [y2 for y2 in Yprime if not any(y2 in set(nscope) for nscope in NScopes)]

                    kappaB = get_kappa(self.B + toplevel_list(self.Bg), Yprime)

    def analyze_and_learn(self, Y):

        NScopes = set()
        QCliques = set()

        # Find all neighbours (not just a specific type)
        self.cl_neighbours = self.get_neighbours(self.C_l.constraints)

        # Gamma precentage in FindQCliques is set to 0.8
        self.find_QCliques(self.X.copy(), set(), set(), QCliques, 0.8, 0)

        # [self.isQClique(clique, 0.8) for clique in QCliques]

        cliques_relations = self.QCliques_relations(QCliques)

        # Find the scopes that have a constraint in B violated, which can fill the incomplete cliques
        if len(QCliques) == 0:
            return set()

        Cq = [c for c in get_kappa(self.B + toplevel_list(self.Bg), Y) if any(
            set(get_scope(c)).issubset(clique) and get_relation(c, self.gamma) in cliques_relations[i] for i, clique in
            enumerate(QCliques))]

        PScopes = {tuple(get_scope(c)) for c in Cq}

        for pscope in PScopes:

            if self.ask_query(pscope):
                # It is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                kappaB = get_kappa(self.B + toplevel_list(self.Bg), pscope)
                self.remove_from_bias(kappaB)

            else:  # User says UNSAT

                # c <- findC(e, findScope(e, {}, grid, false))
                c = self.call_findc(pscope)

                NScopes.add(tuple(pscope))

        if len(NScopes) > 0:
            NScopes = NScopes.union(self.analyze_and_learn(Y))

        return NScopes

    def get_neighbours(self, C, type=None):

        # In case a model is given in the function instead of a list of constraints
        if not (isinstance(C, list) or isinstance(C, set)):
            C = C.constraints

        neighbours = np.zeros((len(self.X), len(self.X)), dtype=bool)

        for c in C:

            flag = False

            if type is not None:
                if self.gamma[get_relation(c, self.gamma)] == type:
                    flag = True
            else:
                flag = True

            if flag:
                scope = get_scope(c)

                i = self.hashX.index(hash(scope[0]))
                j = self.hashX.index(hash(scope[1]))

                neighbours[i][j] = True
                neighbours[j][i] = True

        return neighbours

    def QCliques_relations(self, QCliques):

        cl_relations = [get_relation(c, self.gamma) for c in self.C_l.constraints]
        cliques_relations = [[rel for i, rel in enumerate(cl_relations)
                              if set(get_scope(self.C_l.constraints[i])).issubset(clique)] for clique in QCliques]

        return cliques_relations

    # For debugging
    def is_QClique(self, clique, gammaPerc):

        edges = 0

        q = len(clique)
        q = gammaPerc * (q * (q - 1) / 2)  # number of edges needed to be considered a quasi-clique

        for var1, var2 in all_pairs(clique):
            k = self.hashX.index(hash(var1))
            l = self.hashX.index(hash(var2))

            if self.cl_neighbours[k, l]:
                edges = edges + 1

        if edges < q:
            raise Exception(
                f'findQCliques returned a clique that is not a quasi clique!!!! -> {clique} \nedges = {edges}\nq = {q}')


    def find_QCliques(self, A, B, K, QCliques, gammaPerc, t):
        """
            Find quasi cliques

            A: a mutable list of all variables (nodes in the graph)
            gammaPerc: percentage of neighbors to be considered a quasi clique
            t: total time counter
        """
        global cliques_cutoff

        start = time.time()

        if len(A) == 0 and len(K) > 2:
            if not any(K.issubset(set(clique)) for clique in QCliques):
                QCliques.add(tuple(K))

        while len(A) > 0:

            end = time.time()
            t = t + end - start
            start = time.time()

            if t > cliques_cutoff:
                return

            x = A.pop()

            K2 = K.copy()
            K2.add(x)

            A2 = set(self.X) - K2 - B
            A3 = set()

            # calculate the number of existing edges on K2
            edges = 0
            for var1, var2 in all_pairs(K2):
                k = self.hashX.index(hash(var1))
                l = self.hashX.index(hash(var2))

                if self.cl_neighbours[k, l]:
                    edges = edges + 1

            q = len(K2) + 1
            q = gammaPerc * (q * (q - 1) / 2)  # number of edges needed to be considered a quasi-clique

            # for every y in A2, check if K2 U y is a gamma-clique (if so, store in A3)
            for y in list(A2):  # take (yet another) copy

                edges_with_y = edges

                # calculate the number of from y to K2
                for var in K2:

                    k = self.hashX.index(hash(var))
                    l = self.hashX.index(hash(y))

                    if self.cl_neighbours[k, l]:
                        edges_with_y = edges_with_y + 1

                if edges_with_y >= q:
                    A3.add(y)

            self.find_QCliques(A3, B.copy(), K2.copy(), QCliques, gammaPerc, t)

            B.add(x)

import math
import time

from cpmpy import intvar, boolvar, Model, all, sum, SolverLookup
from sklearn.utils import class_weight
import numpy as np

import cpmpy
import re
from cpmpy.expressions.utils import all_pairs
from itertools import chain

from ConAcq import SOLVER

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


def get_scope(constraint):
    # this code is much more dangerous/too few cases then get_variables()
    if isinstance(constraint, cpmpy.expressions.variables._IntVarImpl):
        return [constraint]
    elif isinstance(constraint, cpmpy.expressions.core.Expression):
        all_variables = []
        for argument in constraint.args:
            if isinstance(argument, cpmpy.expressions.variables._IntVarImpl):
                # non-recursive shortcut
                all_variables.append(argument)
            else:
                all_variables.extend(get_scope(argument))
        return all_variables
    else:
        return []


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


import json
import os
import argparse
import subprocess

from QuAcq import QuAcq
from MQuAcq import MQuAcq
from MQuAcq2 import MQuAcq2
from GrowAcq import GrowAcq
from benchmarks import *
from utils import *

jar_path = './phD.jar'
output_directory = './results'


def parse_args():
    parser = argparse.ArgumentParser()

    # Parsing algorithm
    parser.add_argument("-a", "--algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a", "growacq"],
                        required=True,
                        help="The name of the algorithm to use")
    # Parsing specific to GrowAcq
    parser.add_argument("-ia", "--inner-algorithm", type=str, choices=["quacq", "mquacq", "mquacq2", "mquacq2-a"],
                        required=False,
                        help="Only relevant when the chosen algorithm is GrowAcq - "
                             "the name of the inner algorithm to use")

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
                                 "nurse_rostering_advanced", "nurse_rostering_adv", "custom", "vgc"],
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
    parser.add_argument("-oa", "--onlyActive", type=bool, required=False,
                        help="Run a custom model with only active learning - don't use the Passive Learning CL and bias")
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
    parser.add_argument("-pl", "--run-passive-learning", required=False, type=bool, help="Run passive learning")
    parser.add_argument("-sols", "--solution-set-path", type=str, required=False,
                        help="Path to the solution set JSON file")

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
            constraint = eval(f"variables[var1] {con_str} variables[var2]")
            constraints.append(constraint)
            if model is not None:
                model += constraint
        return constraints

    model = Model()
    vars_file = f"{data_dir}/{experiment}_var"
    vars = parse_vars_file(vars_file)
    dom_file = f"{data_dir}/{experiment}_dom"
    domain_constraints = parse_dom_file(dom_file)
    variables = [intvar(domain_constraints[0][0], domain_constraints[0][1], name=f"var{var}") for var in vars]
    grid = intvar(domain_constraints[0][0], domain_constraints[0][1], shape=(1, len(variables)), name="grid")
    for i, var in enumerate(variables):
        grid[1:i] = var

    if use_learned_model:
        model_file = f"{data_dir}/{experiment}_model"
        parsed_constraints, max_index = parse_model_file(model_file)
        for constraint_type, indices in parsed_constraints:
            if constraint_type == 'ALLDIFFERENT':
                model += AllDifferent([variables[i] for i in indices])

    if args.useCon:
        con_file = f"{data_dir}/{experiment}_con"
        fixed_arity_ct = parse_and_apply_constraints(con_file, variables, model)

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

    return grid, C_T, model, variables, biases, cls


def verify_global_constraints(experiment, data_dir="data/exp", use_learned_model=False):
    biasg = []

    def parse_and_apply_constraints(file_path, variables, model=None):
        parsed_data = parse_con_file(file_path)
        constraints = []
        for con_type, var1, var2 in parsed_data:
            con_str = constraint_type_to_string(con_type)
            constraint = eval(f"variables[var1] {con_str} variables[var2]")
            constraints.append(constraint)
            if model is not None:
                model += constraint
        return constraints

    model = Model()
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
    for constraint_type, indices in parsed_constraints:
        if constraint_type == 'ALLDIFFERENT':
            if use_learned_model:
                model += AllDifferent([variables[i] for i in indices])
            biasg.append(AllDifferent([variables[i] for i in indices]).decompose()[0])

    if args.useCon:
        con_file = f"{data_dir}/{experiment}_con"
        fixed_arity_ct = parse_and_apply_constraints(con_file, variables, model)

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

    return grid, C_T, model, variables, biases, biasg, cls


def construct_benchmark():
    if args.benchmark == "9sudoku":
        grid, C_T, oracle = construct_9sudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "4sudoku":
        grid, C_T, oracle = construct_4sudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "jsudoku":
        grid, C_T, oracle = construct_jsudoku()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "random122":
        grid, C_T, oracle = construct_random122()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "new_random":
        grid, C_T, oracle = construct_new_random()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "random495":
        grid, C_T, oracle = construct_random495()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "golomb8":
        grid, C_T, oracle = construct_golomb8()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2",
                 "abs(var1 - var2) != abs(var3 - var4)"]
    #            "abs(var1 - var2) == abs(var3 - var4)"]

    elif args.benchmark == "murder":
        grid, C_T, oracle = construct_murder_problem()
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "job_shop_scheduling":
        grid, C_T, oracle, max_duration = construct_job_shop_scheduling_problem(args.num_jobs, args.num_machines,
                                                                                args.horizon, args.seed)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"var1 + {i} == var2" for i in range(1, max_duration + 1)] + \
                [f"var2 + {i} == var1" for i in range(1, max_duration + 1)]

    elif args.benchmark == "exam_timetabling" or args.benchmark == "exam_timetabling_simple":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T, oracle = construct_examtt_simple(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                    args.num_timeslots_per_day, args.num_days_for_exams)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})",
                 f"(var1 // {slots_per_day}) == (var2 // {slots_per_day})"]

    elif args.benchmark == "exam_timetabling_adv" or args.benchmark == "exam_timetabling_advanced":
        slots_per_day = args.num_rooms * args.num_timeslots_per_day

        grid, C_T, oracle = construct_examtt_advanced(args.num_semesters, args.num_courses_per_semester, args.num_rooms,
                                                      args.num_timeslots_per_day, args.num_days_for_exams,
                                                      args.num_professors)
        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"] + \
                [f"abs(var1 - var2) // {slots_per_day} <= 2"] + \
                [f"(var1 // {slots_per_day}) != (var2 // {slots_per_day})"]
        # [f"var1 // {slots_per_day} != {d}" for d in range(num_days_for_exams)]
    elif args.benchmark == "nurse_rostering":

        grid, C_T, oracle = construct_nurse_rostering(args.num_nurses, args.num_shifts_per_day,
                                                      args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    elif args.benchmark == "nurse_rostering_adv" or args.benchmark == "nurse_rostering_advanced":

        grid, C_T, oracle = construct_nurse_rostering_advanced(args.num_nurses, args.num_shifts_per_day,
                                                               args.nurses_per_shift, args.num_days_for_schedule)

        gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    else:
        raise NotImplementedError(f'Benchmark {args.benchmark} not implemented yet')

    return args.benchmark, grid, C_T, oracle, gamma


def save_results(alg=None, inner_alg=None, qg=None, tl=None, t=None, blimit=None, fs=None, fc=None, bench=None,
                 start_time=None, conacq=None):
    if conacq is None: conacq = ca_system
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

    print("C_L size: ", len(toplevel_list(conacq.C_l.constraints)))

    res_name = ["results"]
    res_name.append(alg)

    # results_file = "results/results_" + args.algorithm + "_"
    if alg == "growacq":
        if inner_alg is None: inner_alg = args.inner_algorithm
        res_name.append(inner_alg)
        # results_file += args.inner_algorithm + "_"

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
        results_file = args.output + "/mquack2_results_" + args.experiment
    else:
        results_file = "_".join(res_name)

    file_exists = os.path.isfile(results_file)
    f = open(results_file, "a")

    if not file_exists:
        results = "CL\tTot_q\ttop_lvl_q\tgen_q\tfs_q\tfc_q\tavg|q|\tgen_time\tavg_t\tmax_t\ttot_t\tconv\n"
    else:
        results = ""

    results += str(len(toplevel_list(conacq.C_l.constraints))) + "\t" + str(conacq.metrics.queries_count) + "\t" + str(
        conacq.metrics.top_lvl_queries) \
               + "\t" + str(conacq.metrics.generated_queries) + "\t" + str(
        conacq.metrics.findscope_queries) + "\t" + str(
        conacq.metrics.findc_queries)

    avg_size = round(conacq.metrics.average_size_queries / conacq.metrics.queries_count,
                     4) if conacq.metrics.queries_count > 0 else 0

    avg_qgen_time = round(conacq.metrics.generation_time / conacq.metrics.generated_queries,
                          4) if conacq.metrics.generated_queries > 0 else 0
    results += "\t" + str(avg_size) + "\t" + str(avg_qgen_time) \
               + "\t" + str(round(average_waiting_time, 4)) + "\t" + str(
        round(conacq.metrics.max_waiting_time, 4)) + "\t" + \
               str(round(total_time, 4))

    results += "\t" + str(conacq.metrics.converged) + "\n"

    f.write(results)
    f.close()


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
        'activeLearning': False,
        'constraintsToCheck': [
            "allDifferent"
        ],
        'decreasingLearning': False,
        'numberOfSolutionsForDecreasingLearning': 0,
        'enableSolutionGeneratorForActiveLearning': True,
        'plotChart': False,
        'validateConstraints': True,
        'mQuack2MaxIterations': 1,
        'mQuack2SatisfyWithChoco': False,
        'runTestCases': True,
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

    if args.benchmark == "vgc":  # verify global constraints - genacq
        if args.run_passive_learning:
            run_passive_learning_with_jar(jar_path, args.solution_set_path, output_directory)

        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, biasg, C_l = verify_global_constraints(benchmark_name, path, args.use_learned_model)
        print("Size of bias: ", len(bias))
        print("Size of biasg: ", len(biasg))
        print("Size of C_l: ", len(C_l))
        ca_system = MQuAcq2(gamma, grid, C_T, qg="pqgen", obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, X=X, B=bias, Bg=biasg, C_l=C_l)
        ca_system.learn()

        save_results()
        exit()

    if args.benchmark == "custom":  # mquack2 custom problem
        if args.run_passive_learning:
            run_passive_learning_with_jar(jar_path, args.solution_set_path, output_directory)

        benchmark_name = args.experiment
        path = args.input
        grid, C_T, oracle, X, bias, C_l = construct_custom(benchmark_name, path, args.use_learned_model)
    else:
        benchmark_name, grid, C_T, oracle, gamma = construct_benchmark()
    grid.clear()

    print("Size of C_T: ", len(C_T))

    all_cons = []
    X = list(list(grid.flatten()))
    for relation in gamma:
        if relation.count("var") == 2:
            for v1, v2 in all_pairs(X):
                print(v1, v2)
                constraint = relation.replace("var1", "v1")
                constraint = constraint.replace("var2", "v2")
                constraint = eval(constraint)
                all_cons.append(constraint)

    bias = all_cons
    bias.pop()
    C_l = [all_cons[i] for i in range(0, len(all_cons), 2)]
    C_l = [all_cons[-1]]

    bias_filtered = []
    for item_cl in C_l:
        for item_bias in bias:
            if not are_comparisons_equal(item_cl, item_bias):
                bias_filtered.append(item_cl)
    bias = bias_filtered

    if args.benchmark == "custom":
        print("Size of bias: ", len(bias))
        print(bias)
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, B=bias, Bg=bias, C_l=C_l, X=X)
        ca_system.learn()

        save_results()
        exit()

    if args.algorithm == "quacq":
        ca_system = QuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                          time_limit=args.time_limit, findscope_version=fs_version,
                          findc_version=fc_version)
    elif args.algorithm == "mquacq":
        ca_system = MQuAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                           time_limit=args.time_limit, findscope_version=fs_version,
                           findc_version=fc_version)
    elif args.algorithm == "mquacq2":
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version)
    elif args.algorithm == "mquacq2-a":
        ca_system = MQuAcq2(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version, perform_analyzeAndLearn=True)
    elif args.algorithm == "growacq":
        ca_system = GrowAcq(gamma, grid, C_T, qg=args.query_generation, obj=args.objective,
                            time_limit=args.time_limit, findscope_version=fs_version,
                            findc_version=fc_version)

    ca_system.learn()

    save_results()
