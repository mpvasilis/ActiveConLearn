from itertools import product

class GENACQ:
    def __init__(self, variable_types):
        self.variable_types = variable_types  # Dictionary mapping variables to their types
        self.negativeQ = set()  # To store negative generalization queries

    def initialize_table(self, learned_constraint):
        scope = learned_constraint.scope  # The scope of the learned constraint
        type_combinations = self.get_type_combinations(scope)
        return set(type_combinations)

    def get_type_combinations(self, scope):
        # Get the types for each variable in the scope
        variable_types_list = [self.variable_types[var] for var in scope]
        # Generate all combinations of these types
        type_combinations = product(*variable_types_list)
        return type_combinations

    def ask_gen(self, sequence, relation):

        print(f"Do all variables in the sequence {sequence} satisfy the relation {relation}? (yes/no)")
        response = input().strip().lower()
        return response == "yes"

    def genacq(self, learned_constraint, non_target):
        scope = learned_constraint.scope
        relation = learned_constraint.relation
        table = self.initialize_table(learned_constraint)
        generalizations = set()
        no_answers = 0

        while table and no_answers < float('inf'):  # Using infinity as default cutoff
            sequence = table.pop()
            if self.ask_gen(sequence, relation):
                generalizations.add(sequence)
                table = {s for s in table if not self.covers(s, sequence)}
                no_answers = 0
            else:
                table = {s for s in table if not self.covers(sequence, s)}
                self.negativeQ.add((sequence, relation))
                no_answers += 1

        return generalizations

    def covers(self, s1, s2):
        # Check if sequence s1 covers sequence s2
        return all(a == b or b in a for a, b in zip(s1, s2))
