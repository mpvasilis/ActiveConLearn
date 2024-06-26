import glob
import json

from cpmpy import *
from learn import learn
from instance import Instance

"""
    Magic Square, generated by Tias
"""


def model_type21(instance: Instance):
    square = instance.cp_vars['square']
    N = instance.constants['size']

    # latin square has rows/cols permutations (alldifferent)
    def latin_sq(square):
        return [[AllDifferent(row) for row in square],
                [AllDifferent(col) for col in square.T]]

    model = Model()
    # each is a latin square
    model += latin_sq(square)

    return model


if __name__ == "__main__":
    print("Learned model")
    # from experiments.py
    t = 21
    path = f"type{t:02d}/inst*.json"
    files = sorted(glob.glob(path))
    instances = []
    for file in files:
        with open(file) as f:
            instances.append(Instance(int(file.split("/")[-1].split(".")[0][8:]), json.load(f), t))

    bounding_expressions = learn(instances)
    for k, v in bounding_expressions.items():
        print(k, v)


    print("Ground-truth model (Latin Sq)")
    inst = instances[0]
    print("vars:", inst.cp_vars)
    print("data:", inst.input_data)
    print("constants:", inst.constants)
    m = model_type21(inst)
    print(m)

    # sanity check ground truth
    for i,inst in enumerate(instances):
        if inst.has_solutions():
            print(i, inst.check(model_type21(inst)))
