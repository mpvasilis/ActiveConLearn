import os

def list_experiments(directory):
    base_command = "python main.py -a mquacq2-a -b vgc -qg pqgen"
    output_path = "results"
    use_con = "True"

    try:
        directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    except FileNotFoundError:
        print(f"The directory {directory} was not found.")
        return

    for dir_name in directories:
        input_path = os.path.join(directory, dir_name)
        command = f"{base_command} -exp {dir_name} -i {input_path}/ --output {output_path} --useCon {use_con}"
        print(command)

directory = "exps/gts"
list_experiments(directory)
