import os
import json

def run_finetuning(params):
    os.system('mkdir checkpoints')
    params_string = []
    for el in params:
        if isinstance(params[el], list) or isinstance(params[el], dict):
            params_string.append(f"{el}='{params[el]}'")
        else:
            params_string.append(f"{el}={params[el]}")
    params_string = ' '.join(params_string)
    command = f'python3 train.py with {params_string}'
    print(command)
    os.system(command)

configs = json.load(open('configs.json'))

run_finetuning(configs)
