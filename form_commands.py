from itertools import product

FILE = 'commands.txt'
EXPERIMENT_NAME = 'brute-force'
TRAIN_FILE = 'train_model.py'

options = {
    '--model': ("convnext_tiny", "convnext_small", "convnext_base"),
    '--augmentation-type': ("auto", "custom", "none"),
    '--num-trainable-layers': (0, 3, 5, 10),
    '--use-weighted-sampling': ("true", "false"),
    '--learning-rate': (0.001, 0.0008, 0.0005),
}

with open(FILE, 'w') as f:
    for args in product(*options.values()):
        command = f'python3 {TRAIN_FILE} --experiment-name {EXPERIMENT_NAME}'
        for key, value in zip(options.keys(), args):
            command += f' {key} {value}'
        f.write(command + '\n')