import os
import sys


def init_paths(project_name=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    python_paths = [
        '../../../',
        '../../../diffusers_20240908/',
    ]
    if project_name is not None:
        python_paths.append(os.path.join(cur_dir, project_name))
    for python_path in python_paths:
        sys.path.insert(0, python_path)
        if 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] += ':{}'.format(python_path)
        else:
            os.environ['PYTHONPATH'] = python_path
init_paths()

def train_vast_i2v():

    config_path = 'scripts.examples.vastbase.configs.vastbase.config'
    runners = ['scripts.examples.vastbase.adaptors.VASTBASETrainer']

    from vastbase.train import launch_from_config

    launch_from_config(config_path, ','.join(runners))
    # launch_from_config(config_path, ','.join(runners), gpu_memory=78000, seconds=60)


if __name__ == '__main__':
    train_vast_i2v()
