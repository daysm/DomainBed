# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import sagemaker
from sagemaker.estimator import Estimator

def sagemaker_launcher(all_train_args):
    """Launch commands on SageMaker."""
    role = sagemaker.get_execution_role()
    for train_args in all_train_args:       
        estimator = Estimator(image_uri='domainbed',
                              output_dir='s3://sagemaker-us-east-2-302710561802/d073679/runs/',
                              role=role,
                              instance_count=1,
                              instance_type='local_gpu',
                              hyperparameters=train_args)

        data_uri = 'file://../data/DaimlerV2/'
        estimator.fit({'DaimlerV2': data_uri}, wait=False)

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'sagemaker': sagemaker_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
