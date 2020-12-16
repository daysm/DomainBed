# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import sagemaker
from sagemaker.estimator import Estimator

def sagemaker_launcher(all_train_args, local=False):
    """Launch commands on SageMaker."""
    # role = sagemaker.get_execution_role()
    local_image_uri = 'domainbed'
    remote_image_uri = '302710561802.dkr.ecr.us-east-2.amazonaws.com/domainbed'
    image_uri = local_image_uri if local else remote_image_uri

    local_output_dir = 'local_sweep'
    remote_output_dir = 's3://sagemaker-us-east-2-302710561802/d073679/runs/'
    output_dir = local_output_dir if local else remote_output_dir
    
    local_instance_type = 'local_gpu'
    remote_instance_type = 'ml.g4dn.2xlarge'
    instance_type = local_instance_type if local else remote_instance_type

    
    for train_args in all_train_args:       
        estimator = Estimator(image_uri=image_uri,
                              output_dir=output_dir,
                              role=None,
                              instance_count=1,
                              instance_type=instance_type,
                              hyperparameters=train_args)

        remote_data_uri = 's3://sagemaker-us-east-2-302710561802/d073679/data/DaimlerV2/'
        local_data_uri = '../data/DaimlerV2/'
        data_uri = local_data_uri if local else remote_data_uri
        estimator.fit({'DaimlerV2': data_uri}, wait=True if local else False)


def sagemaker_local_launcher(all_train_args):
    sagemaker_launcher(all_train_args, local=True)

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
    'sagemaker': sagemaker_launcher,
    'sagemaker_local': sagemaker_local_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
