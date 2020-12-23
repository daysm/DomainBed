import sys
import time
import os
import signal
import pprint
import json
import shlex
import subprocess
from os import path

class ExitSignalHandler:
    def __init__(self):
        self.exit_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.exit_now = True


def write_failure_file(failure_file_path, failure_reason):
    failure_file = open(failure_file_path, 'w')
    failure_file.write(failure_reason)
    failure_file.close()
    
def save_model_artifacts(model_artifacts_path, net):
    if (path.exists(model_artifacts_path)):
        model_file = open(model_artifacts_path + 'model.dummy', 'w')
        model_file.write("Dummy model.")
        model_file.close()
    
def print_json_object(json_object):
    pprint.pprint(json_object)
        
def load_json_object(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)
    
def print_files_in_path(path):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    for f in files:
        print(f)

hyperparameters_file_path = "/opt/ml/input/config/hyperparameters.json"
inputdataconfig_file_path = "/opt/ml/input/config/inputdataconfig.json"
resource_file_path = "/opt/ml/input/config/resourceconfig.json"
data_files_path = "/opt/ml/input/data/"
failure_file_path = "/opt/ml/output/failure"
model_artifacts_path = "/opt/ml/model/"

training_job_name_env = "TRAINING_JOB_NAME"
training_job_arn_env = "TRAINING_JOB_ARN"

def train():
    try:
        print("\nRunning training...")
        
        train_args = {"data_dir": "data"}
        if os.path.exists(hyperparameters_file_path):
            train_args = load_json_object(hyperparameters_file_path)
            print('\nTrain args:')
            print_json_object(train_args)
            
        command = ['python', '-m', 'domainbed.scripts.train', '--data_dir='+data_files_path]
        
        for k, v in sorted(train_args.items()):
            if isinstance(v, str) and not v:
                v = ''
            command.append(f'--{k} {v}')
            
        command_str = ' '.join(command)
        subprocess.call(command_str, shell=True)
            
        
        if os.path.exists(inputdataconfig_file_path):
            input_data_config = load_json_object(inputdataconfig_file_path)
            print('\nInput data configuration:')
            print_json_object(input_data_config)
        
            
        if (training_job_name_env in os.environ):
            print("\nTraining job name: ")
            print(os.environ[training_job_name_env])
        
        if (training_job_arn_env in os.environ):
            print("\nTraining job ARN: ")
            print(os.environ[training_job_arn_env])
            
        # This object is used to handle SIGTERM and SIGKILL signals.
        signal_handler = ExitSignalHandler()
        
        
        
        print("\nTraining completed!")
        
    except Exception as e:
        write_failure_file(failure_file_path, str(e))
        print(e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if (sys.argv[1] == "train"):
        train()
    else:
        print("Missing required argument 'train'.", file=sys.stderr)
        sys.exit(1)