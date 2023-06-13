import os
import json
import torch
import pickle
import fnmatch
import pandas as pd

from tailnflows.models.flow_models import get_model
from tailnflows.targets import targets

def new_experiment(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
    id_path = f'{path}/experiment_id.pkl'
    if not os.path.exists(id_path):
        with open(id_path, 'wb+') as f:
            pickle.dump(0, f)
    
    with open(id_path, 'rb+') as f:
        experiment_id = pickle.load(f) + 1

    with open(id_path, 'wb+') as f:
        pickle.dump(experiment_id, f)
    
    return experiment_id

def write_experiment_details(path, details):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/details.txt', 'w') as file:
        file.write(json.dumps(details, indent=4))
    
def load_experiment_details(path):
    details = None
    with open(f'{path}/details.txt', 'r') as file:
        details = json.loads(file.read())
    return details

def save_model(path, model):
    torch.save(model.state_dict(), f'{path}/trained.model')

def load_model(details):
    path = details['path']
    sample_and_log_prob, model = get_model(details['model'], details['dim'])
    model.load_state_dict(torch.load(f'{path}/trained.model'))
    return sample_and_log_prob, model

def load_experiment_data():
    path = f'{get_project_root()}/experiment_output'
    experiments = [
        load_experiment_details(dirpath)
        for dirpath, _, files in os.walk(path)
        for f in fnmatch.filter(files, 'training_data.npy') # ensure training complete
    ]
    experiment_data = pd.DataFrame(experiments)

    models = {}
    for ix, details in experiment_data.iterrows():
        _, dim, _ = targets[details['target']](details['target_kwargs'])
        details['dim'] = dim
        models[ix] = load_model(details)

    return experiment_data, models