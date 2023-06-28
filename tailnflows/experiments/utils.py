import os
import json
import torch
import pickle
import fnmatch
import pandas as pd
import tqdm

from tailnflows.models.density_estimation import get_model as get_de_model
from tailnflows.models.flow_models import get_model
from tailnflows.targets import targets
from tailnflows.data import data_sources
from tailnflows.utils import get_project_root

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

def load_model(details, vi=True):
    if vi:
        path = details['path']
        sample_and_log_prob, model = get_model(details['model'], details['dim'])
        model.load_state_dict(torch.load(f'{path}/trained.model'))
        return sample_and_log_prob, model
    else:
        path = details['disk_path']
        model = get_de_model(torch.float64, details['model'], details['dim'], details['model_kwargs'])
        model.load_state_dict(torch.load(f'{path}/trained.model', map_location=torch.device('cpu') ))
        return model

def load_experiment_data(target_dir, load_models=False, filter=None):
    path = f'{get_project_root()}/experiment_output/{target_dir}'
    experiments = [
        {**load_experiment_details(dirpath), 'disk_path': dirpath}
        for dirpath, _, files in os.walk(path)
        for f in fnmatch.filter(files, 'training_data.npy') # ensure training complete
    ]
    if filter is not None:
        experiments = [exp for exp in experiments if filter(exp)]

    experiment_data = pd.DataFrame(experiments)

    if not load_models:
        return experiment_data, None, []
    
    models = {}
    failed = [] # usually due to version mismatch
    for ix, details in tqdm.tqdm(list(experiment_data.iterrows())):
        if details['target'] in targets:
            models[ix] = load_model(details)
        elif details['target'] in data_sources:
            try:
                models[ix] = load_model(details, vi=False)
            except:
                failed.append(details)

    return experiment_data, models, failed