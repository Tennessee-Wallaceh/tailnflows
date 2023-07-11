#type: ignore
import datetime
import numpy as np
import tqdm
import argparse
from functools import partial
import matplotlib.pyplot as plt
import logging

# pytorch
import torch
from torch.optim import Adam

# for hyperparameter tune
import ray
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

# our modules
from tailnflows.models.density_estimation import get_model
from tailnflows.data import data_sources
from tailnflows.experiments.utils import new_experiment, write_experiment_details, save_model
from tailnflows.utils import get_project_root
from tailnflows.models.extreme_transformations import (
    MaskedExtremeAutoregressiveTransform,
    MaskedTailAutoregressiveTransform,
)

precision_types = {
    'float64': torch.float64,
    'float32': torch.float32,
}

def run_experiment(
        seed, 
        target_name,
        target_kwargs,
        lr, 
        num_epochs, 
        batch_size, 
        model_name,
        model_kwargs,
        precision='float64',
        device='cpu'
    ):
    
    # config
    data_seed = seed + 1000 # just start as shift of global seed
    torch.set_default_device(device)
    dtype = precision_types[precision]
    torch.set_default_dtype(dtype)

    # setup target data
    x_trn, x_val, x_tst, dim, _ = data_sources[target_name](dtype, data_seed, standardise=True, **target_kwargs)

    # record experiment details
    base_path = f'{get_project_root()}/experiment_output/{target_name}'
    experiment_id = new_experiment(base_path)
    subfolder = f'id_{experiment_id}|seed_{seed}|num_epochs_{num_epochs}'
    path = f'{base_path}/{subfolder}'
    print(f'targetting {target_name} with {model_name} flow, params:[{subfolder}]...')
    
    details = {
        'seed': seed,
        'lr': lr,
        'target': target_name,
        'target_kwargs': target_kwargs,
        'num_epochs': num_epochs, 
        'batch_size': batch_size, 
        'model': model_name,
        'model_kwargs': model_kwargs,
        'run_time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
        'path': path,
        'dim': dim,
        'device': device
    }
    write_experiment_details(path, details)

    def _train(config, dtype, model_name, dim, model_kwargs, verbose=False):
        torch.manual_seed(seed)

        # setup model
        model = get_model(dtype, model_name, dim, {**config, **model_kwargs}).to(device)
        param_groups = [{'params': model._distribution.parameters()}]
        for layer in model._transform._transforms:
            if (
                isinstance(layer, MaskedExtremeAutoregressiveTransform) or 
                isinstance(layer, MaskedTailAutoregressiveTransform)
            ):
                param_groups.append({'params': layer.parameters(), 'weight_decay': config['weight_decay']})
            else:
                param_groups.append({'params': layer.parameters()})

        # data
        train_loader = torch.utils.data.DataLoader(x_trn.to(device), generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(x_val.to(device), generator=torch.Generator(device=device), batch_size=config['batch_size'], shuffle=True)

        optimizer = Adam(param_groups, lr=config['lr'])

        # set up quantities to track across training
        if verbose:
            loop = tqdm.tqdm(range(num_epochs))
        else:
            loop = range(num_epochs)

        for epoch in loop:  # loop over the dataset multiple times
            # train
            model.train()  # Set the model to training mode
            train_loss = 0.
            for inputs in train_loader:
                optimizer.zero_grad()
                loss = -model.log_prob(inputs).mean()
                train_loss += loss
                loss.backward()
                optimizer.step()
            train_loss = train_loss / len(train_loader)

            # val
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Do not calculate gradients to save memory
                validation_loss = 0
                for inputs in val_loader:
                    loss = -model.log_prob(inputs).mean()
                    validation_loss += loss
                validation_loss = validation_loss / len(val_loader)

            metadata = {
                'val': validation_loss.detach().cpu().numpy().item(),
                'trn': train_loss.detach().cpu().numpy().item(),
                'epoch': epoch,
            }
            if verbose:
                loop.set_postfix(metadata)

            session.report(metadata)

    
    train = partial(_train, dtype=dtype, model_name=model_name, dim=dim, model_kwargs=model_kwargs)
    config = {
        # "lr": tune.loguniform(1e-4, 5e-1),
        'lr': 1e-3,
        # "weight_decay": tune.loguniform(1e-6, 10),
        # "weight_decay": tune.choice([0., 0.5, 1., 10.]),
        "batch_size": 500,
        "weight_decay": 0,
    }

    ray.init(
        configure_logging=True,
        logging_level=logging.INFO,
    )

    scheduler = ASHAScheduler(
        metric="val",
        mode="min",
        max_t=400,
        grace_period=10,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(train, {"cpu": 2}),
        tune_config=tune.TuneConfig(
            num_samples=100,
            scheduler=scheduler,
        ),
        param_space=config,
    )
    results = tuner.fit()


    best_trial = results.get_best_result(metric="val", mode="min", scope="all")
    best_epoch = best_trial.metrics_dataframe.val.argmin()

    print(f"Best trial config: {best_trial.config}")
    print(f'Best trial final validation loss: {best_trial.metrics_dataframe.val.min():.2f}')

    dfs = {result.log_dir: result.metrics_dataframe for result in results}
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        ax = d.val.plot(ax=ax, legend=False)
        ax = d.trn.plot(ax=ax, legend=False, linestyle='--')

    ax.axvline(best_epoch, linestyle='--', c='black', alpha=0.5)
    plt.show()

    # save training data as np arrays
    # training_data = {
    #     'train_losses': np.array([loss.detach().cpu().numpy() for loss in train_losses]),
    #     'val_losses': np.array([loss.detach().cpu().numpy() for loss in val_losses]),
    #     'best_val_loss': best_val_loss.detach().cpu().numpy(),
    #     'best_val_epoch': best_val_epoch,
    #     'test_loss': test_loss.detach().cpu().numpy(),
    #     'iterations': np.arange(num_epochs),
    # }
    
    # with open(f'{path}/training_data.npy', 'wb') as f:
    #     np.save(f, training_data)

# map a string to a more complicated configuration
target_configs = {
    'top_10': {'top_n_symbols': 10},
    'top_20': {'top_n_symbols': 20},
    'top_50': {'top_n_symbols': 50},
    'top_100': {'top_n_symbols': 100},
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, action='store', default=0)
    parser.add_argument('-model_name', type=str, action='store', default='TTF')
    parser.add_argument('-target_config', type=str, action='store', default='top_100')
    parser.add_argument('-num_epochs', type=int, action='store', default=100)
    parser.add_argument('-lr', type=float, action='store', default=1e-3)

    args = parser.parse_args()

    target_kwargs = target_configs[args.target_config]
    
    run_experiment(
        seed=args.seed, 
        target_name='financial_returns',
        target_kwargs=target_kwargs,
        lr=args.lr, 
        num_epochs=args.num_epochs, 
        batch_size=500,
        model_name=args.model_name,
        model_kwargs={},
        device='cpu',
        precision='float64'
    )