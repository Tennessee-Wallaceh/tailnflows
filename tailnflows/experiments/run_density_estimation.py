import datetime
import numpy as np
import tqdm
import argparse

# pytorch
import torch
from torch.optim import Adam

# our modules
from tailnflows.models.density_estimation import get_model
from tailnflows.data import data_sources
from tailnflows.experiments.utils import new_experiment, write_experiment_details, save_model
from tailnflows.utils import get_project_root

precision_types = {
    'float64': torch.float64,
    'float32': torch.float32,
}

def run_experiment(
        seed, 
        target_name,
        target_kwargs,
        grad_clip, 
        lr, 
        num_epochs, 
        batch_size, 
        model_name,
        model_kwargs,
        precision='float64',
        safe_grad=False,
        device='cpu'
    ):
    
    # config
    torch.manual_seed(seed)
    data_seed = seed + 1000 # just start as shift of global seed
    torch.set_default_device(device)
    dtype = precision_types[precision]
    torch.set_default_dtype(dtype)

    # setup target data
    x_trn, x_val, x_tst, dim, _ = data_sources[target_name](dtype, data_seed, standardise=True, **target_kwargs)
    train_loader = torch.utils.data.DataLoader(x_trn.to(device), generator=torch.Generator(device=device), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val.to(device), generator=torch.Generator(device=device), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(x_tst.to(device), generator=torch.Generator(device=device), batch_size=batch_size, shuffle=True)

    # setup model
    model = get_model(dtype, model_name, dim, model_kwargs).to(device)
    parameters = list(model.parameters())

    # record experiment details
    base_path = f'{get_project_root()}/experiment_output/{target_name}'
    experiment_id = new_experiment(base_path)
    subfolder = f'id_{experiment_id}|seed_{seed}|num_epochs_{num_epochs}'
    path = f'{base_path}/{subfolder}'
    print(f'targetting {target_name} with {model_name} flow, params:[{subfolder}]...')
    
    details = {
        'seed': seed, 
        'grad_clip': grad_clip,
        'safe_grad': safe_grad,
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

    # optimizer
    def _check_grad(grad):
        valid_gradients = not (
            torch.isnan(grad).any() or 
            torch.isinf(grad).any()
        )
        if valid_gradients:
            return grad
        else:
            return torch.zeros_like(grad)
        
    for p in parameters:
        if grad_clip:
            p.register_hook(lambda grad: torch.clamp(grad, -10, 10))

        if safe_grad:
            p.register_hook(_check_grad)

    optimizer = Adam(parameters, lr=lr)

    # set up quantities to track across training
    loop = tqdm.tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    best_val_loss = None
    best_val_epoch = None
    # torch.autograd.set_detect_anomaly(True)
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

        val_losses.append(validation_loss)
        train_losses.append(train_loss)

        if best_val_loss is None or validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_val_epoch = epoch
            save_model(path, model) # save the best model

            # record test loss for best model
            test_loss = 0
            for inputs in test_loader:
                loss = -model.log_prob(inputs).mean()
                test_loss += loss
            test_loss = test_loss / len(test_loader)
        
        loop.set_postfix({
            'val': f'{validation_loss:.2f}',
            'trn': f'{train_loss:.2f}',
            '*val': f'{best_val_loss:.2f}',
            '*epoch': f'{int(best_val_epoch)}',
        })

    # save training data as np arrays
    training_data = {
        'train_losses': np.array([loss.detach().cpu().numpy() for loss in train_losses]),
        'val_losses': np.array([loss.detach().cpu().numpy() for loss in val_losses]),
        'best_val_loss': best_val_loss.detach().cpu().numpy(),
        'best_val_epoch': best_val_epoch,
        'test_loss': test_loss.detach().cpu().numpy(),
        'iterations': np.arange(num_epochs),
    }
    with open(f'{path}/training_data.npy', 'wb') as f:
        np.save(f, training_data)

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
        grad_clip=False,
        lr=args.lr, 
        num_epochs=args.num_epochs, 
        batch_size=500,
        model_name=args.model_name,
        model_kwargs={},
        device='cpu',
        precision='float64'
    )