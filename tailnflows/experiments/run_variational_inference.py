import datetime
import numpy as np
import tqdm

# pytorch
import torch
from torch.optim import Adam

# our modules
from tailnflows.models.flow_models import get_model
from tailnflows.experiments.utils import get_project_root, new_experiment, write_experiment_details, save_model
from tailnflows.metrics import ess, marginal_likelihood
from tailnflows.targets import targets

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
        opt_steps, 
        batch_size, 
        model_name,
        model_kwargs,
        loss_name,
        record_interval,
        precision='float64',
        safe_grad=False,
    ):
    
    torch.set_default_dtype(precision_types[precision])
    torch.manual_seed(seed)
    
    # setup target model
    target, dim, label = targets[target_name](target_kwargs)

    # record experiment details
    base_path = f'{get_project_root()}/experiment_output/{target_name}'
    experiment_id = new_experiment(base_path)
    subfolder = f'id_{experiment_id}|seed_{seed}|steps_{opt_steps}|{label}'
    path = f'{base_path}/{subfolder}'
    print(f'targetting {target_name} with {model_name} flow, params:[{subfolder}]...')
    
    details = {
        'seed': seed, 
        'grad_clip': grad_clip,
        'safe_grad': safe_grad,
        'lr': lr, 
        'target': target_name,
        'target_kwargs': target_kwargs,
        'opt_steps': opt_steps, 
        'batch_size': batch_size, 
        'model': model_name,
        'model_kwargs': model_kwargs,
        'run_time': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
        'record_interval': record_interval,
        'path': path,
        'dim': dim,
    }
    write_experiment_details(path, details)

    # configure approximation
    sample_and_log_prob, model = get_model(model_name, dim, model_kwargs)
    parameters = list(model.parameters())

    # loss (ELBO)
    def elbo_loss(sample_and_log_prob, target, samples=batch_size):
        x, log_q_x = sample_and_log_prob(samples)
        log_p_x = target(x)
        return (log_q_x - log_p_x).mean()
    
    losses = {
        'elbo': elbo_loss,
    }
    loss = losses[loss_name]

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
    losses = []
    esses = []
    marginal_liks = []
    iterations = []
    df_track = np.zeros([int(opt_steps / record_interval) + 1, dim])
    loop = tqdm.tqdm(range(opt_steps))

    def record(step, loss_val, ess_val, marginal_lik_val):
        loss_val = loss_val.detach().squeeze().numpy()
        ess_val = ess_val.detach().squeeze().numpy()
        losses.append(loss_val)
        esses.append(ess_val)
        marginal_liks.append(marginal_lik_val)
        iterations.append(step)
        loop.set_postfix({
            'elbo': f'{-loss_val:.3f}', 
            'ess': f'{ess_val:.2f}',
            'log p(y)': f'{marginal_lik_val:.2f}',
        })

    ## record initial state
    record(
        0, 
        loss(sample_and_log_prob, target),
        ess(sample_and_log_prob, target),
        marginal_likelihood(sample_and_log_prob, target),
    )

    # train
    # torch.autograd.set_detect_anomaly(True)
    for step in loop:
        optimizer.zero_grad()
        loss_val = loss(sample_and_log_prob, target)
        loss_val.backward()
        optimizer.step()

        if (step + 1) % record_interval == 0:
            _ess = ess(sample_and_log_prob, target)
            _marginal_lik = marginal_likelihood(sample_and_log_prob, target)
            record(step, loss_val, _ess, _marginal_lik)

    # save model
    save_model(path, model)

    # save training data as np arrays
    training_data = {
        'losses': np.array(losses),
        'esses': np.array(esses),
        'iterations': np.array(iterations),
        'df_track': df_track,
    }
    with open(f'{path}/training_data.npy', 'wb') as f:
        np.save(f, training_data)


if __name__ == '__main__':
    # options
    lr = 1e-3
    opt_steps = int(2e3)
    batch_size = 100

    target_sets = [
        (
            'neals_funnel',
            {
                'scale_model_name': 'funnel',
                'd_nuisance': d,
                'df_nuisance': df,
                'heavy_nuisance': True,
            },
        )
        for d in [20] for df in [1.]
    ]
    seeds = [5]

    # target_sets = [
        # ('diamonds', {}),
        # ('eight_schools', {}),
    # ]

    model_kwargs = [
        {'tail_init': -10.},
    ]

    for seed in seeds:
        for target_name, target_kwargs in target_sets:
            for model_name in ['TTF', 'ATAF']:
                for _model_kwargs in  model_kwargs:
                    run_experiment(
                        seed=seed, 
                        target_name=target_name,
                        target_kwargs=target_kwargs,
                        grad_clip=False,
                        lr=lr, 
                        opt_steps=opt_steps, 
                        batch_size=batch_size,
                        model_name=model_name,
                        model_kwargs=_model_kwargs,
                        loss_name='elbo',
                        record_interval=1000
                    )
