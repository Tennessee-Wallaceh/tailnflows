import argparse
import os
import json
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from nflows import transforms, distributions, flows
from nflows.nn.nets import ResidualNet

# -----------------------------------------------------------------------------
# Data utilities taken from data/power.py in this repository
# -----------------------------------------------------------------------------

def load_power_raw():
    file_path = '../data/household_power_consumption.txt'

    # Read the data, skipping the header, and using semicolon as delimiter
    data = np.genfromtxt(
        file_path,
        delimiter=';',
        skip_header=1,
        dtype=float,
        filling_values=np.nan  # Handles missing values ("?")
    )
    return data


def prepare_power_splits():
    rng = np.random.RandomState(42)
    data = load_power_raw()
    rng.shuffle(data)
    
    # Remove 3 columns to yield 6 dim as in paper
    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    data = np.delete(data, 0, axis=1)

    # Add noise exactly as in the original implementation
    N = data.shape[0]
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_val = data[-N_validate:]
    data_train = data[:-N_validate]

    # Normalise using statistics of train+val
    stats_data = np.vstack((data_train, data_val))
    mu = stats_data.mean(axis=0)
    s = stats_data.std(axis=0)
    data_train = (data_train - mu) / s
    data_val = (data_val - mu) / s
    data_test = (data_test - mu) / s
    return data_train.astype(np.float32), data_val.astype(np.float32), data_test.astype(np.float32)


class ArrayDataset(Dataset):
    def __init__(self, array):
        self.data = torch.tensor(array, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

# -----------------------------------------------------------------------------
# Model creation utilities based on experiments/uci.py
# -----------------------------------------------------------------------------

def create_linear_transform(features, linear_type):
    if linear_type == 'permutation':
        return transforms.RandomPermutation(features=features)
    elif linear_type == 'lu':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.LULinear(features, identity_init=True)
        ])
    elif linear_type == 'svd':
        return transforms.CompositeTransform([
            transforms.RandomPermutation(features=features),
            transforms.SVDLinear(features, num_householder=10, identity_init=True)
        ])
    else:
        raise ValueError('Unknown linear transform type')


def create_base_transform(i, features, args):
    """Return one of the supported base transforms."""
    if args.base_transform_type == 'affine-autoregressive':
        return transforms.MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm,
        )
    elif args.base_transform_type == 'rq-autoregressive':
        return transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=features,
            hidden_features=args.hidden_features,
            context_features=None,
            num_bins=args.num_bins,
            tails='linear',
            tail_bound=args.tail_bound,
            num_blocks=args.num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=args.dropout_probability,
            use_batch_norm=args.use_batch_norm,
        )
    else:
        raise ValueError('Unsupported base transform type')


def create_transform(features, args):
    transform_list = []
    for i in range(args.num_flow_steps):
        transform_list.append(
            transforms.CompositeTransform([
                create_linear_transform(features, args.linear_transform_type),
                create_base_transform(i, features, args)
            ])
        )
    transform_list.append(create_linear_transform(features, args.linear_transform_type))
    return transforms.CompositeTransform(transform_list)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def main(args):

    train_array, val_array, test_array = prepare_power_splits()
    train_loader = DataLoader(ArrayDataset(train_array), batch_size=args.train_batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(ArrayDataset(val_array), batch_size=args.val_batch_size,
                            shuffle=True, drop_last=True)
    test_loader = DataLoader(ArrayDataset(test_array), batch_size=args.val_batch_size,
                             shuffle=False, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = train_array.shape[1]
    distribution = distributions.StandardNormal((features,))
    transform = create_transform(features, args)
    flow = flows.Flow(transform, distribution).to(device)

    n_params = sum(p.numel() for p in flow.parameters() if p.requires_grad)
    print(f'Model has {n_params} trainable parameters')

    optimizer = optim.Adam(flow.parameters(), lr=args.learning_rate)
    scheduler = None
    if args.anneal_learning_rate:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_training_steps, 0)

    log_dir = os.path.join('logs_power_nflows')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    train_iter = iter(train_loader)
    best_val = -1e10
    for step in tqdm(range(args.num_training_steps)):
        flow.train()
        if scheduler is not None:
            scheduler.step(step)
        optimizer.zero_grad()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch.to(device)
        loss = -flow.log_prob(batch).mean()
        loss.backward()
        if args.grad_norm_clip_value is not None:
            clip_grad_norm_(flow.parameters(), args.grad_norm_clip_value)
        optimizer.step()

        if (step + 1) % args.monitor_interval == 0:
            flow.eval()
            with torch.no_grad():
                val_log_prob = torch.cat([flow.log_prob(b.to(device)) for b in val_loader])
                mean_val = val_log_prob.mean().item()
            print(f'Step {step+1}: validation log prob {mean_val:.3f}')
            if mean_val > best_val:
                best_val = mean_val
                torch.save(flow.state_dict(), os.path.join(log_dir, 'best_model.pt'))

    # Load best model and evaluate on test set
    flow.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pt')))
    flow.eval()
    with torch.no_grad():
        test_log_prob = torch.cat([flow.log_prob(b.to(device)) for b in test_loader])
    mean_ll = test_log_prob.mean().item()
    std_ll = test_log_prob.std().item()
    np.save(os.path.join(log_dir, 'test_log_probs.npy'), test_log_prob.cpu().numpy())
    with open(os.path.join(log_dir, 'test_results.txt'), 'w') as f:
        f.write(f'Mean log likelihood: {mean_ll:.4f} \u00b1 {2*std_ll/np.sqrt(len(test_array)):.4f}\n')
    print(f'Final test log likelihood: {mean_ll:.4f} \u00b1 {2*std_ll/np.sqrt(len(test_array)):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Power dataset experiment using nflows')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--num_training_steps', type=int, default=200000)
    parser.add_argument('--anneal_learning_rate', type=int, default=1)
    parser.add_argument('--grad_norm_clip_value', type=float, default=5.)
    parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive',
                        choices=['affine-autoregressive', 'rq-autoregressive'])
    parser.add_argument('--linear_transform_type', type=str, default='lu',
                        choices=['permutation', 'lu', 'svd'])
    parser.add_argument('--num_flow_steps', type=int, default=10)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--tail_bound', type=float, default=3)
    parser.add_argument('--num_bins', type=int, default=8)
    parser.add_argument('--num_transform_blocks', type=int, default=2)
    parser.add_argument('--use_batch_norm', type=int, default=0)
    parser.add_argument('--dropout_probability', type=float, default=0.25)
    parser.add_argument('--apply_unconditional_transform', type=int, default=1)
    parser.add_argument('--monitor_interval', type=int, default=250)
    args = parser.parse_args()
    main(args)