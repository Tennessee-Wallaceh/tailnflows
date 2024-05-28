from ucimlrepo import fetch_ucirepo
from torch.nn.functional import logsigmoid
import matplotlib.pyplot as plt
from nflows.transforms.standard import IdentityTransform
from nflows.transforms.orthogonal import HouseholderSequence
from nflows.transforms.base import CompositeTransform
import torch
from tailnflows.utils import load_torch_data, add_raw_data, parallel_runner
from tailnflows.train import variational_fit
from tailnflows.models import flows

DEFAULT_DTYPE = torch.float32

"""
Model specifications
"""


def base_transformation(dim):
    return flows.base_nsf_transform(
        dim,
        depth=2,
        random_permute=True,
        num_bins=5,
        tail_bound=5.0,
        affine_autoreg_layer=True,
    )


def normal(dim):
    return flows.build_base_model(
        dim, use="variational_inference", base_transformation_init=base_transformation
    )


def ttf_m(dim):
    return flows.build_ttf_m(
        dim,
        use="variational_inference",
        base_transformation_init=base_transformation,
        model_kwargs=dict(
            fix_tails=False,
            pos_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
            neg_tail_init=torch.distributions.Uniform(low=0.05, high=1.0).sample([dim]),
        ),
    )


def gtaf(dim):
    return flows.build_gtaf(
        dim,
        use="variational_inference",
        base_transformation_init=base_transformation,
        model_kwargs=dict(
            fix_tails=True,
            tail_init=torch.distributions.Uniform(low=1.0, high=20.0).sample([dim]),
        ),
    )


model_definitions = {
    "normal": normal,
    "ttf_m": ttf_m,
    "ttf_m_2stage": ttf_m,
    "gtaf": gtaf,
}

# ttf tends to converge faster, so lower num epoch
# gtaf sometimes needs a little longer to converge
model_opt_params = {
    "normal": {
        "lr": 5e-3,
        "num_epochs": 20_000,
        "batch_size": 500,
    },
    "ttf_m": {
        "lr": 1e-3,
        "num_epochs": 20_000,
        "batch_size": 500,
    },
    "ttf_m_2stage": [
        {
            "lr": 1e-2,
            "num_epochs": 10_000,
            "batch_size": 500,
        },
        {
            "lr": 1e-3,
            "num_epochs": 10_000,
            "batch_size": 500,
        },
    ],
    "gtaf": {
        "lr": 1e-3,
        "num_epochs": 20_000,
        "batch_size": 500,
    },
}

"""
Experiment code
"""


def load_lung_cancer_data():
    lung_cancer = fetch_ucirepo(id=451)

    X = lung_cancer.data.features
    y = lung_cancer.data.targets

    X_tensor = torch.tensor(X.to_numpy())
    y_tensor = torch.tensor(y.to_numpy()) - 1  # 0 class 1, 1 class 2

    return X_tensor, y_tensor


def log_reg_likelihood(y, x, params):
    w, b = params.split([x.shape[1], 1], dim=1)
    logits = x.matmul(w.T) + b.T

    # Compute the likelihood of the binary labels y under the model parameters
    loglikelihoods = torch.where(y == 1, logsigmoid(logits), logsigmoid(-logits))

    return loglikelihoods.squeeze(0)


def log_p_y_1(x, params):
    w, b = params.split([x.shape[1], 1], dim=1)
    logits = x.matmul(w.T) + b.T
    return logsigmoid(logits)


def log_cauchy_prior(params):
    return torch.distributions.Cauchy(0, 1).log_prob(params).sum(axis=1)


def importance_sampled_predictive(model, target, X_tst, y_tst, nsamp=10000):
    w_approx, log_q_w = model.sample_and_log_prob(nsamp)
    log_p_w = target(w_approx)
    weights = (log_p_w - log_q_w).exp()

    norm_weights = weights / weights.sum()
    tst_ll = log_reg_likelihood(y_tst, X_tst, w_approx).sum(axis=0)
    is_tst_ll = norm_weights.dot(tst_ll)
    return is_tst_ll


def run_experiment(out_path, seed, label, tst_idx, experiment_ix=1):
    """
    This script is a bit messy, because the Householder rotation cannot 
    be initialised on GPU. So there is a bit of CPU-GPU converting to handle this.
    """
    # load data
    X_tensor, y_tensor = load_lung_cancer_data()
    d = X_tensor.shape[1]
    n = X_tensor.shape[0]

    param_dim = d + 1  # for bias

    

    # general setup
    torch.set_default_dtype(DEFAULT_DTYPE)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = "cuda"
    else:
        device = "cpu"

    # have to create model one CPU here, as not all inits work on GPU
    # convert to GPU later
    model_func = model_definitions[label]
    model = model_func(param_dim)

    trn_mask = torch.ones(n, dtype=torch.bool)
    trn_mask[tst_idx] = False
    X_trn = X_tensor.to(DEFAULT_DTYPE).to(device)[trn_mask, :]
    y_trn = y_tensor.to(DEFAULT_DTYPE).to(device)[trn_mask, :]
    X_tst = X_tensor.to(DEFAULT_DTYPE).to(device)[~trn_mask, :]
    y_tst = y_tensor.to(DEFAULT_DTYPE).to(device)[~trn_mask, :]

    # unnormalised log posterior
    def target(params):
        return log_reg_likelihood(y_trn, X_trn, params).sum(axis=0) + log_cauchy_prior(
            params
        )

    # setup model, add rotation to every model
    model = model.to(DEFAULT_DTYPE).to(device)
    opt_param = model_opt_params[label]

    model._transform = CompositeTransform(
        [
            HouseholderSequence(param_dim, param_dim),
            model._transform,
        ]
    )

    # handle two stage
    # replace the base transform with an identity transform
    if label.endswith("2stage"):
        # transform 0 is rotation
        # transform 1 is the rest of the transform
        base_transformation = model._transform._transforms[1]._transforms[1]
        model._transform._transforms[1]._transforms[1] = IdentityTransform()
        opt_param, second_opt_param = opt_param

    # train
    losses, final_metrics = variational_fit.train(
        model,
        target,
        lr=opt_param["lr"],
        num_epochs=opt_param["num_epochs"],
        batch_size=opt_param["batch_size"],
        label=f"{experiment_ix}|{label}",
    )

    # second stage training
    # add back the base transformation
    if label.endswith("2stage"):
        opt_param = second_opt_param
        model._transform._transforms[1]._transforms[1] = base_transformation
        second_losses, final_metrics = variational_fit.train(
            model,
            target,
            lr=opt_param["lr"],
            num_epochs=opt_param["num_epochs"],
            batch_size=opt_param["batch_size"],
            label=f"{experiment_ix}|{label}-2",
        )
        losses = torch.hstack([losses, second_losses])

    # evaluate
    test_is_predictive = importance_sampled_predictive(
        model, target, X_tst, y_tst
    ).detach()

    add_raw_data(
        out_path,
        label,
        {
            "seed": seed,
            "tst_idx": tst_idx.cpu().numpy(),
            "test_is_predictive": test_is_predictive.cpu().numpy(),
            "tst_elbos": final_metrics.elbo,
            "tst_ess": final_metrics.ess,
            "tst_psis_k": final_metrics.psis_k,
        },
        force_write=True,
    )

    add_raw_data(
        out_path + "_losses",
        label,
        {
            "seed": seed,
            "tst_idx": tst_idx.cpu().numpy(),
            "losses": losses.detach().cpu(),
        },        
        force_write=True,
    )

    add_raw_data(
        out_path + "_final_states",
        label,
        model.state_dict(),
        force_write=True,
    )


def get_splits(seed, num_test):
    torch.manual_seed(seed)
    X_tensor, _ = load_lung_cancer_data()
    n = X_tensor.shape[0]
    test_indexes = torch.randperm(n).split(num_test)
    return test_indexes


def configured_experiments():
    seed = 100
    split_seed = 5
    out_path = "2024-05-vi-real"
    test_indexes = get_splits(split_seed, 10)
    model_labels = model_definitions.keys()

    experiments = [
        dict(
            out_path=out_path,
            seed=seed,
            label=model_label,
            tst_idx=tst_idx,
        )
        for tst_idx in test_indexes
        for model_label in model_labels
    ]

    parallel_runner(run_experiment, experiments, max_runs=2)


if __name__ == "__main__":
    configured_experiments()
