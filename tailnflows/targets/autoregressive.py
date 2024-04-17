import torch
from dataclasses import dataclass
from jaxtyping import Float
from torch import Tensor
from typing import Callable


@dataclass
class ARParams:
    """
    Encapsulates all parameters needed for a simple autoregressive model.
    The first axis contains indexes individual parameter sets.
    e.g.
    betas[0, 0, 0, :] is AR parameters connecting 1st obs dim to 1st obs dim history
    betas[0, 0, 1, :] is AR parameters connecting 1st obs dim to 2nd obs dim history
    """

    ar_length: int
    obs_dim: int
    betas: Float[Tensor, "param_dim obs_dim obs_dim ar_length"]
    obs_df: Float[Tensor, "param_dim"]  # shared across all observations
    obs_scale: Float[Tensor, "param_dim"]  # shared across all observations
    prior_df: Float[Tensor, "1"]
    prior_scale: Float[Tensor, "1"]


@dataclass
class PredictiveARParams(ARParams):
    # Includes additional 'predictive' samples treated as parameters
    pred_length: int
    predictive_samples: Float[Tensor, "param_dim obs_dim pred_length"]


def from_vector(
    parameters: Float[Tensor, "param_dim (obs_dim*obs_dim*ar_length)+pred_length"],
    pred_length: int,
    base_ar_parameters: ARParams,
) -> PredictiveARParams:
    """
    Converts vector for beta and predictive samples into AR params.
    """
    obs_dim = base_ar_parameters.obs_dim
    ar_length = base_ar_parameters.ar_length
    raw_beta_dim = obs_dim * obs_dim * ar_length
    raw_param_dim = raw_beta_dim + pred_length * obs_dim
    assert (
        parameters.shape[1] == raw_param_dim
    ), f"Unconstrained params don't have correct dimension. {parameters.shape} vs [param_dim ar_length+pred_length]"

    betas = parameters[:, :raw_beta_dim].reshape(-1, obs_dim, obs_dim, ar_length)
    predictive_samples = parameters[:, raw_beta_dim:].reshape(-1, obs_dim, pred_length)

    return PredictiveARParams(
        ar_length=ar_length,
        obs_dim=obs_dim,
        pred_length=pred_length,
        predictive_samples=predictive_samples,
        betas=betas,
        obs_df=base_ar_parameters.obs_df,
        obs_scale=base_ar_parameters.obs_scale,
        prior_df=base_ar_parameters.prior_df,
        prior_scale=base_ar_parameters.prior_scale,
    )


def from_vector_df(
    parameters: Float[Tensor, "param_dim (obs_dim*obs_dim*ar_length)+pred_length+1"],
    pred_length: int,
    base_ar_parameters: ARParams,
) -> PredictiveARParams:
    """
    Converts vector for beta and predictive samples into AR params.
    This also includes the degrees of freedom.
    """
    obs_dim = base_ar_parameters.obs_dim
    ar_length = base_ar_parameters.ar_length
    raw_beta_dim = obs_dim * obs_dim * ar_length
    raw_param_dim = raw_beta_dim + pred_length * obs_dim + 1
    assert (
        parameters.shape[1] == raw_param_dim
    ), f"Unconstrained params don't have correct dimension. {parameters.shape} vs [param_dim ar_length+pred_length]"

    betas = parameters[:, :raw_beta_dim].reshape(-1, obs_dim, obs_dim, ar_length)
    predictive_samples = parameters[:, raw_beta_dim:-1].reshape(
        -1, obs_dim, pred_length
    )
    df = 0.5 + torch.exp(parameters[:, [-1]])

    return PredictiveARParams(
        ar_length=ar_length,
        obs_dim=obs_dim,
        pred_length=pred_length,
        predictive_samples=predictive_samples,
        betas=betas,
        obs_df=df,
        obs_scale=base_ar_parameters.obs_scale,
        prior_df=base_ar_parameters.prior_df,
        prior_scale=base_ar_parameters.prior_scale,
    )


def get_log_likelihood(
    input: Float[Tensor, "num_obs obs_dim ar_length"],
    target: Float[Tensor, "num_obs obs_dim"],
) -> Callable[[ARParams], Float[Tensor, "param_dim"]]:
    """
    Returns a log likelihood function which maps a number of parameter
    settings to log likelihood values, given input x and target x.
    """

    def _log_likelihhod(params: ARParams) -> Float[Tensor, "param_dim"]:
        # [param_dim obs_dim num_obs]
        next_mean = torch.einsum("oir,pjir->pjo", input, params.betas)
        obs_dist = torch.distributions.StudentT(
            df=params.obs_df.reshape(-1, 1, 1),
            loc=next_mean,
            scale=params.obs_scale.reshape(-1, 1, 1),
        )
        ll_values = obs_dist.log_prob(target.T)  # [param_dim obs_dim num_obs]
        # sum across the observation sequence and dimension
        return ll_values.sum(axis=[1, 2])

    return _log_likelihhod


def beta_log_prior(params: ARParams):
    prior_dist = torch.distributions.StudentT(
        df=params.prior_df, loc=0.0, scale=params.prior_scale
    )
    ll_values = prior_dist.log_prob(params.betas)  # [param_dim]
    # sum across the observation sequence and dimension
    return ll_values.sum(axis=[1, 2, 3])


def obs_df_log_prior(params: ARParams):
    prior_dist = torch.distributions.Gamma(2.0, 0.5)
    ll_values = prior_dist.log_prob(params.obs_df)  # [param_dim]
    return ll_values.sum(axis=1)


def get_predictive_ll(pred_input: Float[Tensor, "1 obs_dim ar_length"]):
    """
    Builds the predictive ll, which is a function of the input needed for the
    first predictive samples and the parameters.
    """

    def _pred_ll(params: ARParams) -> Float[Tensor, "param_dim"]:
        param_dim = params.predictive_samples.shape[0]
        ar_length = params.ar_length

        # build the sample observations, prepending the predictive input
        sample_observations = torch.cat(
            [pred_input.repeat(param_dim, 1, 1), params.predictive_samples], dim=-1
        )

        pred_ll = torch.zeros([param_dim])

        # sum over autoregressive steps
        for lag in range(params.pred_length):
            lag_input = sample_observations[:, :, lag : (ar_length + lag)]
            lag_target = sample_observations[:, :, (ar_length + lag)]
            # should be 1 mean for each param
            next_mean = torch.einsum("pir,pjir->pj", lag_input, params.betas)
            target_log_prob = torch.distributions.StudentT(
                df=params.obs_df,
                loc=next_mean,
                scale=params.obs_scale,
            ).log_prob(lag_target)
            pred_ll += target_log_prob.sum(axis=1)  # sum over dimensions
        return pred_ll

    return _pred_ll


def sample_path(params: ARParams, path_length: int = 1000):
    # Only meaningful to sample from a single parameter set
    assert params.betas.shape[0] == 1
    betas = params.betas[0]

    # start from arbitrary point
    x_init = torch.distributions.Normal(0.0, params.obs_scale).sample(
        [params.ar_length]
    )

    xs = torch.zeros([path_length])
    xs[: params.ar_length] = x_init
    for step in range(params.ar_length, path_length):
        next_mean = betas.dot(xs[step - params.ar_length : step])
        next_obs = (
            torch.distributions.StudentT(
                df=params.obs_df,
                loc=next_mean,
                scale=params.obs_scale,
            )
            .sample([1])
            .squeeze()
        )
        xs[step] = next_obs

    return xs.reshape(-1, 1)


def lag_series(series: Float[Tensor, "obs_dim num_obs"], ar_length: int):
    """
    Prepares a series for inference by converting it to tensor on basis of provided ar_length.
    """
    obs_dim = series.shape[0]
    num_obs = series.shape[1] - ar_length

    input = torch.zeros([num_obs, obs_dim, ar_length])
    target = torch.zeros([num_obs, obs_dim])  # next step
    pred_input = torch.zeros([1, obs_dim, ar_length])

    for dim_ix in range(obs_dim):
        for lag in range(ar_length):
            input[:, dim_ix, lag] = series[dim_ix, lag : -(ar_length - lag)]
        target[:, dim_ix] = series[dim_ix, ar_length:]
        pred_input[0, dim_ix, :] = series[dim_ix, -ar_length:]

    return input, target, pred_input
