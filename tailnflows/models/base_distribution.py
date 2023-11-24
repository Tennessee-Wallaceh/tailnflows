import torch
from nflows.distributions.base import Distribution
from torch.distributions.utils import _standard_normal
from torch.nn.functional import softplus
from nflows.distributions.normal import StandardNormal
from tailnflows.models.utils import inv_sftplus
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.nn.functional import softplus, softmax
from torch.special import gammaln


def generalized_normal_log_pdf(x, beta):
    """
    Compute the log PDF of the standard generalized normal distribution.

    Parameters:
    - x: Tensor of input values.
    - beta: Shape parameter.

    Returns:
    - Tensor of the same shape as `x`, containing the log PDF values.
    """

    log_normalization = torch.log(beta) - torch.log(torch.tensor(2))
    log_normalization -= gammaln(1 / beta)

    exponent = -torch.abs(x).pow(beta)

    return log_normalization + exponent


class TrainableStudentT(Distribution):
    MIN_DF = 1e-3  # minimum degrees of freedom, needed for numerical stability

    def __init__(self, dim=2, init=1.0):
        super().__init__()
        self._shape = torch.Size([dim])
        self.dim = dim
        if hasattr(init, "shape"):
            _init_unc = inv_sftplus(init).reshape(-1)
        else:
            _init_unc = inv_sftplus(torch.ones([dim]) * init).reshape(-1)

        assert (
            _init_unc.shape == self._shape
        ), "Degrees of freedom incorrectly initialised!"

        self.unc_dfs = torch.nn.parameter.Parameter(_init_unc)
        self.batch_shape = torch.Size([])
        self.event_shape = torch.Size([dim])

    @property
    def dfs(self):
        return self.MIN_DF + softplus(self.unc_dfs)

    def _log_prob(self, inputs, context):
        log_prob = torch.distributions.studentT.StudentT(self.dfs).log_prob(inputs)
        return log_prob.sum(axis=1)

    def _sample(self, num_samples, context):
        """
        Roll my own sample for stability. Important step is to clamp the chi2 sample to a minimum value.
        """
        # return torch.distributions.studentT.StudentT(self.dfs).rsample([num_samples])
        X = _standard_normal(
            [num_samples, self.dim], dtype=self.dfs.dtype, device=self.dfs.device
        )
        Z = torch.distributions.chi2.Chi2(self.dfs).rsample([num_samples])
        Z.detach().clamp_(min=torch.finfo(self.dfs.dtype).tiny * 1e13)
        Y = X * torch.rsqrt(Z / self.dfs)
        return Y

    def rsample(self, size, **kwargs):
        return self.sample(size[0], **kwargs)


class GeneralisedNormal(Distribution):
    def __init__(self, dim=2, beta_init=torch.tensor(2.0)):
        super().__init__()

        # nflow Distribution properties
        self._shape = torch.Size([dim])
        self.batch_shape = torch.Size([])
        self.event_shape = torch.Size([dim])

        self.dim = dim

        # dist params
        if beta_init.shape == torch.Size([]):
            # scalar init, across all dims
            _unc_init = inv_sftplus(beta_init) * torch.ones([dim])
        elif beta_init.shape == self._shape:
            # per dim init provided
            _unc_init = inv_sftplus(beta_init)
        else:
            raise Exception("Cannot interpret beta init!")

        self.unc_beta = torch.nn.parameter.Parameter(_unc_init)

    @property
    def betas(self):
        return softplus(self.unc_beta)

    def _log_prob(self, inputs, context):
        return generalized_normal_log_pdf(inputs, self.betas).sum(axis=1)

    def _sample(self, num_samples, context):
        raise NotImplementedError

    def rsample(self, size, **kwargs):
        raise NotImplementedError


class JointDistribution(Distribution):
    def __init__(self, marginal_distributions, marginals):
        super(JointDistribution, self).__init__()

        assert len(marginal_distributions) == len(marginals)
        self.dim = sum(md._shape[0] for md in marginal_distributions)
        assert (
            torch.concatenate(marginals).sort().values == torch.arange(self.dim)
        ).all()

        # use of ModuleList registers each submodule
        self.marginal_distributions = torch.nn.ModuleList(marginal_distributions)
        self.marginals = marginals
        self._shape = torch.Size([self.dim])
        self._device_p = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self._device_p.device

    @property
    def dtype(self):
        return self._device_p.dtype

    def _log_prob(self, inputs, context):
        _log_prob = torch.zeros_like(inputs[:, 0])
        for marginal_dist, marginal_ix in zip(
            self.marginal_distributions, self.marginals
        ):
            _log_prob += marginal_dist.log_prob(inputs[:, marginal_ix])

        return _log_prob

    def _sample(self, num_samples, context):
        x = torch.zeros([num_samples, self.dim], dtype=self.dtype, device=self.device)
        for marginal_dist, marginal_ix in zip(
            self.marginal_distributions, self.marginals
        ):
            x[:, marginal_ix] = marginal_dist.sample(num_samples)

        return x


class NormalStudentTJoint(JointDistribution):
    def __init__(self, degrees_of_freedom):
        normal_dim = 0
        t_dim = 0
        normal_marginal_ix = []
        t_marginal_ix = []

        for df_ix, df in enumerate(degrees_of_freedom):
            if df == 0:
                normal_dim += 1
                normal_marginal_ix.append(df_ix)
            else:
                t_dim += 1
                t_marginal_ix.append(df_ix)

        if normal_dim == 0:
            marginal_distributions = [
                TrainableStudentT(dim=t_dim, init=degrees_of_freedom)
            ]
            marginals = [torch.arange(t_dim, dtype=torch.int)]
        elif t_dim == 0:
            marginal_distributions = [StandardNormal([normal_dim])]
            marginals = [torch.arange(normal_dim, dtype=torch.int)]

        else:
            marginal_distributions = [
                StandardNormal([normal_dim]),
                TrainableStudentT(dim=t_dim, init=degrees_of_freedom[t_marginal_ix]),
            ]
            marginals = [
                torch.tensor(normal_marginal_ix),
                torch.tensor(t_marginal_ix),
            ]

        super(NormalStudentTJoint, self).__init__(marginal_distributions, marginals)


class NormalMixture(JointDistribution):
    def __init__(self, dim, n_component):
        marginal_distributions = [GMM(n_component) for _ in range(dim)]
        super(NormalMixture, self).__init__(*marginal_distributions)


class GMM(Distribution):
    def __init__(self, n_components):
        super().__init__()
        self.mixture = torch.nn.Parameter(torch.rand(n_components))
        self.means = torch.nn.Parameter(torch.rand(n_components))
        self.unc_scales = torch.nn.Parameter(torch.rand(n_components))
        self._shape = torch.Size([1])
        self.batch_shape = torch.Size([])
        self.event_shape = torch.Size([1])

    @property
    def mixture_weights(self):
        return softmax(self.mixture)

    def _log_prob(self, inputs, context):
        return self._gmm().log_prob(inputs)

    def _sample(self, num_samples, context):
        return self._gmm().sample(inputs)

    def _gmm(self):
        mix = Categorical(probs=self.mixture_weights)
        comp = Normal(self.means, softplus(self.unc_scales))
        gmm = MixtureSameFamily(mix, comp)
        return gmm
