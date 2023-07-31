import torch
from nflows.distributions.base import Distribution
from torch.distributions.utils import _standard_normal
from torch.nn.functional import softplus
from nflows.distributions.normal import StandardNormal
from tailnflows.models.utils import inv_sftplus

class TrainableStudentT(Distribution):
    MIN_DF = 1e-3 # minimum degrees of freedom, needed for numerical stability
    def __init__(self, dim=2, init=1.):
        super().__init__()
        self._shape = torch.Size([dim])
        self.dim = dim
        if hasattr(init, 'shape'):
            _init_unc = inv_sftplus(init)
        else:
            _init_unc = inv_sftplus(torch.ones([dim]) * init)
        self.unc_dfs = torch.nn.parameter.Parameter(_init_unc)
        self.register_parameter(f'unc_df', self.unc_dfs)
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
        X = _standard_normal([num_samples, self.dim], dtype=self.dfs.dtype, device=self.dfs.device)
        Z = torch.distributions.chi2.Chi2(self.dfs).rsample([num_samples])
        Z.detach().clamp_(min=torch.finfo(self.dfs.dtype).tiny * 1e13)
        Y = X * torch.rsqrt(Z / self.dfs)
        return Y
    
    def rsample(self, size, **kwargs):
        return self.sample(size[0], **kwargs)
    

class JointDistribution(Distribution):
    def __init__(self, *marginal_distributions):
        super(JointDistribution, self).__init__()
        
        assert all(md._shape == torch.Size([1]) for md in marginal_distributions)

        self.marginal_distributions = marginal_distributions
        self.dim = len(marginal_distributions)
        self._shape = torch.Size([self.dim])

    def _log_prob(self, inputs, context):
        _log_prob = torch.zeros(inputs.shape[0])
        for _dim_ix in range(self.dim):
            _log_prob += self.marginal_distributions[_dim_ix]._log_prob(inputs[:, [_dim_ix]], context)
        return _log_prob

    def _sample(self, num_samples, context):
        return torch.hstack([
            self.marginal_distributions[_dim_ix].sample(num_samples, context)
            for _dim_ix in range(self.dim)
        ])


class NormalStudentTJoint(JointDistribution):
    def __init__(self, degrees_of_freedom):
        marginal_distributions = [
            StandardNormal([1]) if df == 0 else TrainableStudentT(dim=1, init=torch.tensor(df))
            for df in degrees_of_freedom
        ]
        super(NormalStudentTJoint, self).__init__(*marginal_distributions)