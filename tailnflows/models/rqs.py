from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
class RQSTransform():
    def __init__(self, tail_bound, num_bins, min_bin_width=1e-3, min_bin_height=1e-3, min_derivative=1e-3):
        # width, height, derivative
        self.param_shape = (num_bins, num_bins, num_bins - 1)
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tail_bound = tail_bound

    def forward_and_lad(self, z, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        x, lad = unconstrained_rational_quadratic_spline(
            inputs=z,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=False,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tails='linear',
            tail_bound=self.tail_bound,
        )
        return x, lad.sum(axis=1)

    def inverse_and_lad(self, x, unnormalized_widths, unnormalized_heights, unnormalized_derivatives):
        z, lad = unconstrained_rational_quadratic_spline(
            inputs=x,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=True,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tails='linear',
            tail_bound=self.tail_bound,
        )
        return z, lad.sum(axis=1)