import torch
from nflows.utils import torchutils

MIN_DERIVATIVE = 1e-3


def _check_inputs(inputs, input_knots, output_knots, derivatives):
    assert (
        derivatives > MIN_DERIVATIVE
    ).all(), "Derivatives must be > 0"  # check valid derivatives
    assert all(
        (
            inputs.shape[0] == input_knots.shape[0],
            inputs.shape[0] == output_knots.shape[0],
            inputs.shape[0] == derivatives.shape[0],
        )
    ), "Knot points and derivatives must have same batch size as inputs"
    assert all(
        (
            inputs.shape[1] == input_knots.shape[1],
            inputs.shape[1] == output_knots.shape[1],
            inputs.shape[1] == derivatives.shape[1],
        )
    ), "Knot points and derivatives must have same dimension as inputs"
    assert all(
        (
            input_knots.shape[2] == output_knots.shape[2],
            input_knots.shape[2] == derivatives.shape[2],
        )
    ), "Knot points and derivatives must contain same number of knots"


def forward_rqs(inputs, input_knots, output_knots, derivatives):
    # K + 1 points for K spline segments
    # derivatives batch x dim x (K + 1)
    # input_knots batch x dim x (K + 1)
    # inputs batch x dim
    _check_inputs(inputs, input_knots, output_knots, derivatives)

    in_bin_widths = input_knots[..., 1:] - input_knots[..., :-1]
    out_bin_widths = output_knots[..., 1:] - output_knots[..., :-1]

    # convert to bin data
    # unsqueeze to match input dimensions ie batch x dim x 1
    bin_idx = torchutils.searchsorted(input_knots, inputs).unsqueeze(-1)

    # all are single values for each (input, dim) combo [batch x dim]
    input_left_knot = input_knots.gather(-1, bin_idx).squeeze(-1)
    input_bin_width = in_bin_widths.gather(-1, bin_idx).squeeze(-1)
    input_left_derivative = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_right_derivative = derivatives.gather(-1, bin_idx + 1).squeeze(-1)
    output_left_knot = input_knots.gather(-1, bin_idx).squeeze(-1)
    output_bin_width = out_bin_widths.gather(-1, bin_idx).squeeze(-1)

    # compute intermediates
    input_delta = input_bin_width / output_bin_width  # s_k in the paper

    theta = (inputs - input_left_knot) / input_bin_width  # eta(x) in the paper
    theta_one_minus_theta = theta * (1 - theta)

    # # compute
    numerator = output_bin_width * (
        input_delta * theta.square() + input_left_derivative * theta_one_minus_theta
    )

    denominator = input_delta + (
        (input_left_derivative + input_right_derivative - 2 * input_delta)
        * theta_one_minus_theta
    )

    outputs = output_left_knot + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_right_derivative * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_left_derivative * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


def inverse_rqs(outputs, input_knots, output_knots, derivatives):
    # K + 1 points for K spline segments
    # derivatives batch x dim x (K + 1)
    # input_knots batch x dim x (K + 1)
    # outputs batch x dim
    _check_inputs(outputs, input_knots, output_knots, derivatives)

    in_bin_widths = input_knots[..., 1:] - input_knots[..., :-1]
    out_bin_widths = output_knots[..., 1:] - output_knots[..., :-1]

    # convert to bin data
    # unsqueeze to match input dimensions ie batch x dim x 1
    bin_idx = torchutils.searchsorted(output_knots, outputs).unsqueeze(-1)

    # all are single values for each (input, dim) combo [batch x dim]
    input_left_knot = input_knots.gather(-1, bin_idx).squeeze(-1)
    input_bin_width = in_bin_widths.gather(-1, bin_idx).squeeze(-1)
    input_left_derivative = derivatives.gather(-1, bin_idx).squeeze(-1)
    input_right_derivative = derivatives.gather(-1, bin_idx + 1).squeeze(-1)
    output_left_knot = input_knots.gather(-1, bin_idx).squeeze(-1)
    output_bin_width = out_bin_widths.gather(-1, bin_idx).squeeze(-1)

    # compute intermediates
    input_delta = input_bin_width / output_bin_width  # s_k in the paper

    # compute root
    a = output_bin_width * (input_delta - input_left_derivative)
    a += (outputs - output_left_knot) * (
        input_left_derivative + input_right_derivative - 2 * input_delta
    )

    b = output_bin_width * input_left_derivative
    b -= (outputs - output_left_knot) * (
        input_left_derivative + input_right_derivative - 2 * input_delta
    )

    c = -input_delta * (outputs - output_left_knot)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    theta = (2 * c) / (-b - torch.sqrt(discriminant))
    theta_one_minus_theta = theta * (1 - theta)

    inputs = theta * input_bin_width + input_left_knot

    # compute the lad
    denominator = input_delta + (
        (input_left_derivative + input_right_derivative - 2 * input_delta)
        * theta_one_minus_theta
    )

    derivative_numerator = input_delta.pow(2) * (
        input_right_derivative * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_left_derivative * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return inputs, logabsdet
