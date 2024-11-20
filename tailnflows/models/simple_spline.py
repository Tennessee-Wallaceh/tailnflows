"""
My own implementation of the work in https://arxiv.org/pdf/1906.04032
"""

import torch

MIN_DERIVATIVE = 1e-2


def _check_inputs(inputs, input_knots, output_knots, derivatives):
    assert (
        derivatives > MIN_DERIVATIVE
    ).all(), "Derivatives must be > 0"  # check valid derivatives
    assert all(
        (
            inputs.shape[:-1] == input_knots.shape[:-2],
            inputs.shape[:-1] == output_knots.shape[:-2],
            inputs.shape[:-1] == derivatives.shape[:-2],
        )
    ), "Knot points and derivatives must have same batch size as inputs"
    assert all(
        (
            inputs.shape[-1] == input_knots.shape[-2],
            inputs.shape[-1] == output_knots.shape[-2],
            inputs.shape[-1] == derivatives.shape[-2],
        )
    ), "Knot points and derivatives must have same dimension as inputs"
    assert all(
        (
            input_knots.shape[-1] == output_knots.shape[-1],
            input_knots.shape[-1] == derivatives.shape[-1],
        )
    ), "Knot points and derivatives must contain same number of knots"


def univariate_forward_rqs(inputs, input_knots, output_knots, forward_derivatives):
    # K + 1 points for K spline segments
    # derivatives batch x (K + 1)
    # input_knots batch x (K + 1)
    # inputs batch x dim
    # _check_inputs(inputs, input_knots, output_knots, derivatives)

    in_bin_widths = input_knots[:, 1:] - input_knots[:, :-1]
    out_bin_widths = output_knots[:, 1:] - output_knots[:, :-1]

    # how many knots are the inputs greater than?
    # we assume that we are always inside the first bin
    # [batch x (K + 1)]
    bin_idx = (inputs[:, None] >= input_knots).sum(dim=-1, keepdim=True) - 1

    # select appropriate knot data, and squeeze the trailing dimension
    input_left_knot = input_knots.gather(-1, bin_idx).squeeze()
    input_bin_width = in_bin_widths.gather(-1, bin_idx).squeeze()
    input_left_derivative = forward_derivatives.gather(-1, bin_idx).squeeze()
    input_right_derivative = forward_derivatives.gather(-1, bin_idx + 1).squeeze()
    output_left_knot = output_knots.gather(-1, bin_idx).squeeze()
    output_bin_width = out_bin_widths.gather(-1, bin_idx).squeeze()

    # compute intermediates
    input_delta = output_bin_width / input_bin_width  # s_k in the paper

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


def univariate_inverse_rqs(outputs, input_knots, output_knots, forward_derivatives):
    # K + 1 points for K spline segments
    # derivatives batch x 1 x (K + 1)
    # output_knots batch x 1 x (K + 1)
    # input_knots batch x 1 x (K + 1)
    # outputs batch x 1
    # _check_inputs(outputs, input_knots, output_knots, derivatives)

    in_bin_widths = input_knots[:, 1:] - input_knots[:, :-1]
    out_bin_widths = output_knots[:, 1:] - output_knots[:, :-1]

    # how many knots are the inputs greater than?
    # we assume that we are always inside the first bin
    # [batch x (K + 1)]
    bin_idx = (outputs[:, None] >= output_knots).sum(dim=-1, keepdim=True) - 1

    input_left_knot = input_knots.gather(-1, bin_idx).squeeze()
    input_bin_width = in_bin_widths.gather(-1, bin_idx).squeeze()
    input_left_derivative = forward_derivatives.gather(-1, bin_idx).squeeze()
    input_right_derivative = forward_derivatives.gather(-1, bin_idx + 1).squeeze()
    output_left_knot = output_knots.gather(-1, bin_idx).squeeze()
    output_bin_width = out_bin_widths.gather(-1, bin_idx).squeeze()

    # compute intermediates
    input_delta = output_bin_width / input_bin_width  # s_k in the paper

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

    discriminant = b.square() - 4 * a * c

    theta = (2 * c) / (-b - torch.sqrt(discriminant))
    theta_one_minus_theta = theta * (1 - theta)

    inputs = theta * input_bin_width + input_left_knot

    # compute the lad
    denominator = input_delta + (
        (input_left_derivative + input_right_derivative - 2 * input_delta)
        * theta_one_minus_theta
    )

    derivative_numerator = input_delta.square() * (
        input_right_derivative * theta.square()
        + 2 * input_delta * theta_one_minus_theta
        + input_left_derivative * (1 - theta).square()
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return inputs, -logabsdet


def unc_forward_rqs(inputs, input_knots, output_knots, derivatives):
    con_inputs = torch.clamp(
        inputs,
        input_knots[..., 0] + 1e-3,
        input_knots[..., -1] - 1e-3,
    )

    spline_outputs, spline_lad = forward_rqs(
        con_inputs, input_knots, output_knots, derivatives
    )

    inside_box = torch.logical_and(
        inputs > input_knots[..., 0], inputs < input_knots[..., -1]
    )
    outputs = torch.where(
        inside_box,
        spline_outputs,
        inputs,
    )
    lad = torch.where(
        inside_box,
        spline_lad,
        0.0,
    )
    return outputs, lad


inverse_rqs = torch.vmap(univariate_inverse_rqs, in_dims=1, out_dims=1)
forward_rqs = torch.vmap(univariate_forward_rqs, in_dims=1, out_dims=1)
