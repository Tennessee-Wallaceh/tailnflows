import torch
import matplotlib.pyplot as plt


def plot_contour(
    ax,
    target,
    dim,
    inspect_loc=None,
    target_x_0=0,
    target_x_1=1,
    cmap="viridis",
    min=-5,
    max=5,
    contour_type="fill",
    num_levels=10,
    num_points=100,
):

    x_0 = torch.linspace(min, max, num_points)
    x_1 = torch.linspace(min, max, num_points)
    _x = torch.hstack([_x.reshape(-1, 1) for _x in torch.meshgrid(x_0, x_1)])

    if inspect_loc is None:
        input = torch.zeros([_x.shape[0], dim])
    else:
        input = inspect_loc.repeat(_x.shape[0], 1)
        _x[:, 0] += inspect_loc[target_x_0]
        _x[:, 1] += inspect_loc[target_x_1]

    input[:, target_x_0] = _x[:, 0]
    input[:, target_x_1] = _x[:, 1]

    out = target(input)

    if contour_type == "fill":
        contour_fcn = ax.contourf
    else:
        contour_fcn = ax.contour
        cmap = "magma"

    cont = contour_fcn(
        _x[:, 0].reshape(num_points, num_points),
        _x[:, 1].reshape(num_points, num_points),
        out.reshape(num_points, num_points),
        cmap=cmap,
        levels=num_levels,
    )

    return out, cont
