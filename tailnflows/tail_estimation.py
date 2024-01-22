import torch
import matplotlib.pyplot


def vi_marginal_tail(
    log_posterior, max_x=1000.0, corr_thresh=0.9999, plot_diagnostic=False
):
    ## perform a laplace approximation
    mu = torch.tensor(0.0, requires_grad=True)

    for i in range(10):
        approx_post = dist.Normal(mu, 1.0)

        log_unnormalized_posterior = log_posterior(mu)

        grad_log_posterior = torch.autograd.grad(
            log_unnormalized_posterior, mu, create_graph=True
        )
        hessian_log_posterior = torch.autograd.grad(
            grad_log_posterior[0], mu, create_graph=True
        )

        update_step = grad_log_posterior[0] / (-hessian_log_posterior[0])
        mu = mu + update_step

    approx_std = 1.0 / torch.sqrt(-hessian_log_posterior[0])

    # consider a grid
    x_ins = torch.linspace(1e-3, max_x, 1000)
    adj_x = x_ins * approx_std + mu
    log_x_ins = torch.log(adj_x)
    log_p = log_posterior(adj_x)

    x_us = torch.linspace(1e-5, max_x * 0.99, 500)
    for x_u in x_us:
        corr = torch.corrcoef(
            torch.vstack(
                [
                    log_x_ins[log_x_ins > torch.log(x_u)],
                    log_p[log_x_ins > torch.log(x_u)],
                ]
            )
        )[0, 1]
        corrs.append(-corr.detach())
        res = linregress(
            log_x_ins[log_x_ins > torch.log(x_u)].detach(),
            log_p[log_x_ins > torch.log(x_u)].detach(),
        )
        slopes.append(-1 / (res.slope + 1))

    # select most correlated point
    select = np.argmax(corrs)
    if select != len(x_us):
        threshold = torch.log(x_us[select])
    else:
        threshold = None

    if plot_diagnostic:
        fig, axarr = plt.subplots(1, 2)

        axarr[0]

        if select != len(x_us):
            ax.axvline(torch.log(x_us[select]))
        else:
            ax.set_title("No fit")

        ax.plot(
            np.log(x_us),
            slopes,
            c="orange",
            label=f"gradient tail estimate: {slopes[select]: .2f}",
        )
        ax.axhline(1 / df, c="green", linestyle="--", alpha=0.5, label="true tail")

        ax.set_ylabel("gradient tail estimate")
        ax.set_xlabel("threshold (log x)")

        ax2 = ax.twinx()
        ax2.plot(np.log(x_us), corrs, label="correlation")
        ax2.set_ylabel("correlation")
        ax2.legend()
        plt.title(f"Thresholded gradient plot @ correlation {0.999}")
        plt.tight_layout()
        ax.legend(loc="center")

    return slopes[select], threshold
