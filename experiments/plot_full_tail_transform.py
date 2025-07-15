import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from tailnflows.models.extreme_transformations import GTransform, ExpTailTransform, FullTailMarginalTransform


def main():
    z = torch.linspace(-4, 4, 2000).unsqueeze(1)
    base = Normal(0.0, 1.0)

    g = GTransform(1, p=1.0, kappa=1.0, epsilon=1.0, delta=1.0)
    lambdas = [0.5, 1.0, 2.0]

    fig, ax = plt.subplots()
    for lam in lambdas:
        t = ExpTailTransform(1, lam=lam)
        x1, lad1 = g.forward(z)
        y, lad2 = t.forward(x1)
        pdf_z = torch.exp(base.log_prob(z.squeeze()))
        pdf_y = pdf_z * torch.exp(lad1.squeeze() + lad2.squeeze())
        idx = torch.argsort(y.squeeze())
        ax.plot(y.squeeze()[idx].detach().numpy(), pdf_y[idx].detach().numpy(), label=f"lambda={lam}")

    ax.set_xlabel("y")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
