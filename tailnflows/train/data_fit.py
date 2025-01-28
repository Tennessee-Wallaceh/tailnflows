from math import ceil
from collections.abc import Iterable
import torch
from torch import optim
import tqdm
import schedulefree


def batch_loader(n: int, batch_size: int) -> Iterable[torch.Tensor]:
    batches = torch.randperm(n).split(batch_size)
    while True:
        for batch_ix in batches:
            yield batch_ix
        batches = torch.randperm(n).split(batch_size)

def train(
    model,
    x_trn,
    x_val,
    x_tst,
    lr=1e-3,
    num_steps=500,
    batch_size=100,
    label="",
    hook=None,
    early_stop_patience=None,
    grad_clip=None,
    optimizer=None,
    lr_scheduler=None,
    preprocess_transformation=None,
    eval_period=1,
):
    parameters = list(model.parameters())
    if optimizer is None:
        optimizer = optim.Adam(parameters, lr=lr)

    # optimizer = schedulefree.AdamWScheduleFree(parameters, lr=lr)

    if lr_scheduler == "cosine_anneal":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps, 0)
    
    if lr_scheduler == "cosine_anneal_wr":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=50, 
            T_mult=1, 
            eta_min=5e-7,
        )

    if preprocess_transformation is not None:
        x_trn, _ = preprocess_transformation(x_trn)
        x_val, _ = preprocess_transformation(x_val)
        x_pre, pre_process_lad = preprocess_transformation(x_tst)

    # training loop data
    loop = tqdm.tqdm(range(num_steps))
    num_evals = ceil(num_steps / eval_period) + 1 # include final eval
    losses = torch.empty(num_steps)
    vlosses = torch.empty(num_evals)
    steps = torch.empty(num_evals, dtype=torch.int)
    hook_data = {}
    tst_loss = torch.tensor(torch.inf)
    best_val_loss = torch.tensor(torch.inf)
    tst_ix = -1

    # data loading
    n = x_trn.shape[0]
    if batch_size is None:
        batch_size = n
    batches = batch_loader(n, batch_size)

    for step in loop:
        # mini batch
        model.train()
        getattr(optimizer, 'train', lambda: None)()

        batch_ix = next(batches)
        batch = x_trn[batch_ix, :]
        optimizer.zero_grad()
        trn_loss = -model.log_prob(batch).mean()
        trn_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)

        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        losses[step] = trn_loss.detach()

        if step % eval_period == 0 or (step + 1) == num_steps:
            model.eval()
            getattr(optimizer, 'eval', lambda: None)()

            with torch.no_grad():
                eval_number = ceil(step / eval_period)

                if hook is not None:
                    hook(model, hook_data)

                val_loss = -model.log_prob(x_val).mean()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if preprocess_transformation is not None:
                        tst_loss = -(model.log_prob(x_pre) + pre_process_lad).mean()
                    else:
                        tst_loss = -model.log_prob(x_tst).mean()
                    tst_eval = eval_number

                steps[eval_number] = step
                vlosses[eval_number] = val_loss.detach()

                if (
                    early_stop_patience is not None
                    and step - steps[tst_eval] > early_stop_patience
                ):
                    break
        
        loop.set_postfix(
            {
                "loss": f"{losses[step]:.2f} ({vlosses[eval_number]:.2f}) {label}: *{tst_loss.detach():.3f} @ {steps[tst_eval]}"
            }
        )
    
    tst_ix = steps[tst_eval]
    val_loss = vlosses[tst_eval]

    return (
        tst_loss.cpu(), 
        val_loss.cpu(), 
        tst_ix.cpu(), 
        losses[:step + 1].cpu(), 
        vlosses[:eval_number + 1].cpu(),  
        steps[:eval_number + 1].cpu(), 
        hook_data
    )