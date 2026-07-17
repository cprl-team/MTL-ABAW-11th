from __future__ import annotations

import torch


def pcgrad_backward(losses, shared, heads, rng):
    """Populate .grad on `shared` (PCGrad-combined) and `heads` (per-task summed).

    losses: list of T scalar task-loss tensors (raw, unweighted).
    shared: list of shared (backbone) params requiring grad.
    heads:  list of task-head (+aux) params requiring grad.
    rng:    a random.Random for the projection order (seeded -> reproducible).
    """
    T = len(losses)
    params = shared + heads
    n_sh = len(shared)
    task_shared, task_head = [], []
    for i, l in enumerate(losses):
        g = torch.autograd.grad(l, params, retain_graph=(i < T - 1), allow_unused=True)
        gs = [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g[:n_sh], shared)]
        task_shared.append(torch.cat([x.reshape(-1) for x in gs]) if shared
                           else torch.zeros(0))
        task_head.append(g[n_sh:])

    pc = [g.clone() for g in task_shared]
    for i in range(T):
        order = list(range(T))
        rng.shuffle(order)
        for j in order:
            if i == j:
                continue
            gj = task_shared[j]
            dot = torch.dot(pc[i], gj)
            if dot < 0:
                pc[i] = pc[i] - (dot / gj.pow(2).sum().clamp_min(1e-12)) * gj
    combined = torch.stack(pc).sum(0) if shared else None

    idx = 0
    for p in shared:
        n = p.numel()
        p.grad = combined[idx:idx + n].view_as(p)
        idx += n
    for k, p in enumerate(heads):
        gs = [task_head[i][k] for i in range(T) if task_head[i][k] is not None]
        p.grad = (sum(gs) if gs else None)
