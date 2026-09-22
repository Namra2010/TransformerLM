from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float, float] = (0.9, 0.999),
            weight_decay: float = 0.01,
            eps: float = 1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}, must each be in [0, 1)")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay, 'eps': eps}
        super().__init__(params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                t = self.state[p].get('t', 1)
                m = self.state[p].get('m', torch.zeros_like(p))
                v = self.state[p].get('v', torch.zeros_like(p))

                m = betas[0] * m + (1 - betas[0]) * grad
                v = betas[1] * v + (1 - betas[1]) * (grad ** 2)

                m_hat = m / (1 - betas[0] ** t)
                v_hat = v / (1 - betas[1] ** t)

                p.data = p.data - ((lr * m_hat) / (torch.sqrt(v_hat) + eps))

                p.data = p.data - (lr * weight_decay * p.data)
                self.state[p]['m'] = m
                self.state[p]['v'] = v
                self.state[p]['t'] = t + 1

        return loss



if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1)

    for t in range(100):
        opt.zero_grad() # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.

        
