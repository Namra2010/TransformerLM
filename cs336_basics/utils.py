import argparse
import math
from collections.abc import Iterable
import os
import torch
from einops import einsum
import yaml

def default_dff(d_model: int, dim_multiple:int = 64) -> int:
    d_ff = math.ceil((8.0 * d_model) / 3.0)
    return int(((d_ff + dim_multiple - 1) // dim_multiple) * dim_multiple)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    x_stable = x - x_max

    x_softmax = torch.exp(x_stable) / torch.sum(torch.exp(x_stable), dim=dim, keepdim=True)
    return x_softmax

def scaled_dot_product_attention(
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: torch.Tensor | None = None) -> torch.Tensor:
    # Query.shape: (batch_size, ..., seq_len, d_k)
    # Key.shape:   (batch_size, ..., seq_len, d_k)
    # Value.shape: (batch_size, ..., seq_len, d_v)
    # Mask.shape:  (seq_len, seq_len) ---> Optional

    d_k = query.shape[-1]
    Q = query.to(torch.float32)
    K = key.to(torch.float32)
    V = value.to(torch.float32)

    pre_attn = einsum(Q, K, '... n d, ... m d -> ... n m') / math.sqrt(d_k)

    if mask is not None:
        pre_attn = torch.where(mask, pre_attn, float('-inf'))

    softmax_out = softmax(pre_attn, dim=-1)
    attention = einsum(softmax_out, V, '... n m, ... m d -> ... n d')

    return attention

def causal_mask(seq_len: int, device: torch.device | None=None) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    m = logits.max(dim=-1, keepdim=True).values #[B, 1]

    logsumexp = m + torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))

    logit_target = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1))  #[B, 1]

    loss = -(logit_target - logsumexp).mean()

    return loss

def lr_cosine_schedule(t: int, lr_max: float, lr_min: float, num_warmup_iters: int, num_cosine_iters: int):
    if num_warmup_iters < 0 or num_cosine_iters < 0:
        raise ValueError("num_warmup_iters and num_cosine_iters must be non-negative")
    if num_cosine_iters <= num_warmup_iters:
        raise ValueError("num_cosine_iters must be greater than num_warmup_iters")

    if t < num_warmup_iters:
        return (t * lr_max) / num_warmup_iters
    elif t >= num_warmup_iters and t <= num_cosine_iters:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - num_warmup_iters) / (num_cosine_iters - num_warmup_iters))) * (lr_max - lr_min)
    elif t >= num_cosine_iters:
        return lr_min

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    params_list = list(params)
    grads = torch.cat([p.grad.view(-1) for p in params_list if p.grad is not None])
    l2_norm = torch.sqrt(torch.sum(grads ** 2))

    if l2_norm > max_l2_norm:
        clip_coeff = max_l2_norm / (l2_norm + eps)
        for p in params_list:
            if p.grad is None:
                continue
            p.grad.mul_(clip_coeff)

    return l2_norm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config.yaml")
    	)
    parser.add_argument("--prompt")
    return parser.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")