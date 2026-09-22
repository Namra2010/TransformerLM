import json
import math
import os
import time

import numpy as np
import torch

from cs336_basics.optimizer import AdamW
from cs336_basics.transformer import TransformerLM
from cs336_basics.train import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.utils import cross_entropy, gradient_clipping, lr_cosine_schedule, parse_args, load_config, pick_device


def load_tokens(path: str, dtype: str) -> np.ndarray:
    """Memory-map a token file instead of reading it into RAM."""
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def validate_data(data, name, vocab_size, context_length):
    if len(data) <= context_length:
        raise ValueError(
            f"{name} has {len(data)} tokens, but more than "
            f"{context_length} tokens are required."
        )

    sample = np.asarray(data[: min(len(data), 1_000_000)])
    if sample.max() >= vocab_size:
        raise ValueError(
            f"{name} contains token id {int(sample.max())}, "
            f"which is >= vocab_size ({vocab_size})."
        )


def perplexity(loss: float) -> float:
    return math.exp(loss) if loss < 50 else float("inf")


class MetricLogger:
    def __init__(self, config: dict):
        self.wandb = None
        if config["wandb"]:
            import wandb
            self.wandb = wandb
            wandb.init(project=config["project"], name=config["run_name"], config=config, dir=config["out_dir"])

    def log(self, step: int, metrics: dict) -> None:
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()


@torch.no_grad()
def evaluate(model, data, batch_size: int, context_length: int, device: str, num_batches: int) -> float:
    model.eval()
    rng_state = np.random.get_state()
    np.random.seed(1234)

    total_loss = 0.0
    
    for _ in range(num_batches):
        inputs, targets = get_batch(data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        total_loss += loss.item()

    np.random.set_state(rng_state)
    model.train()
    
    return total_loss / num_batches


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    
    model_cfg = config["model"]
    optimizer_cfg = config["optimizer"]
    training_cfg = config["training"]
    data_cfg = config["data"]
    
    device = pick_device(config["device"])
    torch.manual_seed(training_cfg["seed"])
    np.random.seed(training_cfg["seed"])
    
    
    batch_size = training_cfg["batch_size"]
    context_length = training_cfg["context_length"]
    tokens_per_step = batch_size * context_length
    
    max_steps = training_cfg["max_steps"]
    if max_steps is None:
        max_steps = training_cfg["total_tokens"] // tokens_per_step
	
    min_lr = optimizer_cfg["min_lr"] if optimizer_cfg["min_lr"] is not None else 0.1 * optimizer_cfg["lr"]
    
    if optimizer_cfg["warmup_iters"] >= max_steps:
        raise ValueError("warmup_iters must be smaller than max_steps")

    os.makedirs(config["out_dir"], exist_ok=True)
    
    with open(os.path.join(config["out_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    train_data = load_tokens(data_cfg["train_data"], data_cfg["data_dtype"])
    val_data = load_tokens(data_cfg["val_data"], data_cfg["data_dtype"])
    validate_data(train_data, "train data", model_cfg["vocab_size"], context_length)
    validate_data(val_data, "val data", model_cfg["vocab_size"], context_length)

    model = TransformerLM(
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        theta=model_cfg["rope_theta"],
        vocab_size=model_cfg["vocab_size"],
        context_length=context_length,
        num_layers=model_cfg["num_layers"],
    ).to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_cfg["lr"],
        betas=(optimizer_cfg["beta1"], optimizer_cfg["beta2"]),
        weight_decay=optimizer_cfg["weight_decay"],
        eps=optimizer_cfg["eps"],
    )

    start_step = 0
    resume_path = config["resume"]
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer)
        print(f"resumed from {resume_path} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params / 1e6:.1f}M steps={max_steps} "
          f"tokens/step={tokens_per_step} total_tokens={max_steps * tokens_per_step:,}")

    last_ckpt = os.path.join(config["out_dir"], "ckpt_last.pt")
    best_ckpt = os.path.join(config["out_dir"], "ckpt_best.pt")
    logger = MetricLogger(config)

    best_val_loss = float("inf")
    last_eval_step = None
    log_start_time = time.time()

    model.train()
    
    for step in range(start_step, max_steps):
        lr = lr_cosine_schedule(step, optimizer_cfg["lr"], min_lr, optimizer_cfg["warmup_iters"], max_steps)
        
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(train_data, batch_size, context_length, str(device))
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        grad_norm = None
        if optimizer_cfg["grad_clip"] > 0:
            grad_norm = gradient_clipping(model.parameters(), optimizer_cfg["grad_clip"])
        optimizer.step()

        completed = step + 1
        
        if completed % config["log_interval"] == 0:
            train_loss = loss.item()
            tokens_per_sec = config["log_interval"] * tokens_per_step / (time.time() - log_start_time)
            metrics = {
                "train_loss": train_loss,
                "perplexity": perplexity(train_loss),
                "learning_rate": lr,
                "tokens_per_sec": tokens_per_sec
            	}
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm.item()
            logger.log(completed, metrics)
            print(f"step {completed}/{max_steps} | loss {train_loss:.4f} | ppl {metrics['perplexity']:.2f} | lr {lr:.2e} | {tokens_per_sec:,.0f} tok/s")

            log_start_time = time.time()
            
            if not math.isfinite(train_loss):
                raise RuntimeError("Training loss is not finite, stopping (diverged)")
            
        if completed % config["eval_interval"] == 0:
            val_loss = evaluate(model, val_data, batch_size, context_length, device, config["eval_batches"])
            val_ppl = perplexity(val_loss)
            last_eval_step = completed
            logger.log(completed, metrics)
            print(f"step {completed} | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, completed, best_ckpt)

        if completed % config["ckpt_interval"] == 0:
            save_checkpoint(model, optimizer, completed, last_ckpt)

    if last_eval_step != max_steps:
        val_loss = evaluate(model, val_data, batch_size, context_length, device, config["eval_batches"])
        val_ppl = perplexity(val_loss)
        logger.log(max_steps, metrics)
        print(f"final | val loss {val_loss:.4f} | val ppl {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, max_steps, best_ckpt)

    save_checkpoint(model, optimizer, max_steps, last_ckpt)
	
    logger.close()


if __name__ == "__main__":
    main()