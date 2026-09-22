import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import load_checkpoint
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import pick_device, softmax, parse_args, load_config


def top_p_sampling(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the range (0, 1]")

    probabilities = softmax(logits / temperature, dim=-1)

    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)

        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        remove = cumulative_probabilities - sorted_probabilities > top_p
        sorted_probabilities[remove] = 0.0

        probabilities = torch.zeros_like(probabilities)
        probabilities.scatter_(0, sorted_indices, sorted_probabilities)

        probabilities /= probabilities.sum()

    return int(torch.multinomial(probabilities, num_samples=1).item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> list[int]:
    """Generate up to max_new_tokens, returning prompt and sampled token IDs."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not prompt_ids:
        raise ValueError("prompt_ids must contain at least one token")
    if eos_token_id is not None and prompt_ids[-1] == eos_token_id:
        return prompt_ids

    device = next(model.parameters()).device
    output_ids = prompt_ids.copy()

    model.eval()
    for _ in range(max_new_tokens):
        input_ids = torch.tensor(
            [output_ids[-model.context_length:]], device=device, dtype=torch.long
        )

        logits = model(input_ids)[0, -1]

        next_token = top_p_sampling(logits, temperature, top_p)
        output_ids.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return output_ids


def load_tokenizer(config: dict) -> Tokenizer:
    EOT = "<|endoftext|>"
    tokenizer = Tokenizer.from_files(
        config["data"]["vocab_path"], 
        config["data"]["merges_path"], 
        special_tokens=[EOT])
    return tokenizer


def load_model(config, device: torch.device) -> TransformerLM:
    model_config = config["model"]
    model = TransformerLM(
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        theta=model_config["rope_theta"],
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
        num_layers=model_config["num_layers"],
    ).to(device)
    return model


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = pick_device(config["device"])
    
    tokenizer = load_tokenizer(config)

    model = load_model(config, device)

    load_checkpoint(config["data"]["ckpt_path"], model=model)

    eos_token_id = tokenizer.bytes_to_vocab_seq.get(config["eos_token"].encode("utf-8"))
    prompt = args.prompt if args.prompt is not None else "Once upon a time"

    output_ids = generate(
        model,
        tokenizer.encode(prompt),
        config["max_new_tokens"],
        config["temperature"],
        config["top_p"],
        eos_token_id,
    )
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
