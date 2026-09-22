import torch
from cs336_basics.module import Linear, Embedding, RMSNorm, TransformerBlock
from cs336_basics.utils import softmax

class TransformerLM(torch.nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, vocab_size: int, context_length: int, num_layers: int,
                 eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if vocab_size <= 0 or context_length <= 0:
            raise ValueError("vocab_size and context_length must be positive")
        
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)
        self.num_layers = int(num_layers)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.max_seq_len = int(context_length)
        self.theta = float(theta)

        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)

        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                theta=self.theta,
                max_seq_len=self.max_seq_len,
                eps=eps,
            ) for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model=self.d_model, eps=eps)

        self.lm_head = Linear(in_features=self.d_model, out_features=self.vocab_size) 

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 2:
            raise ValueError(f"expected input of shape (batch, seq_len), got {tuple(input.shape)}")
        
        batch, seq_len = input.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"input seq_len ({seq_len}) exceeds model context_length ({self.context_length})"
            )
        
        token_positions = torch.arange(seq_len, device=input.device, dtype=torch.long)
        x = self.token_embeddings(input)

        for block in self.layers:
            x = block(x, token_positions)

        x = self.ln_final(x)

        out = self.lm_head(x)
        return out
        