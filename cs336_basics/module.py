import torch
import math
from einops import einsum, rearrange
from cs336_basics.utils import default_dff, silu, causal_mask, scaled_dot_product_attention

class Linear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(f"in_features and out_features must be positive, got {in_features}, {out_features}")

        self.in_features = int(in_features) #d_in
        self.out_features = int(out_features) #d_out
        self.weight = torch.nn.Parameter(
            torch.empty((self.out_features, self.in_features), device=device, dtype=dtype) # (d_out, d_in)
        )

        sigma = math.sqrt(2.0/(self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... i, o i -> ... o')


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(f"num_embeddings and embedding_dim must be positive, got {num_embeddings}, {embedding_dim}")

        self.num_embeddings = int(num_embeddings) # vocab_size
        self.embedding_dim = int(embedding_dim) # d_model
        self.weight = torch.nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype) # (vocab_size, d_model)
        )

        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None=None, dtype: torch.dtype | None=None):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(
            torch.ones((self.d_model), device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {x.shape[-1]}")
        
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

        result = (x / rms) * self.weight.to(torch.float32)

        return result.to(in_dtype)


class SwiGLU(torch.nn.Module):

    def __init__(self, d_model: int, d_ff: int, dim_multiple: int = 64,device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        
        self.d_model = int(d_model)
        self.d_ff = int(d_ff) if d_ff is not None else default_dff(d_model, dim_multiple)

        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RoPE(torch.nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"d_k must be a positive even number, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")


        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        inv_pow = self.theta ** (-torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32) / self.d_k)
        token_pos = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)

        angles = einsum(token_pos, inv_pow, 'i , j -> i j')

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_pos", cos, persistent=False)
        self.register_buffer("sin_pos", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape: (..., seq_len, d_k)
        # token_positions.shape: (..., seq_len)
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected last dim {self.d_k}, got {x.shape[-1]}")
        if token_positions.numel() and token_positions.max().item() >= self.max_seq_len:
            raise ValueError(
                f"token position {token_positions.max().item()} exceeds max_seq_len {self.max_seq_len} that RoPE tables were precomputed for")
        
        cos = self.cos_pos[token_positions]
        sin = self.sin_pos[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even*cos - x_odd*sin
        x_odd_rot = x_even*sin + x_odd*cos

        x_rot = torch.stack((x_even_rot, x_odd_rot), dim=-1)
        x_rot = x_rot.flatten(-2)

        return x_rot


class CausalMultiHeadSelfAttention(torch.nn.Module):

    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError(f"d_model and num_heads must be positive, got {d_model}, {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        self.head_dim = self.d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device, dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2] # T
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Rearrange [B, T, D] -> [B, H, T, Dh] (Dmodel = h*Dhead)
        Q = rearrange(Q, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)
        K = rearrange(K, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)
        V = rearrange(V, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)

        mask = causal_mask(seq_len, x.device)

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = rearrange(attn_out, '... h t d -> ... t (h d)', h = self.num_heads, d = self.head_dim)

        return self.output_proj(attn_out)

class CausalMultiHeadSelfAttentionWithRoPE(torch.nn.Module):

    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            theta: float,
            max_seq_len: int,
            device: torch.device | None=None, 
            dtype: torch.dtype | None=None):
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError(f"d_model and num_heads must be positive, got {d_model}, {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)

        self.head_dim = self.d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device, dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device, dtype)

        self.rope = RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2] # T
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Rearrange [B, T, D] -> [B, H, T, Dh] (Dmodel = h*Dhead)
        Q = rearrange(Q, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)
        K = rearrange(K, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)
        V = rearrange(V, '... t (h d) -> ... h t d', h = self.num_heads, d=self.head_dim)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        mask = causal_mask(seq_len, x.device)

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        attn_out = rearrange(attn_out, '... h t d -> ... t (h d)', h = self.num_heads, d = self.head_dim)
        
        return self.output_proj(attn_out)

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, 
                 eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_ff = int(d_ff)

        self.ln1 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttentionWithRoPE(self.d_model, self.num_heads, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.ln2 = RMSNorm(self.d_model, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sub_layer1 = self.attn(self.ln1(x), token_positions)
        x = x + sub_layer1
        sub_layer2 = self.ffn(self.ln2(x))
        x = x + sub_layer2

        return x