import torch
import math
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.empty(self.out_features, self.in_features, dtype=self.dtype, device=self.device))
        W = torch.nn.init.trunc_normal_(self.W, mean=0, std=math.sqrt(2/(self.in_features+self.out_features)), a=-3*math.sqrt(2/(self.in_features+self.out_features)), b=3*math.sqrt(2/(self.in_features+self.out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.W = torch.nn.Parameter(torch.empty(self.num_embeddings, self.embedding_dim, dtype=self.dtype, device=self.device))
        torch.nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5, device: torch.device=None, dtype: torch.dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g_i = torch.nn.Parameter(torch.ones(self.d_model, dtype=self.dtype, device=self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_a = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        result = x * self.g_i / rms_a
        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = Linear(self.d_model, self.d_ff)
        self.W_2 = Linear(self.d_ff, self.d_model)
        self.W_3 = Linear(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor):
        w1x = self.W_1(x)
        return self.W_2(w1x * torch.sigmoid(w1x) * self.W_3(x))
    
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k = torch.arange(1, self.d_k // 2 + 1, device=self.device) #(d_k/2, )
        i = torch.arange(self.max_seq_len, device=self.device) #(max_seq_len, )
        freqs = theta ** (-(2*k-2)/self.d_k) #(d_k/2, )
        angles = torch.outer(i, freqs) #(max_seq_lenm d_k/2)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cache[token_positions] #(..., seq_len, d_k / 2)
        sin = self.sin_cache[token_positions] #(..., seq_len, d_k / 2)
        x1 = x[..., 0::2] #(..., seq_len, d_k/2)
        x2 = x[..., 1::2] #(..., seq_len, d_k/2)

        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos

        result = torch.stack([new_x1, new_x2], dim=-1) #(..., seq_len, d_k/2, 2)
        result = rearrange(result, "... d two -> ... (d two)") # (..., seq_len, d_k)
        return result

def softmax(x: torch.Tensor, i: int):
    x = x - torch.max(x, dim=i, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=i, keepdim=True)

def scaled_dot_product_attention(q, k, v, mask):
    d_k = q.shape[-1]
    atten = einsum(q, k, "batch ... q d_k, batch ... k d_k -> batch ... q k") / math.sqrt(d_k)
    if mask is not None:
        atten = atten.masked_fill(~mask, float("-inf")) # fill in those mask = False with inf let e^-inf=0 to avoid influence softmax

    atten = einsum(softmax(atten, -1), v, "batch ... q k, batch ... k d_v -> batch ... q d_v")
    return atten

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float=None, max_seq_len: int=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.W_q = Linear(self.d_model, self.num_heads*self.d_k)
        self.W_k = Linear(self.d_model, self.num_heads*self.d_k)
        self.W_v = Linear(self.d_model, self.num_heads*self.d_k)
        self.W_o = Linear(self.num_heads*self.d_k, self.d_model)
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
        else:
            self.rope = None
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self.W_q(x) #(..., seq_len, num_heads*d_k)
        k = self.W_k(x) #(..., seq_len, num_heads*d_k)
        v = self.W_v(x) #(..., seq_len, num_heads*d_k)
        q = rearrange(q, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads) #(..., num_heads, seq_len, d_k)
        k = rearrange(k, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads) #(..., num_heads, seq_len, d_k)
        v = rearrange(v, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads) #(..., num_heads, seq_len, d_k)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)) #(seq_len, seq_len)
        atten = scaled_dot_product_attention(q, k, v, mask) # (..., num_heads, seq_len, d_k)
        atten = rearrange(atten, "... h seq_len d_k -> ... seq_len (h d_k)") #(..., seq_len, num*heads*d_k)
        multi_atten = self.W_o(atten)
        return multi_atten
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float=None, max_seq_len: int=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Build layers
        self.rmsnorm1 = RMSNorm(self.d_model)
        self.rmsnorm2 = RMSNorm(self.d_model)
        self.mha = MultiHeadSelfAttention(self.d_model, self.num_heads, theta, max_seq_len)
        self.swiglu = SwiGLU(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)
        y = x + self.mha(self.rmsnorm1(x), token_positions)
        y = y + self.swiglu(self.rmsnorm2(y))
        return y
    
class TransformerLM(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int,context_length: int=None, theta: float=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        self.theta = theta

        # Build layers and blocks
        self.layers = torch.nn.ModuleList([
            TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.theta, self.context_length)
            for _ in range(self.num_layers)
        ])
        self.emb = Embedding(self.vocab_size, d_model)
        self.rmsnorm = RMSNorm(self.d_model)
        self.linear = Linear(d_model, vocab_size)


    def forward(self, x: torch.Tensor):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x)
        x = self.rmsnorm(x)
        x = self.linear(x)
        return x

        
    








        

