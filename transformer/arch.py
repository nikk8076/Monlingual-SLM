import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) layer.
    Simplifies LayerNorm by scaling inputs based on root mean square, 
    which is computationally more efficient and stable.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Input tensor shape: (batch_size, sequence_length, embed_dim)
        # Output tensor shape: (batch_size, sequence_length, embed_dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # Input tensor shape: (batch_size, sequence_length, embed_dim)
        # Returns normalized float tensor scaled by weight parameter
        return self._norm(x.float()).type_as(x) * self.weight

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) generator.
    Precomputes Llama-style frequency parameters for fast cache lookup during attention.
    """
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies: shape (head_dim // 2)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        # Angle frequencies outer product: shape (sequence_length, head_dim // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate each frequency to match pair rotations: shape (sequence_length, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False) # shape (sequence_length, head_dim)
        self.register_buffer("sin_cached", emb.sin(), persistent=False) # shape (sequence_length, head_dim)

    def forward(self, x, seq_len=None):
        # Input tensor x shape: (batch_size, sequence_length, embed_dim)
        current_seq_len = seq_len if seq_len is not None else x.shape[2]
        if current_seq_len > self.max_seq_len:
            self._set_cos_sin_cache(current_seq_len)
        # Returns cosine and sine slices matching sequence length: shape (sequence_length, head_dim)
        return self.cos_cached[:current_seq_len], self.sin_cached[:current_seq_len]

def rotate_half(x):
    """Rotates half the dimensions of the input tensor."""
    # Input shape: (batch_size, num_heads, sequence_length, head_dim)
    x1 = x[..., :x.shape[-1] // 2] # shape (batch_size, num_heads, sequence_length, head_dim // 2)
    x2 = x[..., x.shape[-1] // 2:] # shape (batch_size, num_heads, sequence_length, head_dim // 2)
    # Output shape: (batch_size, num_heads, sequence_length, head_dim)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies precomputed rotary position embeddings to query and key tensors."""
    # Input q and k shapes: (batch_size, num_heads, sequence_length, head_dim)
    # Input cos and sin shapes: (sequence_length, head_dim)
    
    # Reshape cos/sin to align with 4D attention tensors: (1, 1, sequence_length, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    
    # Apply rotation matrices
    # Output query and key shapes: (batch_size, num_heads, sequence_length, head_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MultiHeadCausalAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism with Rotary Position Embeddings (RoPE).
    Computes scaled dot-product attention with causal masking.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        # Input x shape: (batch_size, sequence_length, embed_dim)
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear projections
        # Output shapes: (batch_size, sequence_length, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape and transpose to isolate attention heads
        # Split shape: (batch_size, sequence_length, num_heads, head_dim)
        # Transposed shape: (batch_size, num_heads, sequence_length, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply Rotary Position Embeddings (RoPE)
        # Output shapes: (batch_size, num_heads, sequence_length, head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute raw dot-product attention scores
        # Output scores shape: (batch_size, num_heads, sequence_length, sequence_length)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Construct and apply causal mask to prevent attending to future tokens
        # Causal mask shape: (1, 1, sequence_length, sequence_length)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Compute normalized softmax probabilities
        # Attention weights shape: (batch_size, num_heads, sequence_length, sequence_length)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Weighted sum over value vectors
        # Context matrix shape: (batch_size, num_heads, sequence_length, head_dim)
        context = torch.matmul(attn_probs, v)
        
        # Concatenate attention heads back into a single vector
        # Transposed shape: (batch_size, sequence_length, num_heads, head_dim)
        # Re-assembled shape: (batch_size, sequence_length, embed_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output linear projection
        # Return output shape: (batch_size, sequence_length, embed_dim)
        output = self.resid_dropout(self.out_proj(context))
        
        return output, attn_probs

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) sublayer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x shape: (batch_size, sequence_length, embed_dim)
        # Intermediate projected shape: (batch_size, sequence_length, expansion_dim)
        # Output projected shape: (batch_size, sequence_length, embed_dim)
        return self.dropout(self.w_2(self.activation(self.w_1(x))))

class TransformerBlock(nn.Module):
    """
    Single Decoder Transformer Block using Pre-RMSNorm formulation.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln_1 = RMSNorm(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, dropout)
        self.ln_2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, cos, sin):
        # Input x shape: (batch_size, sequence_length, embed_dim)
        
        # Pre-Norm Causal Attention block (residual addition)
        attn_out, attn_weights = self.attn(self.ln_1(x), cos, sin)
        x = x + attn_out # shape (batch_size, sequence_length, embed_dim)
        
        # Pre-Norm FFN block (residual addition)
        x = x + self.ffn(self.ln_2(x)) # shape (batch_size, sequence_length, embed_dim)
        
        return x, attn_weights

class Transformer(nn.Module):
    """
    Scratch-built GPT-style Decoder-only Language Model with RoPE and RMSNorm.
    Supports checkpoint saving, weight tying, and attention visualization outputs.
    """
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Embedding layer (Positional embedding removed, handled by RoPE)
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)
        
        # Rotary embedding helper
        self.rope = RoPE(dim=self.d_k, max_seq_len=max_seq_len)
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        
        # Final Norm layer (RMSNorm)
        self.ln_f = RMSNorm(d_model)
        
        # Output language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Enable parameter-saving via weight-tying if requested
        if tie_weights:
            self.lm_head.weight = self.token_embeddings.weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Special initialization for residual linear layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, targets=None):
        # Input input_ids shape: (batch_size, sequence_length)
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds context length {self.max_seq_len}"
        
        # Retrieve precomputed RoPE cos/sin frequencies: shape (sequence_length, head_dim)
        cos, sin = self.rope(input_ids, seq_len=seq_len)
        
        # Apply token embeddings: shape (batch_size, sequence_length, embed_dim)
        x = self.token_embeddings(input_ids)
        x = self.emb_dropout(x)
        
        # Run blocks and collect attention matrices for visualization
        all_attns = []
        for block in self.blocks:
            # Output x shape: (batch_size, sequence_length, embed_dim)
            x, attn_weights = block(x, cos, sin)
            all_attns.append(attn_weights)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # shape (batch_size, sequence_length, vocab_size)
        
        # Compute loss if targets are provided (Causal Language Modeling objective)
        loss = None
        if targets is not None:
            # Shift logits and targets for next-token prediction loss
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                targets.view(-1)
            )
            
        return logits, loss, all_attns
