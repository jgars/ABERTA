import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B*F, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerAttentionPooling(nn.Module):
    def __init__(self, dim_in=1024, dim_out=768, num_heads=8, use_posenc=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim_in,
            num_heads=num_heads,
            batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, dim_in))
        self.proj = nn.Linear(dim_in, dim_out)
        self.posenc = PositionalEncoding(dim_in) if use_posenc else None

    def forward(self, x, key_padding_mask=None):
        B, F, T, D = x.shape
        x_flat = x.reshape(B * F, T, D)

        if self.posenc is not None:
            x_flat = self.posenc(x_flat)

        q = self.query.expand(B * F, 1, D)

        if key_padding_mask is not None:
            kpm = key_padding_mask.reshape(B * F, T) # bool, True=PAD
            all_pad = kpm.all(dim=1)  # (B*F,)

            out = x_flat.new_zeros((B * F, 1, D))
            attn_weights = x_flat.new_zeros((B * F, 1, T))

            ok = ~all_pad
            if ok.any():
                out_ok, attn_ok = self.mha(
                    q[ok], x_flat[ok], x_flat[ok],
                    key_padding_mask=kpm[ok]
                )
                out[ok] = out_ok
                attn_weights[ok] = attn_ok

        else:
            out, attn_weights = self.mha(q, x_flat, x_flat)

        out = out.squeeze(1)          # (B*F, D)
        out = self.proj(out)          # (B*F, dim_out)
        out = out.reshape(B, F, -1)

        return out, attn_weights
    

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        use_posenc: bool = False,
        max_len: int = 5000,
        batch_first = True,
    ):
        super().__init__()

        # Optional positional encoding
        self.use_posenc = use_posenc
        if use_posenc:
            self.positional_encoding = PositionalEncoding(d_model, max_len)
        else:
            self.positional_encoding = None

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first
        )

        # Feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Activación no soportada: {activation}")

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: (batch, seq_len, d_model)
        """
        # Apply positional encoding if it is enabled
        if self.positional_encoding is not None:
            src = self.positional_encoding(src)

        # --- Self-Attention ---
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # --- Feed-Forward ---
        ff = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + ff
        src = self.norm2(src)

        return src, attn_weights