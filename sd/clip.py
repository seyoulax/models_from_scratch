import torch
import torch.nn as nn
import torch.nn.functional as F

from sd.attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, embed_dim: int, max_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_tokens, embed_dim))

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        # (B, T) -> (B, T, C)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x
    

class CLIPLayer(nn.Module):

    def __init__(self, n_heads: int, embed_dim: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(n_heads, embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x

        # self-attention

        x = self.layernorm_1(x)
        
        x = self.attention(x, causal_mask=True)

        x += residual

        # feed forward
        residual = x

        x = self.layernorm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU

        x = self.linear_2(x)

        return x + residual


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77) # (77) - max seq len

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:

        tokens = tokens.type(torch.long)

        # (B, T) -> (B, T, C) 
        state = self.embedding(tokens)

        for module in self.layers:
            state = module(state)

        # (B, T, C)
        return self.layernorm(state)