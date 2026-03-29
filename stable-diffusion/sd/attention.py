import torch
import torch.nn as nn
import torch.nn.functional as F


###########################################
#             Self-Attention              #
###########################################

class SelfAttention(nn.Module):
    def __init__(self, n_heads : int, embed_dim : int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        # combine all Q, K, V matrices into one
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=in_proj_bias)
        # out projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

        # number af heads to use in attention
        self.n_heads = n_heads

        # dimension of each head
        self.head_dim = embed_dim // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (B, T, C)
        input_shape = x.shape
        B, T, C = input_shape

        # these is the intermidiate shape with heads representation
        intermid_shape = (B, T, self.n_heads, self.head_dim)

        # get query, keys and values
        # # x: (B, T, C) -> (B, T, C * 3) -> 3 x (B, T, C)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # reshape them into shape that is fit to head-wise operations
        # (B, T, C) -> (B, T, H, C / H) -> (B, H, T, C / H)
        q = q.view(intermid_shape).transpose(1, 2)
        k = k.view(intermid_shape).transpose(1, 2)
        v = v.view(intermid_shape).transpose(1, 2)

        # getting attention matrix
        wei = q @ k.transpose(-1, -2)

        # if we using causal modelling than apply mask to hide future information
        if causal_mask:
            # upper triangle
            mask = torch.ones_like(wei, dtype=torch.bool).triu(1)
            wei = wei.masked_fill(mask, -torch.inf)
        
        # Normalizing according to the paper
        wei /= self.head_dim ** 0.5

        # Getting attention weights with softmax
        wei = F.softmax(wei, dim=-1)

        # Getting final token represetation
        # (B, H, T, T) @ (B, H, T, C / H) -> (B, H, T, C / H)
        out = wei @ v

        # Inverse process of reshaping
        # (B, H, T, C / H) - >(B, T, H, C / H)
        out = out.transpose(1, 2)

        out = out.reshape(input_shape)

        return self.out_proj(out)
    



###########################################
#             Cross-Attention             #
###########################################

class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, embed_dim: int, cross_dim:int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        # x(latent): (B, T_Q, dim_Q)
        # y(context): (B, T_K_V, dim_K_V) = (B, 77, 768)
        input_shape = x.shape
        B, T_Q, C_Q = input_shape

        intermid_shape = (B, -1, self.n_heads, self.head_dim)

        # (B, T_Q, H, H_dim)
        q = self.q_proj(x)
        # (B, T_K, H, H_dim)
        k = self.k_proj(y)
        # (B, T_K, H, H_dim)
        v = self.v_proj(y)

        # (B, H, T_Q, H_dim)
        q = q.view(intermid_shape).transpose(1, 2)
        # (B, H, T_K, H_dim)
        k = k.view(intermid_shape).transpose(1, 2)
        # (B, H, T_K, H_dim)
        v = v.view(intermid_shape).transpose(1, 2)

        # (B, H, T_Q, H_dim) @ (B, H, H_dim, T_K) = (B, H, T_Q, T_K)
        wei = q @ k.transpose(-1, -2)

        wei /= self.head_dim ** 0.5


        wei = F.softmax(wei, dim=-1)

        # (B, H, T_Q, T_K) @ (B, H, T_K, H_dim) = (B, H, T_Q, H_dim)
        out = wei @ v

        out = out.transpose(1, 2).contiguous()

        out = out.view(input_shape)

        return self.out_proj(out)