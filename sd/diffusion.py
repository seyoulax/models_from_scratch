import torch
import torch.nn as nn
import torch.nn.functional as F

from sd.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # (1, 320 * 4 = 1280)
        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(time_dim, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (B, C, H, W)
        # time: (1, 1280)

        residual = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        # feature: (B, out_channels, H, W) + (1, out_channels, 1, 1)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residual)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int, context_dim: int = 768):
        super().__init__()
        channels = n_heads * embed_dim

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_heads, channels, in_proj_bias=False)

        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_heads, channels, context_dim, in_proj_bias=False
        )

        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, channels * 4 * 2)
        self.linear_geglu_2 = nn.Linear(channels * 4, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # context: (B, T, context_dim)

        residual_long = latent

        latent = self.groupnorm(latent)
        latent = self.conv_input(latent)

        B, C, H, W = latent.shape

        # (B, C, H, W) -> (B, C, H * W)
        latent = latent.view(B, C, H * W)
        latent = latent.transpose(-1, -2)

        # Norm & self-attention with skip-connection
        residual_short = latent
        latent = self.layernorm_1(latent)
        latent = self.attention_1(latent)
        latent += residual_short

        residual_short = latent

        # Norm & cross-attention with skip-connection
        latent = self.layernorm_2(latent)
        latent = self.attention_2(latent, context)

        latent += residual_short

        residual_short = latent

        # Norm & FF with GeGLU and skip connection
        latent = self.layernorm_3(latent)

        latent, gate = self.linear_geglu_1(latent).chunk(2, dim=-1)
        latent = latent * F.gelu(gate)

        latent = self.linear_geglu_2(latent)

        latent += residual_short

        latent = latent.transpose(-1, -2)

        latent = latent.contiguous().view(B, C, H, W)

        return self.conv_output(latent) + residual_long


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C, H * 2, W * 2)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    def forward(
        self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:

        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNET(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320),
                    UNET_AttentionBlock(8, 40),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 320),
                    UNET_AttentionBlock(8, 40),
                ),
                # (B, 320, H / 8, W / 8) -> (B, 640, H / 16, W / 16)
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(320, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 640),
                    UNET_AttentionBlock(8, 80),
                ),
                # (B, 640, H / 16, W / 16) -> (B, 1280, H / 32, W / 32)
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 1280),
                    UNET_AttentionBlock(8, 160),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280),
                    UNET_AttentionBlock(8, 160),
                ),
                # (B, 1280, H / 32, W / 32) -> (B, 1280, H / 64, W / 64)
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 1280),
                ),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # (B, 2560, H / 64, W / 64) -> (B, 1280, H / 64, W / 64)
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280),
                ),
                SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 1280),
                    UNET_AttentionBlock(8, 160),
                    Upsample(1280),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(960, 640),
                    UNET_AttentionBlock(8, 80),
                    Upsample(640),
                ),
                SwitchSequential(
                    UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNET_ResidualBlock(640, 320),
                    UNET_AttentionBlock(8, 40),
                ),
            ]
        )

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (B, T, C)
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, context, time)

        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 320, H / 8, W / 8)
        x = self.groupnorm(x)
        x = F.silu(x)
        # (B, 4, H / 8, W / 8)
        return self.conv(x)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        # latent: (B, 4, H / 8, W / 8)
        # context: (B, T, C)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
        out = self.unet(latent, context, time)

        # (B, 320, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
        out = self.final(out)

        return out
