import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) -> (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            # (B, 128, H, W) -> (B, 128, H / 2, W / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # (B, 128, H / 2, W / 2) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(128, 256),
            # (B, 256, H, W) -> (B, 256, H / 2, W / 2)
            VAE_ResidualBlock(256, 256),
            # (B, 256, H / 2, W / 2) -> (B, 256, H / 4, W / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # (B, 256, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(256, 512),
            # (B, 512, H / 4, W / 4) -> (B, 512, H / 4, W / 4)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 4, W / 4) -> (B, 512, H / 8, W / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_AttentionBlock(512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            VAE_ResidualBlock(512, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            nn.GroupNorm(32, 512),
            # (B, 512, H / 8, W / 8) -> (B, 512, H / 8, W / 8)
            nn.SiLU(),
            # (B, 512, H / 8, W / 8) -> (B, 8, H / 8, W / 8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # (B, 8, H / 8, W / 8) -> (B, 8, H / 8, W / 8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H , W)
        # noise: (B, out_channels, H / 8 , W / 8)
        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # (Pad_left, Pad_right, Pad_top, Pad_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        # (B, 8, H / 8, W / 8) -> 2 x (B, 4, H / 8, W / 8)
        mean, logvar = torch.chunk(x, 2, dim=1)
        # (B, 4, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
        logvar = torch.clamp(logvar, -30, 20)
        # (B, 4, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
        var = logvar.exp()
        # (B, 4, H / 8, W / 8) -> (B, 4, H / 8, W / 8)
        stdev = var.sqrt()

        # Z = N(0, 1) -> X = N(mean, variance)
        # X = mean + stdev * Z
        x = mean + stdev * noise

        #Scale output
        x *= 0.18215

        return x
