import torch.nn as nn
from timm.models.layers import DropPath


class StarFusionConvBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: int,
            groups: int = 1,
            with_bn: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=kernel_size,
                stride=1,
                padding='same',
                groups=groups,
            ),
            nn.BatchNorm1d(out_channel) if with_bn else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class StarFusion(nn.Module):
    def __init__(
            self,
            in_channel: int,
            kernel_size: int,
            dropout: float,
            mlp_ratio: int = 4,
            act_in_photo: bool = False,  # when True act layer will be in photometric input, or in spec input
    ):
        super().__init__()
        self.photo_branch = nn.Sequential(
            StarFusionConvBlock(
                in_channel=in_channel,
                out_channel=in_channel,
                kernel_size=kernel_size,
                groups=in_channel,
                with_bn=True,
            ),
            StarFusionConvBlock(
                in_channel=in_channel,
                out_channel=in_channel * mlp_ratio,
                kernel_size=kernel_size,
                with_bn=False,
            ),
        )
        self.spec_branch = nn.Sequential(
            StarFusionConvBlock(
                in_channel=in_channel,
                out_channel=in_channel,
                kernel_size=kernel_size,
                groups=in_channel,
                with_bn=True,
            ),
            StarFusionConvBlock(
                in_channel=in_channel,
                out_channel=in_channel * mlp_ratio,
                kernel_size=kernel_size,
                with_bn=False,
            ),
        )
        self.act = nn.GELU()
        self.act_in_photo = act_in_photo
        self.fusion = nn.Sequential(
            StarFusionConvBlock(
                in_channel=in_channel * mlp_ratio,
                out_channel=in_channel,
                kernel_size=kernel_size,
                with_bn=True,
            ),
            StarFusionConvBlock(
                in_channel=in_channel,
                out_channel=in_channel,
                kernel_size=kernel_size,
                groups=in_channel,
                with_bn=False,
            ),
        )
        self.dropout = DropPath(dropout) if dropout > 0 else nn.Identity()

    def forward(self, photo, spec):
        photo = self.photo_branch(photo)
        spec = self.spec_branch(spec)
        if self.act_in_photo:
            x = self.act(photo) * spec
        else:
            x = photo * self.act(spec)
        return self.dropout(self.fusion(x))
