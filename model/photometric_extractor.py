import timm
import torch.nn as nn


class PhotometricExtractor(nn.Module):
    def __init__(
            self,
            in_channel: int,
            timm_out_indices: list,
            timm_model_name: str,
            timm_model_out_channel: int,
            timm_model_out_feature: int,
            hidden_out_channel: int = 32,  # following the spec feature extractor
            hidden_out_dim: int = 28,  # following the spec feature extractor
            pre_trained: bool = False,
    ):
        super().__init__()
        # in: (batch_size, 3, 128, 128)
        # out: (batch_size, 192, 8, 8)
        self.feature_extractor = timm.create_model(
            timm_model_name,
            pretrained=pre_trained,
            in_chans=in_channel,
            features_only=True,
            out_indices=timm_out_indices,
        )

        self.conv_blk = nn.Sequential(
            nn.Conv2d(
                in_channels=timm_model_out_channel,
                out_channels=hidden_out_channel,
                kernel_size=3,
                padding='same',
            ),
            nn.BatchNorm2d(hidden_out_channel),
            nn.Conv2d(
                in_channels=hidden_out_channel,
                out_channels=hidden_out_channel,
                kernel_size=3,
                padding='same',
            )
        )
        self.linear_blk = nn.Sequential(
            nn.GELU(),
            nn.Linear(
                in_features=timm_model_out_feature, out_features=hidden_out_dim
            )
        )

    def forward(self, x):
        x = self.feature_extractor(x)[0]
        x = self.conv_blk(x)
        b, c, _, _ = x.shape
        x = x.view(b, c, -1)
        x = self.linear_blk(x)
        return x
