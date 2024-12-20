import torch
import torch.nn as nn

from model.convnext_1d import CONVNEXT1D
from model.photometric_extractor import PhotometricExtractor
from model.star_fusion import StarFusion
from model.trans_layers import TransBlock


class ClassificationHead(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            in_channel: int,
            hidden_channel: int,
            out_channel: int,
            reduction: int = 16,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.se_blk = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid(),
        )
        self.conv_blk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=hidden_channel,
                kernel_size=kernel_size,
                padding='same',
            ),
            nn.BatchNorm1d(hidden_channel),
        )
        self.out_linear = nn.Linear(hidden_channel, out_channel)

    def forward(self, x1, x2=None, x3=None):
        if x2 is not None and x3 is not None:
            x = torch.concat([x1, x2, x3], dim=1)
        else:
            x = x1
        b, c, _ = x.shape
        x_se_attn = self.se_blk(x).view(b, c, 1)
        # _tmp = x_se_attn.detach().squeeze().cpu().numpy()
        # print(_tmp.shape)
        # print(_tmp[5])
        x = x * x_se_attn
        x = self.conv_blk(x)
        x = x.mean(dim=-1)
        x = self.out_linear(x)
        return x, x_se_attn


class BuildModel(nn.Module):
    def __init__(
            self,
            spec_in_channel: int,
            photo_in_channel: int,
            star_fusion_kernel_size: int,
            star_fusion_mlp_ratio: int,
            star_fusion_act_in_photo: bool,
            dropout: float,
            fusion_blk_num_heads: int,
            trans_blk_num_heads: int,
            num_trans_blk: int,
            final_out_channel: int,
            poolformer: bool,

            softmax_one: bool = True,
            photo_timm_out_indices: list = None,
            photo_timm_model_name: str = 'resnet10t',
            photo_timm_model_out_channel: int = 32,
            photo_timm_model_out_feature: int = 64 * 64,
            spec_out_channel: int = 32,
            spec_out_dim: int = 28,
    ):
        super().__init__()
        self.spec_feature_extractor = CONVNEXT1D(
            in_channel=spec_in_channel,
        )
        self.photo_feature_extractor = PhotometricExtractor(
            in_channel=photo_in_channel,
            timm_model_name=photo_timm_model_name,
            timm_out_indices=photo_timm_out_indices,
            timm_model_out_channel=photo_timm_model_out_channel,
            timm_model_out_feature=photo_timm_model_out_feature,
            hidden_out_channel=spec_out_channel,
            hidden_out_dim=spec_out_dim,
        )
        self.star_fusion = StarFusion(
            in_channel=spec_out_channel,
            kernel_size=star_fusion_kernel_size,
            dropout=dropout,
            mlp_ratio=star_fusion_mlp_ratio,
            act_in_photo=star_fusion_act_in_photo,
        )
        self.fusion_trans_blk = TransBlock(
            in_channel=spec_out_channel * 3,
            embedding_dim=spec_out_dim,
            num_heads=fusion_blk_num_heads,
            embedding_enable=False,
            dropout=dropout,
            softmax_one=softmax_one,
            expansion_ratio=4,
            poolformer=False,
        )
        self.trans_blks = nn.ModuleList([
            TransBlock(
                in_channel=spec_out_channel * 3,
                embedding_dim=spec_out_dim,
                num_heads=trans_blk_num_heads,
                out_dim=spec_out_dim if index == num_trans_blk - 1 else None,
                embedding_enable=True,
                dropout=dropout,
                softmax_one=softmax_one,
                expansion_ratio=2,
                poolformer=poolformer,
            ) for index in range(num_trans_blk)
        ])
        self.classification_head = ClassificationHead(
            kernel_size=star_fusion_kernel_size,
            in_channel=spec_out_channel * 3,
            hidden_channel=spec_out_channel,
            out_channel=final_out_channel,
        )

    def forward(
            self,
            photo: torch.Tensor,
            spec: torch.Tensor,
    ):
        x_spec = self.spec_feature_extractor(spec)
        x_photo = self.photo_feature_extractor(photo)

        x_fusion = self.star_fusion(x_spec, x_photo)
        x, _attn_w = self.fusion_trans_blk(x_photo, x_fusion, x_spec)
        attn_weights = [_attn_w]
        for _layer in self.trans_blks:
            x, _attn_w = _layer(x)
            attn_weights.append(_attn_w)
        x, se_weights = self.classification_head(x)
        return x, attn_weights, se_weights
