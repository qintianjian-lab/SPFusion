import torch
import torch.nn as nn
from timm.models.layers import DropPath


class FFNLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels))
        self.bias = nn.Parameter(torch.zeros(1, num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1) * x + self.bias.unsqueeze(-1)
        return x


class FFN(nn.Module):
    def __init__(
            self,
            in_dim: int,
            in_channel: int,
            dropout: float,
            out_dim: int = None,
            expansion_ratio: int = 4,
    ):
        super().__init__()
        self.ln = FFNLayerNorm(in_channel)
        self.ffn = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=in_channel * expansion_ratio,
                kernel_size=1,
                stride=1,
                padding='same',
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=in_channel * expansion_ratio,
                out_channels=in_channel,
                kernel_size=1,
                stride=1,
                padding='same',
            ),
            nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
                bias=False,
            ) if out_dim is not None else nn.Identity(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(self.ln(x))


class SoftmaxOne(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x - x.max(dim=self.dim, keepdim=True).values
        exp_x = torch.exp(x)
        return exp_x / (1 + exp_x.sum(dim=self.dim, keepdim=True))


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_feature_dim: int,
            hidden_channel: int,
            num_heads: int,
            embedding_enable: bool = False,
            dropout: float = 0.1,
            softmax_one: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.softmax = SoftmaxOne(dim=-1) if softmax_one else nn.Softmax(dim=-1)
        self.ln_q = FFNLayerNorm(hidden_channel)
        self.ln_k = FFNLayerNorm(hidden_channel)
        self.ln_v = FFNLayerNorm(hidden_channel)
        self.linear_q = nn.Linear(hidden_feature_dim, hidden_feature_dim) if embedding_enable else nn.Identity()
        self.linear_k = nn.Linear(hidden_feature_dim, hidden_feature_dim) if embedding_enable else nn.Identity()
        self.linear_v = nn.Linear(hidden_feature_dim, hidden_feature_dim) if embedding_enable else nn.Identity()
        self.dropout = DropPath(dropout) if dropout > 0 else nn.Identity()
        self.hidden_feature_dim = hidden_feature_dim

    def forward(self, q, k, v):
        assert q.shape == k.shape == v.shape, '[Error] q, k, v must have the same shape'
        assert self.hidden_feature_dim % self.num_heads == 0, \
            f'[Error] hidden_feature_dim: {self.hidden_feature_dim} must be divisible by num_heads: {self.num_heads}'
        b, c, _ = q.shape
        # b, h, d, n -> b, h, n, d
        q, k, v = self.ln_q(q), self.ln_k(k), self.ln_v(v)
        q = self.linear_q(q).reshape(b, self.num_heads, c, self.hidden_feature_dim // self.num_heads)
        k = self.linear_k(k).reshape(b, self.num_heads, c, self.hidden_feature_dim // self.num_heads)
        v = self.linear_v(v).reshape(b, self.num_heads, c, self.hidden_feature_dim // self.num_heads)
        attn = (q / ((self.hidden_feature_dim // self.num_heads) ** 0.5)) @ k.transpose(-2, -1)
        attn_weights = self.softmax(attn)
        attn = self.dropout(attn_weights) @ v
        attn = attn.transpose(1, 2).reshape(b, c, -1)
        return attn, attn_weights

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TransBlock(nn.Module):
    def __init__(
            self,
            in_channel: int,
            embedding_dim: int,
            num_heads: int,
            out_dim: int = None,
            embedding_enable: bool = False,
            dropout: float = 0.1,
            softmax_one: bool = True,
            expansion_ratio: int = 4,
            poolformer: bool = False,
    ):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(
            hidden_feature_dim=embedding_dim,
            hidden_channel=in_channel,
            num_heads=num_heads,
            embedding_enable=embedding_enable,
            dropout=dropout,
            softmax_one=softmax_one,
        ) if not poolformer else nn.AvgPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        self.ffn = FFN(
            in_dim=embedding_dim,
            in_channel=in_channel,
            out_dim=out_dim,
            dropout=dropout,
            expansion_ratio=expansion_ratio,
        )
        self.out_dim = out_dim
        self.poolformer = poolformer

    def forward(self, x1, x2=None, x3=None):
        assert (x2 is not None and x3 is not None) or (x2 is None and x3 is None), \
            '[Error] input: x2, x3 must be all None or not None at the same time'
        if x2 is not None and x3 is not None:
            x = torch.concat([x1, x2, x3], dim=1)
        elif x2 is None and x3 is None:
            x = x1
        else:
            raise '[Error] input: x2, x3 must be all None or not None at the same time'
        attn_weights = None
        if self.poolformer:
            x_attn = self.mhsa(x)
        else:
            x_attn, attn_weights = self.mhsa(x, x, x)
            # if x2 is not None and x3 is not None:
            #     _tmp = attn_weights.detach().cpu().numpy()
            #     print(_tmp.shape)
            #     print(_tmp[5][0][48])
        x = x + x_attn
        x_ffn = self.ffn(x)
        if self.out_dim is None:
            return (x + x_ffn), attn_weights
        else:
            return x_ffn, attn_weights
