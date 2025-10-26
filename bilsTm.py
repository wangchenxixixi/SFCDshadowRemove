import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super(BiLSTMBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)  # 先归一化
        # 双向LSTM：hidden_size=dim//2（双向合并后总维度=dim，与输入匹配）
        self.bilstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            bidirectional=True,  # 双向模式
            batch_first=True,  # 输入格式：(batch, seq_len, dim)
            bias=bias
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)  # FFN前归一化
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)  # 保留原FFN

    def forward(self, x):
        # x: 4D特征 (batch, dim, h, w)
        residual = x  # 残差连接
        h, w = x.shape[-2:]  # 获取特征图尺寸

        # 1. 归一化 + BiLSTM计算（需先转3D序列）
        x = self.norm1(x)  # 4D → 4D（LayerNorm内部自动转3D计算再还原）
        x_seq = to_3d(x)  # 4D → 3D：(batch, h×w, dim)（seq_len=h×w）
        x_seq, _ = self.bilstm(x_seq)  # LSTM计算：(batch, h×w, dim)
        x = to_4d(x_seq, h, w)  # 3D → 4D：还原为图像特征维度

        # 2. 残差连接 + FFN
        x = x + residual  # 第一部分残差（BiLSTM输出 + 输入）
        residual2 = x
        x = self.norm2(x)  # FFN前归一化
        x = self.ffn(x)  # FFN特征映射
        x = x + residual2  # 第二部分残差（FFN输出 + 归一化后输入）

        return x


# ---------------------- 下采样/上采样/图像嵌入（未修改） ----------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)  # 下采样：尺寸减半，通道数翻倍
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)  # 上采样：尺寸翻倍，通道数减半
        )

    def forward(self, x):
        return self.body(x)



class Restormer_BiLSTM(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],  # 各层级Block数量（与原模型一致）
                 num_refinement_blocks=4,
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  # 保持原归一化类型
                 dual_pixel_task=False  # 双像素任务适配（未修改）
                 ):
        super(Restormer_BiLSTM, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # ---------------------- 编码器（Encoder）：BiLSTMBlock替代TransformerBlock ----------------------
        # Level 1：dim=48
        self.encoder_level1 = nn.Sequential(*[
            BiLSTMBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # Level 2：dim=48×2=96（下采样后通道翻倍）
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        # Level 3：dim=48×4=192
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        # Level 4（Latent）：dim=48×8=384
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 3), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        # Level 3解码：上采样后通道减半，concat编码器特征后降维
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        # Level 2解码
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        # Level 1解码
        self.up2_1 = Upsample(int(dim * 2 ** 1))
        self.decoder_level1 = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        # 精修层（Refinement）
        self.refinement = nn.Sequential(*[
            BiLSTMBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                        LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)
        ])

        # 双像素任务适配（未修改）
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim * 2 ** 1), kernel_size=1, bias=bias)

        # 输出层（未修改）
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # ---------------------- 编码器前向 ----------------------
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # ---------------------- 解码器前向 ----------------------
        # Level 4 → Level 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)  # 跨层连接（编码器特征）
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # Level 3 → Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # Level 2 → Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # 精修层
        out_dec_level1 = self.refinement(out_dec_level1)

        # 输出层（双像素任务/普通任务分支）
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img  # 残差连接（输入图像）

        return out_dec_level1