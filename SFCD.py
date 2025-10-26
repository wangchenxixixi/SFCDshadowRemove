import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange



def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if layer_norm_type == 'BiasFree' else WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class BiLSTMBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, layer_norm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, layer_norm_type)  # 特征归一化
        self.bilstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim // 2,  # 双向合并后维度=dim，与输入匹配
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # 输入格式：(batch, seq_len, dim)
            bias=bias
        )
        self.norm2 = LayerNorm(dim, layer_norm_type)  # FFN前归一化
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        residual = x
        h, w = x.shape[-2:]

        # 1. 归一化 + BiLSTM特征建模（4D→3D→4D）
        x = self.norm1(x)
        x_seq = to_3d(x)  # 转为序列格式：(batch, h*w, dim)
        x_seq, _ = self.bilstm(x_seq)
        x = to_4d(x_seq, h, w)  # 还原为图像特征格式

        # 2. 残差连接 + FFN特征增强
        x = x + residual
        residual2 = x
        x = self.norm2(x)
        x = self.ffn(x)
        return x + residual2


class ResBlock(nn.Module):
    """空域残差块（未修改核心逻辑）"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.main(x)


class ResBlock_fft_bench(nn.Module):
    """频域残差块（未修改核心逻辑）"""

    def __init__(self, in_channel, out_channel, norm='backward'):
        super().__init__()
        self.main_fft = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel * 2, kernel_size=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channel * 2, out_channel * 2, kernel_size=1, stride=1, bias=False)
        )
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        # 频域转换：空域→频域（实部+虚部分离）
        y = torch.fft.rfft2(x, norm=self.norm)
        y_f = torch.cat([y.real, y.imag], dim=1)
        # 频域特征处理
        y_f = self.main_fft(y_f)
        # 频域→空域（重组实部+虚部）
        y_real, y_imag = torch.chunk(y_f, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        return torch.fft.irfft2(y, s=(H, W), norm=self.norm)


class irnn_layer(nn.Module):
    """IRNN方向传播层（未修改，供SAM使用）"""

    def __init__(self, in_channels):
        super().__init__()
        self.weight = nn.ModuleDict({
            'left': nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False),
            'right': nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False),
            'up': nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False),
            'down': nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        })

    def forward(self, x):
        _, _, H, W = x.shape
        # 四方向传播计算
        top_left = x.clone()
        top_left[:, :, :, 1:] = F.relu(self.weight['left'](x)[:, :, :, :W - 1] + x[:, :, :, 1:])

        top_right = x.clone()
        top_right[:, :, :, :-1] = F.relu(self.weight['right'](x)[:, :, :, 1:] + x[:, :, :, :W - 1])

        top_up = x.clone()
        top_up[:, :, 1:, :] = F.relu(self.weight['up'](x)[:, :, :H - 1, :] + x[:, :, 1:, :])

        top_down = x.clone()
        top_down[:, :, :-1, :] = F.relu(self.weight['down'](x)[:, :, 1:, :] + x[:, :, :H - 1, :])

        return top_up, top_right, top_down, top_left


class Attention(nn.Module):
    """注意力权重预测层（未修改，供SAM使用）"""

    def __init__(self, in_channels):
        super().__init__()
        mid_dim = int(in_channels / 2)
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 4, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class SAM(nn.Module):
    """空间注意力模块（未修改核心逻辑）"""

    def __init__(self, in_channels, out_channels, use_attention=1):
        super().__init__()
        self.out_channels = out_channels
        self.use_attention = use_attention
        # 基础组件
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.irnn1 = irnn_layer(out_channels)
        self.irnn2 = irnn_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.relu2 = nn.ReLU(True)
        self.conv_out = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 注意力分支（可选）
        if self.use_attention:
            self.attention_layer = Attention(in_channels)

    def forward(self, x):
        # 注意力权重预测（若启用）
        weight = self.attention_layer(x) if self.use_attention else 1.0

        # 第一轮IRNN传播
        out = self.conv1(x)
        up1, right1, down1, left1 = self.irnn1(out)
        # 注意力加权
        if self.use_attention:
            up1 *= weight[:, 0:1, :, :]
            right1 *= weight[:, 1:2, :, :]
            down1 *= weight[:, 2:3, :, :]
            left1 *= weight[:, 3:4, :, :]
        out = self.conv2(torch.cat([up1, right1, down1, left1], dim=1))

        # 第二轮IRNN传播
        up2, right2, down2, left2 = self.irnn2(out)
        # 注意力加权
        if self.use_attention:
            up2 *= weight[:, 0:1, :, :]
            right2 *= weight[:, 1:2, :, :]
            down2 *= weight[:, 2:3, :, :]
            left2 *= weight[:, 3:4, :, :]
        out = self.conv3(torch.cat([up2, right2, down2, left2], dim=1))

        # 生成注意力掩码
        out = self.relu2(out)
        return self.sigmoid(self.conv_out(out))


class OverlapPatchEmbed(nn.Module):
    """图像块嵌入层（未修改，供BiLSTM输入使用）"""

    def __init__(self, in_c=3, embed_dim=32, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class SFCD(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,  # 与patch_embed输出维度一致
                 num_bilstm_blocks=4,  # 原encoder_level1的Block数量
                 ffn_expansion_factor=2.66,
                 bias=False,
                 layer_norm_type='WithBias'):

        super().__init__()
        # 1. 输入嵌入与BiLSTM特征建模
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.bilstm_encoder = nn.Sequential(*[
            BiLSTMBlock(dim, ffn_expansion_factor, bias, layer_norm_type)
            for _ in range(num_bilstm_blocks)
        ])

        # 2. 批量创建重复模块（简化17个ResBlock和FFT Block）
        self.res_blocks = nn.ModuleList([ResBlock(32, 32) for _ in range(17)])
        self.fft_blocks = nn.ModuleList([ResBlock_fft_bench(32, 32) for _ in range(17)])

        # 3. 注意力与输出层
        self.sam = SAM(32, 32, use_attention=1)  # 统一用一个SAM模块（原代码重复调用同一结构）
        self.conv_out = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # 步骤1：输入嵌入 + BiLSTM特征建模
        x = self.patch_embed(x)
        x = self.bilstm_encoder(x)

        # 步骤2：空域+频域残差融合（分4段注意力引导，保留原逻辑）
        # 段1：无注意力
        for i in range(3):
            x = F.relu(self.res_blocks[i](x) + x + self.fft_blocks[i](x))

        # 段2：Attention1引导
        att1 = self.sam(x)
        for i in range(3, 6):
            x = F.relu(self.res_blocks[i](x) * att1 + x + self.fft_blocks[i](x))

        # 段3：Attention2引导
        att2 = self.sam(x)
        for i in range(6, 9):
            x = F.relu(self.res_blocks[i](x) * att2 + x + self.fft_blocks[i](x))

        # 段4：Attention3引导
        att3 = self.sam(x)
        for i in range(9, 12):
            x = F.relu(self.res_blocks[i](x) * att3 + x + self.fft_blocks[i](x))

        # 段5：Attention4引导（作为输出的注意力返回）
        att4 = self.sam(x)
        for i in range(12, 15):
            x = F.relu(self.res_blocks[i](x) * att4 + x + self.fft_blocks[i](x))

        # 段6：无注意力（最后2个Block）
        for i in range(15, 17):
            x = F.relu(self.res_blocks[i](x) + x + self.fft_blocks[i](x))

        # 步骤3：输出重建
        out = self.conv_out(x)
        return att4, out  # 保留原输出：最后一个注意力权重 + 重建图像



class Generator(nn.Module):
    def __init__(self, gpu_ids=None):
        super().__init__()
        self.gpu_ids = gpu_ids if gpu_ids else []
        # 包装SFCD模型
        self.gen = nn.Sequential(OrderedDict([('sfcd_bilstm', SFCD())]))
        # 权重初始化（若models_utils存在则用自定义初始化，否则用默认）
        try:
            self.gen.apply(weights_init)
        except NameError:
            print("未导入weights_init，使用PyTorch默认权重初始化")

    def forward(self, x):
        # 多GPU并行（若启用）
        if self.gpu_ids and torch.cuda.is_available():
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)


if __name__ == "__main__":
    # 初始化模型（单GPU/CPU）
    model = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
    # 测试输入（batch=2, channel=3, height=64, width=64）
    x = torch.randn(2, 3, 64, 64).cuda() if torch.cuda.is_available() else torch.randn(2, 3, 64, 64)
    # 前向传播
    att4, out = model(x)
    # 验证输出维度
    print(f"输入维度: {x.shape}")
    print(f"注意力输出维度: {att4.shape} (应与输入H/W一致)")
    print(f"图像输出维度: {out.shape} (应与输入完全一致)")
    print("模型前向传播正常！")