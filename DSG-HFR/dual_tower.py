import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightcnn import LightCNN_29v2, CosineMarginProduct


def create_model(model_type, model_path, device, num_classes):
    if model_type == "DualTowerTransformer":
        model = DualTowerTransformer(model_path, device, num_classes)
    else:
        model = DualTowerModel(model_path, device, num_classes)
    return model


class DualTowerModel(nn.Module):
    def __init__(self, model_path, device, num_classes=80013):
        super(DualTowerModel, self).__init__()
        self.vis = LightCNN_29v2(model_path, device)
        self.inf = LightCNN_29v2(model_path, device)
        self.weight = CosineMarginProduct(256, num_classes)

    def eval_(self, x, mode='vis'):
        if mode == 'vis':
            return F.normalize(self.vis(x), p=2, dim=1)
        else:
            return F.normalize(self.inf(x), p=2, dim=1)

    def forward(self, x, y, l=None, flag=0):
        fc_vis = self.vis(x)
        fc_inf = self.inf(y)
        if flag == 0:
            out1 = self.weight(fc_vis, l)
            out2 = self.weight(fc_inf, l)
            return out1, out2, F.normalize(fc_vis, p=2, dim=1), F.normalize(fc_inf, p=2, dim=1)
        else:
            return F.normalize(fc_vis, p=2, dim=1), F.normalize(fc_inf, p=2, dim=1)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分多头
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attn = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.matmul(attn, v)

        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        return self.w_o(output), attn


class FeedForward(nn.Module):
    """位置wise前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数

    def forward(self, x):
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Encoder层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class DualTowerTransformer(nn.Module):
    def __init__(self, model_path, device, num_classes=80013):
        super(DualTowerTransformer, self).__init__()
        self.vis = LightCNN_29v2(model_path, device)
        self.inf = LightCNN_29v2(model_path, device)
        self.transformer = nn.Sequential(
            EncoderLayer(128, 8, 256),
            EncoderLayer(128, 8, 256),
        )
        self.fc = nn.Linear(128*64, 256)
        self.weight = CosineMarginProduct(256, num_classes)

    def eval_(self, x, mode='vis'):
        if mode == 'vis':
            fc = self.vis.conv_only(x)
        else:
            fc = self.inf.conv_only(x)

        B, C = fc.shape[:2]
        fc = fc.view(B, C, -1).transpose(1, 2)
        fc = self.transformer(fc).view(B, -1)
        fc = self.fc(fc)
        return F.normalize(fc, p=2, dim=1)

    def forward(self, x, y, l=None, flag=0):
        fc_vis = self.eval_(x, mode='vis')
        fc_inf = self.eval_(y, mode='inf')
        if flag == 0:
            out1 = self.weight(fc_vis, l)
            out2 = self.weight(fc_inf, l)
            return out1, out2, F.normalize(fc_vis, p=2, dim=1), F.normalize(fc_inf, p=2, dim=1)
        else:
            return F.normalize(fc_vis, p=2, dim=1), F.normalize(fc_inf, p=2, dim=1)



