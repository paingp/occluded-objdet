<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from collections import OrderedDict

# --- CBAM Block ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# --- Self-Attention Module ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- BoT Layer ---
class BottleneckTransformerLayer(nn.Module):
    def __init__(self, in_channels, heads=4, mlp_ratio=2):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.transformer = TransformerBlock(in_channels, heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_proj(x)
        x_flat = x.flatten(2).transpose(1, 2)
        x_trans = self.transformer(x_flat)
        x = x_trans.transpose(1, 2).view(B, C, H, W)
        return x

# --- Custom BoTNet Backbone with CBAM ---
class BoTNetBackbone(nn.Module):
    def __init__(self, in_channels=128, cbam_reduction=16, heads=4, mlp_ratio=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.botnet = BottleneckTransformerLayer(in_channels=128, heads=heads, mlp_ratio=mlp_ratio)
        self.cbam = CBAM(128, reduction=cbam_reduction)
        self.out_channels = 128

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.botnet(x)
        x = self.cbam(x)
        return x

# --- Custom CIFAR10 Model with BoTNet Backbone ---
class CustomCIFAR10Model(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# --- Custom Dataset Model with BoTNet Backbone ---
class CustomDataSetModel(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super(CustomDataSetModel, self).__init__()
        self.backbone = backbone
        # Adapt the classifier to the output channels of BoTNetBackbone
        self.classifier = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x):
        # The backbone returns a single feature map
        features = self.backbone(x)
        features = torch.mean(features, dim=[2, 3])  # Global Average Pooling
        logits = self.classifier(features)
        return logits

=======
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from collections import OrderedDict

# --- CBAM Block ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# --- Self-Attention Module ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.unbind(2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- BoT Layer ---
class BottleneckTransformerLayer(nn.Module):
    def __init__(self, in_channels, heads=4, mlp_ratio=2):
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.transformer = TransformerBlock(in_channels, heads=heads, mlp_ratio=mlp_ratio)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_proj(x)
        x_flat = x.flatten(2).transpose(1, 2)
        x_trans = self.transformer(x_flat)
        x = x_trans.transpose(1, 2).view(B, C, H, W)
        return x

# --- Custom BoTNet Backbone with CBAM ---
class BoTNetBackbone(nn.Module):
    def __init__(self, in_channels=128, cbam_reduction=16, heads=4, mlp_ratio=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.botnet = BottleneckTransformerLayer(in_channels=128, heads=heads, mlp_ratio=mlp_ratio)
        self.cbam = CBAM(128, reduction=cbam_reduction)
        self.out_channels = 128

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.botnet(x)
        x = self.cbam(x)
        return x

# --- Custom CIFAR10 Model with BoTNet Backbone ---
class CustomCIFAR10Model(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

# --- Custom Dataset Model with BoTNet Backbone ---
class CustomDataSetModel(nn.Module):
    def __init__(self, backbone, num_classes=7):
        super(CustomDataSetModel, self).__init__()
        self.backbone = backbone
        # Adapt the classifier to the output channels of BoTNetBackbone
        self.classifier = nn.Linear(backbone.out_channels, num_classes)

    def forward(self, x):
        # The backbone returns a single feature map
        features = self.backbone(x)
        features = torch.mean(features, dim=[2, 3])  # Global Average Pooling
        logits = self.classifier(features)
        return logits

>>>>>>> 893121523444da7561e1c06c8222f5f3645a7603
