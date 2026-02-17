import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import math
import torch.fft as fft


class SFMLP(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super(SFMLP,self).__init__()

        self.dwconv1 = SeparableConv(in_features,2*in_features,8)
        self.dwconv2 = SeparableConv(2*in_features,in_features,8)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        self.aa1 = SeparableConv2(in_features, in_features, kernel_size=(3,5),padding = ((3 - 1) // 2, (5 - 1) // 2))
        self.aa2 = SeparableConv2(in_features, in_features, kernel_size=(5,3),padding = ((5 - 1) // 2, (3 - 1) // 2))
        self.aa3 = SeparableConv2(in_features, in_features, kernel_size=(3,7),padding = ((3 - 1) // 2, (7 - 1) // 2))
        self.aa4 = SeparableConv2(in_features, in_features, kernel_size=(7,3),padding = ((7 - 1) // 2, (3 - 1) // 2))
        # self.aa5 = ConvBNReLU(2*in_features,in_features,1)
        # self.softmax = nn.Softmax(dim=1)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()
    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x
    def forward(self, x, H, W):
 
        x = blc2bchw(x,H,W)
        x = self.pad_out(x)
        x = self.dwconv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.pad_out(x)
        x = self.dwconv2(x)
        x = self.act(x)
        x = self.drop(x)
        f = fft.fftn(x)

        f1 = self.aa1(f.real)
        f11 = self.aa2(f1)

        f1 = f11 + f.imag * 1j
        f1 = fft.ifftn(f1)
        
        f2 = self.aa3(f.real)
        f22 = self.aa4(f2)
        f2 = f22 + f.imag * 1j
        f2 = fft.ifftn(f2)
      
        x1 = f1 + f2
        x1 =torch.abs(x1)

        x = x1 + x

        return x


def blc2bchw(x,h,w):
    b,l,c = x.shape
    assert l==h*w, "in blc to bchw, h*w != l."
    return x.view(b,h,w,c).permute(0,3,1,2).contiguous()


def bchw2blc(x,h,w):
    b,c,_,_ = x.shape
    return x.permute(0,2,3,1).view(b,-1,c).contiguous()


def window_partition(x, window_size):
    """
        x: (B, H, W, C) Returns:windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C).contiguous()
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C).contiguous()
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)   Returns:x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1).contiguous()
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).contiguous()
    return x
def create_learnable_tensor(x, B, H, W, eps=1e-8):
    initial_value = torch.zeros((B, H, W, 1),device=x.device)
    learnable_tensor = nn.Parameter(initial_value + eps)
    return learnable_tensor
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class Conv11(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,padding=0):
        super(Conv11, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, stride=stride,padding = padding)
        )
        
        
class ConvReLu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1):
        super(ConvReLu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            nn.ReLU6()
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

class Separable(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(Separable, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
class SeparableConv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,padding=0):
        super(SeparableConv2, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=padding,
                      groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        
class SeparableConv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv1, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels//3),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

class WindowMSA(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, 
                attn_drop=0., proj_drop=0.,use_relative_pe=False):

        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
           
        self.use_relative_pe = use_relative_pe
        if self.use_relative_pe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: [num_windows*B, N, C]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.use_relative_pe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class ASFFAGate(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ASFFAGate, self).__init__()
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, dim, 1, bias=False),
            nn.Sigmoid()
        )
        self.channel_adjust = None
    def forward(self, spatial_feat, freq_feat):
        B, C, H, W = spatial_feat.shape
        B_f, C_f, H_f, W_f = freq_feat.shape
        if C_f != C:
            if C_f == 1:
                freq_feat = freq_feat.expand(B, C, H, W)
            else:
                if not hasattr(self, 'channel_adjust') or self.channel_adjust is None:
                    self.channel_adjust = nn.Conv2d(C_f, C, kernel_size=1, bias=False).to(spatial_feat.device)
                    nn.init.kaiming_normal_(self.channel_adjust.weight, mode='fan_out', nonlinearity='relu')
                freq_feat = self.channel_adjust(freq_feat)
        spatial_weight = self.spatial_gate(spatial_feat)
        freq_weight = self.freq_gate(freq_feat)
        total_weight = spatial_weight + freq_weight + 1e-8
        spatial_weight_norm = spatial_weight / total_weight
        freq_weight_norm = freq_weight / total_weight
        weighted_spatial = spatial_weight_norm * spatial_feat
        weighted_freq = freq_weight_norm * freq_feat
        fused_feat = weighted_spatial + weighted_freq
        return fused_feat, weighted_spatial, weighted_freq

class ASFFACrossAttention(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ASFFACrossAttention, self).__init__()
        self.cross_attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim//reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction, dim, 1),
            nn.Sigmoid()
        )
    def forward(self, fused_feat, weighted_spatial, weighted_freq):
        concat_feat = torch.cat([weighted_spatial, weighted_freq], dim=1)
        cross_attn = self.cross_attention(concat_feat)
        enhanced_feat = fused_feat * cross_attn + fused_feat
        return enhanced_feat

class ASFFARefine(nn.Module):
    def __init__(self, dim):
        super(ASFFARefine, self).__init__()
        self.feature_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True)
        )
    def forward(self, enhanced_feat):
        return self.feature_refine(enhanced_feat)

class ASFFA(nn.Module):
    def __init__(self, dim, reduction=4):
        super(ASFFA, self).__init__()
        self.gate = ASFFAGate(dim, reduction)
        self.cross = ASFFACrossAttention(dim, reduction)
        self.refine = ASFFARefine(dim)
    def forward(self, spatial_feat, freq_feat):
        fused_feat, weighted_spatial, weighted_freq = self.gate(spatial_feat, freq_feat)
        enhanced_feat = self.cross(fused_feat, weighted_spatial, weighted_freq)
        refined_feat = self.refine(enhanced_feat)
        return refined_feat

class MSFPF(nn.Module):
    """
    Multi-Scale Feature Pyramid Fusion Module
    """
    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super(MSFPF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aspp_convs = nn.ModuleList()
        for rate in rates:
            if rate == 1:
                conv = nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
            else:
                conv = nn.Conv2d(in_channels, out_channels//4, 3, padding=rate, dilation=rate, bias=False)
            self.aspp_convs.append(nn.Sequential(
                conv,
                nn.BatchNorm2d(out_channels//4),
                nn.ReLU(inplace=True)
            ))
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels//4, 1, bias=False),
            nn.BatchNorm2d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//8, 1, bias=False),
            nn.BatchNorm2d(out_channels//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//8, out_channels//8, 3, padding=1, groups=out_channels//8, bias=False),
            nn.Conv2d(out_channels//8, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(out_channels + out_channels//4, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(out_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels//16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//16, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        b, c, h, w = x.shape
        aspp_features = []
        for aspp_conv in self.aspp_convs:
            aspp_features.append(aspp_conv(x))
        global_feat = self.global_avg_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode='bilinear', align_corners=False)
        edge_feat = self.edge_conv(x)
        aspp_concat = torch.cat(aspp_features + [global_feat], dim=1)
        fused_feat = self.conv_concat(aspp_concat)
        fused_feat = fused_feat + edge_feat
        ca_weight = self.channel_attention(fused_feat)
        fused_feat = fused_feat * ca_weight
        sa_weight = self.spatial_attention(fused_feat)
        fused_feat = fused_feat * sa_weight
        if self.in_channels == self.out_channels:
            fused_feat = fused_feat + x
        return fused_feat

class SFFormer(nn.Module):
    def __init__(self, dim, heads):
        super(SFFormer, self).__init__()
        self.local_attn = WindowMSA(dim+1, (8,8), heads, use_relative_pe=True)
        self.cpool = ZPool()
        self.ww = Conv(2,1)
        self.mlp = SFMLP(in_features=dim, act_layer=nn.GELU, drop=0.1)
        self.norm = nn.LayerNorm(dim)
        self.qkv = Conv(2, 3*2, kernel_size=1)
        self.by = Conv(1, 1, kernel_size=3)
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.GELU(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, 1),
            nn.GELU(),
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
        )
        self.sigmoid = nn.Sigmoid()
        self.apply(self._init_weights)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.asffa = ASFFA(dim, reduction=4)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def get_index(self,real_h):
        index = []
        windows = real_h // 4
        for i in range(windows):
            if i==0:
                index.append(4-1)
            elif i==windows-1:
                index.append(real_h-4)
            else:
                index.append(i*4)
                index.append(i*4+3)
        return index
  
    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        L = self.cpool(x)
        L = self.ww(L)
        f = fft.fftn(L)
        f = self.conv(f.real) + self.conv(f.imag) * 1j
        fshift = fft.fftshift(f)
        rows, cols = a, b
        crow, ccol = rows // 2, cols // 2
        mask = torch.ones((rows, cols), dtype=torch.float32, device=L.device)
        mask[..., crow-crow//16:crow+crow//16, ccol-ccol//16:ccol+ccol//16] = 0
        fshift_filtered = fshift * mask
        f_ishift = fft.ifftshift(fshift_filtered)
        img_back = fft.ifftn(f_ishift)
        freq_feat = torch.abs(img_back)
        freq_feat = self.by(freq_feat)
        B, _, H, W = freq_feat.shape
        c = x.shape[1]
        freq_feat = freq_feat.expand(B, c, H, W)
        x = x.permute(0,2,3,1).contiguous()
        b, h, w, c = x.shape
        x1 = create_learnable_tensor(x, b, h, w)
        x = torch.cat((x, x1), dim=-1)
        local_in = window_partition(x, [8,8]).view(-1, 64, x.shape[3]).contiguous()
        local_windows = self.local_attn(local_in)
        local_out = window_reverse(local_windows, [8,8], x.shape[1], x.shape[2]).contiguous()
        L1 = local_out[:,:,:,-1:]
        spatial_feat = local_out[:,:,:,:-1]
        spatial_feat_bchw = spatial_feat.permute(0,3,1,2).contiguous()
        fused_feat = self.asffa(spatial_feat_bchw, freq_feat)
        fused_feat_bhwc = fused_feat.permute(0,2,3,1).contiguous()
        bb, hh, ww, cc = fused_feat_bhwc.shape
        local_windows = fused_feat_bhwc.view(bb, hh*ww, cc).contiguous()
        local_windows = self.mlp(self.norm(local_windows), hh, ww)
        local_windows = local_windows.view(bb, hh, ww, cc).contiguous() + fused_feat_bhwc
        local_windows = local_windows.permute(0,3,1,2).contiguous()
        return local_windows

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.1, 
                 layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
    
        self.sps = nn.ModuleList([
                SFFormer(dims[0],1),
                SFFormer(dims[1],1),
                SFFormer(dims[2],1),
                SFFormer(dims[3],1),
            ])
        
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        stages_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.sps[i](x) + x
            stages_out.append(x)
        return stages_out

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[64, 128, 256, 512],mode="tiny", **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768],**kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], mode = "base",**kwargs,)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"],strict=False)
    return model


class LRDU(nn.Module):
    def __init__(self,in_c,factor):
        super(LRDU,self).__init__()
        self.up_factor = factor
        self.factor1 = factor*factor//2
        self.factor2 = factor*factor
        self.up = nn.Sequential(
            nn.Conv2d(in_c, self.factor1*in_c, (1,7), padding=(0, 3), groups=in_c),
            nn.Conv2d(self.factor1*in_c, self.factor2*in_c, (7,1), padding=(3, 0), groups=in_c),
            nn.PixelShuffle(factor),
            nn.Conv2d(in_c, in_c, 3 ,groups= in_c//4,padding=1), 
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()

        self.up = nn.Sequential(
            LRDU(ch_in,2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    def __init__(self, n_class=6, pretrained=True):
        super(Model, self).__init__()
        self.n_class = n_class
        self.in_channel = 3
        config = [96, 192, 384, 768]  # channles of convnext-small
        self.backbone = convnext_small(pretrained, True)

        self.Up5 = up_conv(ch_in=config[3], ch_out=config[3] // 2)
        self.Up_conv5 = conv_block(ch_in=config[3], ch_out=config[3] // 2)

        self.Up4 = up_conv(ch_in=config[2], ch_out=config[2] // 2)
        self.Up_conv4 = conv_block(ch_in=config[2], ch_out=config[2] // 2)

        self.Up3 = up_conv(ch_in=config[1], ch_out=config[1] // 2)
        self.Up_conv3 = conv_block(ch_in=config[1], ch_out=config[1] // 2)
        self.Up4x = LRDU(config[0], 4)
        self.msfpf_final = MSFPF(config[0], config[0])
        self.convout = nn.Conv2d(config[0], n_class, kernel_size=1, stride=1, padding=0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x128, x64, x32, x16 = self.backbone(x)
        d32 = self.Up5(x16)
        d32 = torch.cat([x32, d32], dim=1)
        d32 = self.Up_conv5(d32)
        d64 = self.Up4(d32)
        d64 = torch.cat([x64, d64], dim=1)
        d64 = self.Up_conv4(d64)
        d128 = self.Up3(d64)
        d128 = torch.cat([x128, d128], dim=1)
        d128 = self.Up_conv3(d128)
        d512 = self.Up4x(d128)
        d512 = self.msfpf_final(d512)
        out = self.convout(d512)
        return out