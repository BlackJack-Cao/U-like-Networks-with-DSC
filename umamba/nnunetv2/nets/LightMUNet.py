from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from nnunetv2.nets.TTT import TTTLinear, TTTConfig
from nnunetv2.nets.DMSK_3d import DMSKModule
from torch.cuda.amp import autocast

from mamba_ssm import Mamba

class TTTLayer(nn.Module):
    def __init__(self, dim, d_state = 16):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.TTTConfig = TTTConfig(hidden_size=dim, num_attention_heads=d_state, num_hidden_layers=d_state)
        self.TTTLinear = TTTLinear(
                #  hidden_size=dim, 
                #  num_attention_heads=12, 
                #  num_hidden_layers=d_state, 
                #  mini_batch_size=16, 
                #  share_qk=False, 
                #  conv_kernel=4, 
                #  ttt_base_lr=1,
                #  rope_theta=10000.0, 
                #  use_gate=False,
                #  scan_checkpoint_group_size=0,
                self.TTTConfig
        )
    
    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        # print(f"x_norm size: {x_norm.size()}")
        position_ids = torch.arange(x_norm.size(1), device=x_norm.device).unsqueeze(0).expand(x_norm.size(0), -1)
        x_TTT = self.TTTLinear(x_norm, position_ids=position_ids)
        out = x_TTT.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

def get_dwconv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels, 
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, 
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)

class RVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


def get_rvm_layer(
    spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1
):
    mamba_layer = RVMLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims==2:
            return nn.Sequential(mamba_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
        if spatial_dims==3:
            return nn.Sequential(mamba_layer, nn.MaxPool3d(kernel_size=stride, stride=stride))
    return mamba_layer


class ResMambaBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_rvm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )
        self.conv2 = get_rvm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class ResUpBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv = get_dwconv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.skip_scale= nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv(x) + self.skip_scale * identity
        x = self.norm2(x)
        x = self.act(x)
        return x


class LightMUNet(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: list = [1, 2, 2, 4],
        blocks_up: list = [1, 1, 1],
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")

        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.down_layers = self._make_down_layers()
        self.up_layers, self.up_samples = self._make_up_layers()
        self.conv_final = self._make_final_conv(out_channels)

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

        # 修改为使用blocks_up的长度和对应的通道数
        self.dmsk_modules = nn.ModuleList()
        self.ttt_layers = nn.ModuleList()
        
        n_up = len(blocks_up)
        for i in range(n_up):
            layer_channels = init_filters * (2 ** (n_up - i))
            # 使用每个上采样层对应的通道数
            skip_channels = layer_channels // 2  # 跳跃连接的通道数
            self.dmsk_modules.append(DMSKModule(dim=skip_channels))
            self.ttt_layers.append(TTTLayer(dim=skip_channels))

    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2**i
            downsample_mamba = (
                get_rvm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba, *[ResMambaBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers

    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.down_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            skip_connection = down_x[i + 1]
            processed_skip = self.dmsk_modules[i](skip_connection)  # 先经过DMSK
            processed_skip = self.ttt_layers[i](processed_skip)     # 再经过TTT
            
            # 使用处理后的跳跃连接
            x = up(x) + processed_skip
            x = upl(x)
            # x = up(x) + down_x[i + 1]
            # x = upl(x)

        if self.use_conv_final:
            x = self.conv_final(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, down_x = self.encode(x)
        down_x.reverse()

        x = self.decode(x, down_x)
        return x