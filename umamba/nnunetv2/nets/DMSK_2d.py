import torch
import torch.nn as nn
class DynamicKernelSelection(nn.Module):
    def __init__(self, in_channel, kernel_sizes_1=[3, 5], kernel_sizes_2=[7, 9, 11]):
        super().__init__()
        self.in_channel = in_channel
        self.kernel_sizes_1 = kernel_sizes_1  # att_conv1 的卷积核尺寸
        self.kernel_sizes_2 = kernel_sizes_2  # att_conv2 的卷积核尺寸
        
        # 定义 att_conv1 的卷积核（较小的卷积核，提取局部特征）
        self.conv_layers_1 = nn.ModuleList([
            nn.Conv2d(in_channel, in_channel, kernel_size=k, padding=k//2, groups=in_channel)
            for k in kernel_sizes_1
        ])
        
        # 定义 att_conv2 的卷积核（较大的卷积核，提取上下文特征）
        self.conv_layers_2 = nn.ModuleList([
            nn.Conv2d(in_channel, in_channel, kernel_size=k, padding=k//2 + (k//2) * 2, dilation=3, groups=in_channel)
            for k in kernel_sizes_2
        ])
        
        # att_conv1 的注意力机制
        self.attention_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channel, len(kernel_sizes_1), kernel_size=1),  # 生成卷积核权重
            nn.Softmax(dim=1)  # 归一化为权重
        )
        
        # att_conv2 的注意力机制
        self.attention_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channel, len(kernel_sizes_2), kernel_size=1),  # 生成卷积核权重
            nn.Softmax(dim=1)  # 归一化为权重
        )

    def forward(self, x):
        # 计算 att_conv1 的卷积核权重
        weights_1 = self.attention_1(x)  # [B, num_kernels_1, 1, 1, 1]
        selected_kernel_index_1 = torch.argmax(weights_1, dim=1)  # [B, 1, 1, 1]
        
        # 计算 att_conv2 的卷积核权重
        weights_2 = self.attention_2(x)  # [B, num_kernels_2, 1, 1, 1]
        selected_kernel_index_2 = torch.argmax(weights_2, dim=1)  # [B, 1, 1, 1]
        
        # 使用选择的卷积核提取特征（顺序处理：大核使用小核输出，模仿DLK设计）
        # 使用列表避免 inplace 赋值问题
        output_1_list = []
        output_2_list = []
        for i in range(x.size(0)):  # 遍历 batch
            # att_conv1 的特征提取（小核）
            selected_kernel_1 = self.conv_layers_1[selected_kernel_index_1[i]].to(x.device)
            out1 = selected_kernel_1(x[i].unsqueeze(0)).squeeze(0)
            output_1_list.append(out1)
            
            # att_conv2 的特征提取（大核使用小核输出，顺序处理，模仿DLK）
            selected_kernel_2 = self.conv_layers_2[selected_kernel_index_2[i]].to(x.device)
            out2 = selected_kernel_2(out1.unsqueeze(0)).squeeze(0)
            output_2_list.append(out2)
        
        output_1 = torch.stack(output_1_list, dim=0)
        output_2 = torch.stack(output_2_list, dim=0)
        return output_1, output_2

class DMSK(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        # 前置 1x1 降维到 C/2，和 DLK 一致
        self.channel_proj = nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, bias=False)
        self.dynamic_kernel_selection = DynamicKernelSelection(in_channel // 2)

        self.spatial_se = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先用 1x1 将通道降到 C/2（与 DLK 一致）
        x_proj = self.channel_proj(x)
        att1, att2 = self.dynamic_kernel_selection(x_proj)

        # 模仿DLK：先拼接两个分支的输出
        out = torch.cat([att1, att2], dim=1)
        
        # 计算空间注意力权重
        avg_att = torch.mean(out, dim=1, keepdim=True)
        max_att, _ = torch.max(out, dim=1, keepdim=True)
        att = torch.cat([avg_att, max_att], dim=1)
        att = self.spatial_se(att)
        
        # 完全模仿DLK的实现方式：对拼接后的out分别加权后相加
        out = out * att[:, 0, :, :].unsqueeze(1) + out * att[:, 1, :, :].unsqueeze(1)
        
        output = out + x
        return output

class DMSKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.spatial_gating_unit = DMSK(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = x.to(self.proj_1.weight.device)
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.act(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x