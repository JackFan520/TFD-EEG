import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TFDEEG(nn.Module):
    def temporal_learner(
            self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )
    def __init__(self, num_classes, res_scale ,input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(TFDEEG, self).__init__()

        self.num_T = num_T
        self.out_graph = out_graph
        self.dropout_rate = dropout_rate
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.pool_step_rate = pool_step_rate
        self.idx = idx_graph
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        ###################
        self.model_dim = round(num_T / 2)
        self.num_heads = 8
        if sampling_rate == 200:
            self.window_size = 100
            self.stride = 20
        else:
            self.window_size = 64
            self.stride = 16
        ###################
        hidden_features = input_size[2]

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        #######################################
        self.feature_integrator = FeatureIntegrator(sr=sampling_rate, res_scale=res_scale , in_channels=32, out_channels=self.model_dim)
        self.sliding_window_processor = SlidingWindowProcessor(
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            stride=self.stride,
            dropout=0.0  # 根据需要设置
        )
        #######################################
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        # 表示局部滤波器的权重。它被定义为一个形状为(self.channel, size[-1])的浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        # 用来对local_filter_weight进行初始化，采用的是Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.local_filter_weight)
        # 表示局部滤波器的偏置。它被定义为一个形状为(1, self.channel, 1)的浮点型张量，初始值为全零，并设置为需要梯度计算
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        self.aggregate = Aggregator(self.idx)

        #树卷积
        # 自动划分半球 ——
        N = len(self.idx)
        half = N // 2
        left_regions = list(range(0, half))  # 前半子区
        right_regions = list(range(half, N))  # 后半子区
        self.tree_conv = GlobalTreeConvSimple(
            idx=self.idx,
            left_idxs=left_regions,
            right_idxs=right_regions,
            in_f=size[-1],
            out_f=out_graph
        )

        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # 改了： 14 子区 + 1 全局节点
        self.bn = nn.BatchNorm1d(len(self.idx) + 2)
        self.bn_ = nn.BatchNorm1d(len(self.idx) + 2)

        # 改了： 全连接，输入维度 = (14+1)*out_graph
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear((len(self.idx) + 2) * out_graph, num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        #######################################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out) # [B, 14, 2300]
        # 2) 树形聚合（方案二简单版）→ [B,15,out_graph]
        out = self.tree_conv(out)# [B,15,32]
        # 3) 标准化
        out = self.bn(out)
        out = self.bn_(out)
        # 4) 平坦化 + FC
        out = out.view(out.size(0), -1)  # [B, 15*out_graph]
        out = self.fc(out)  # [B, num_classes]
        return out

##############################
# —— 左右半球节点简单版 ——
##############################
class GraphConvolution_tree(nn.Module):
    """树形结构的简单GCN层（1层卷积+ReLU）。"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features)))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        # x: [B, N, in_features], adj: [N, N]
        h = torch.matmul(x, self.weight)  # [B, N, out_features]
        if self.bias is not None:
            h = h + self.bias
        h = torch.matmul(adj, h)         # 聚合邻居节点特征
        return F.relu(h)

class GlobalTreeConvSimple(nn.Module):
    def __init__(self, idx, left_idxs, right_idxs, in_f, out_f, device='cpu'):
        super().__init__()
        self.idx = idx
        self.N = len(idx)
        self.left_idxs = left_idxs
        self.right_idxs = right_idxs
        self.device = device

        # 半球融合：输入 2*in_f -> out_f
        self.gc_hemi_fuse = GraphConvolution_tree(2 * in_f, out_f)

        # 叶子投影（把原始 in_f→out_f）便于后续统一维度
        self.gc_leaf_proj = GraphConvolution_tree(in_f, out_f)

        # 全图最终层：在 (N+2) 个节点上做一次 GCN -> out_f
        self.gc_final = GraphConvolution_tree(out_f, out_f)

        # 邻接矩阵构建
        # adj_leaf_proj: identity for leaf projection (N x N)
        adj_leaf = torch.eye(self.N, dtype=torch.float32)
        self.register_buffer('adj_leaf', adj_leaf)

        # adj_hemi_fuse: single-node self-loop (1x1)
        self.register_buffer('adj_hemi', torch.eye(1, dtype=torch.float32))

        # adj_final: size (N+2)   indices: 0..N-1 leaves, N = hemi, N+1 = global
        M = self.N + 2
        adj_final = torch.zeros(M, M, dtype=torch.float32)
        # connect leaves and hemi to global (index = N+1)
        for i in list(range(self.N)) + [self.N]:
            adj_final[i, self.N + 1] = 1.0
            adj_final[self.N + 1, i] = 1.0
        # add self-loops
        adj_final += torch.eye(M, dtype=torch.float32)
        self.register_buffer('adj_final', adj_final)

    def forward(self, x):
        """
        x: [B, N, in_f]
        returns: out [B, N+2, out_f]  (leaves + hemi + global)
        """
        B, N, Fin = x.shape
        assert N == self.N, f"Expect N={self.N}, got {N}"
        # ---------- 1) 半球融合（在原始特征上做 hemi） ----------
        # left/right mean from original leaves
        left_feat = x[:, self.left_idxs, :].mean(dim=1, keepdim=True)   # [B,1,Fin]
        right_feat = x[:, self.right_idxs, :].mean(dim=1, keepdim=True) # [B,1,Fin]

        # hemi pair 拼接在特征维 -> [B,1,2*Fin]
        hemi_pair = torch.cat([left_feat, right_feat], dim=-1)

        # 单节点 GCN 融合 -> [B,1,out_f]
        hemi_fuse = self.gc_hemi_fuse(hemi_pair, self.adj_hemi)  # [B,1,out_f]

        # ---------- 2) 叶子投影到 out_f ----------
        h_leaves = self.gc_leaf_proj(x, self.adj_leaf)  # [B,N,out_f]

        # ---------- 3) 形成 N+1 节点并生成 global ----------
        in_Np1 = torch.cat([h_leaves, hemi_fuse], dim=1)   # [B, N+1, out_f]
        global_feat = in_Np1.mean(dim=1, keepdim=True)    # [B,1,out_f]

        # ---------- 4) 拼接为 N+2 并做 final GCN ----------
        in_final = torch.cat([in_Np1, global_feat], dim=1)  # [B, N+2, out_f]
        out = self.gc_final(in_final, self.adj_final)       # [B, N+2, out_f]
        return out

#######################################################

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))
#原版带通(简单版),自己的采样率
class BandPassSimpleMA(nn.Module):
    def __init__(self,res_scale, sr, bands=None ):
        super().__init__()
        self.sr = sr
        self.bands = bands if bands is not None else [(8.0,12.0),(12.0,30.0),(30.0,50.0)]
        self.res_scale = res_scale
        self.num_bands = len(self.bands)
        # 固定权重为1，可扩展为可学习
        self.register_buffer('band_w', torch.ones(self.num_bands))

    def _kernel_size(self, cutoff):
        k = max(3, int(round(self.sr / (cutoff + 1e-6))))
        return k + (k % 2 == 0)  # 保证为奇数

    def forward(self, x):
        # x: (B, C, T)
        bands_out = []
        for (f_low, f_high) in self.bands:
            k_high = self._kernel_size(f_high)
            k_low  = self._kernel_size(f_low)
            # 两次均值池化作为低通滤波
            lp_high = F.avg_pool1d(x, kernel_size=k_high, stride=1, padding=k_high//2)
            lp_low  = F.avg_pool1d(x, kernel_size=k_low,  stride=1, padding=k_low//2)
            # 带通信号 = 高截止低通 - 低截止低通
            band = lp_high - lp_low  # (B, C, T)
            bands_out.append(band)
        # 将各频段信号加权求和
        stacked = torch.stack(bands_out, dim=0)  # (num_bands, B, C, T)
        w = self.band_w.view(self.num_bands, 1, 1, 1)
        weighted = (stacked * w).sum(dim=0)     # (B, C, T)
        # 残差相加
        out = x + self.res_scale * weighted
        return out

class FeatureIntegrator(nn.Module):
    def __init__(self, sr, res_scale,in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bandPass =  BandPassSimpleMA(sr, res_scale)

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        x = self.bandPass(x)

        return x

#######滑动窗口技术###########################################################

#####因果多头注意力模块（大队长3.2CT_MSA）######
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True,
                 device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # 注册mask为buffer，自动跟随模型设备迁移
        self.register_buffer("mask", torch.tril(torch.ones(window_size, window_size)))

    def forward(self, x):
        # (b*n, t, c)
        B_prev, T_prev, C_prev = x.shape

        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)

        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            # 这里使用buffer中的mask，无需额外调用to(x.device)
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x

# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        # 设置1层temporal attention即可
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: (b, c, n, t)
        b, c, n, t = x.shape
        # 对输入x进行reshape,转换成便于计算的shape: (b, c, n, t) --> (b,n,t,c) --> (b*n,t,c)
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        # 为输入x添加位置编码: (b*n,t,c) + (1,t,c) = (b*n,t,c)
        x = x + self.pos_embedding

        # 执行注意力机制 和 前馈神经网络层
        for attn, ff in self.layers:
            x = attn(x) + x   # 执行注意力机制并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
            x = ff(x) + x   # 执行前馈神经网络并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2) #  (b*n,t,c)--reshape-->(b,n,t,c)--permute-->(b,c,n,t)
        return x

###简化版本频域提取########################
class FrequencyBranchSimp(nn.Module):
    def __init__(self, n_fft=64, hop_length=32, win_length=64):
        super(FrequencyBranchSimp, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # 提前创建 Hann 窗口，并注册为 buffer，避免每次计算时重复创建
        self.register_buffer("hann_window", torch.hann_window(win_length))

    def forward(self, x):
        B, C, T = x.shape  # C == model_dim
        # 合并批次和通道，使用 reshape 替换 view
        x_reshaped = x.reshape(B * C, T)
        # 计算 STFT（批量计算）
        stft_result = torch.stft(
            x_reshaped,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            return_complex=True
        )  # (B*C, F, T_f)
        mag = stft_result.abs()  # (B*C, F, T_f)
        # 对频率维度求平均，得到简单的频域特征
        mag_mean = mag.mean(dim=1)  # (B*C, T_f)
        # 恢复为 (B, C, T_f)
        mag_mean = mag_mean.reshape(B, C, -1)
        # 如果时间步 T_f 与原始窗口 T 不一致，则上采样回 T
        if mag_mean.size(-1) != T:
            mag_mean = F.interpolate(mag_mean, size=T, mode='nearest')
        return mag_mean  # (B, model_dim, window_size)
###########################################

####频谱细节提取##########################

class FrequencyDetailBranch(nn.Module):

    def __init__(self,channels, n_fft=64, hop_length=32, win_length=64):
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Hann 窗口
        self.register_buffer('hann', torch.hann_window(win_length))
        # 用于高频滤波（Laplacian 核）
        lap = torch.tensor([[0., -1., 0.],
                            [-1., 4., -1.],
                            [0., -1., 0.]]).view(1,1,3,3)
        self.register_buffer('laplacian', lap)

        # 2D CNN：3C → C
        # 第一层降维并提取局部特征
        self.conv1 = nn.Conv2d(in_channels=4*channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        # 第二层空洞卷积扩大感受野
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.GELU()

    def forward(self, x):
        B, C, T = x.shape  # C == model_dim
        # 1) STFT 并取幅值谱
        x_flat = x.reshape(B*C, T)
        spec = torch.stft(x_flat,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.hann,
                          return_complex=True)        # (B*C, F, T_f)
        mag = spec.abs()                   # (B*C, F, T_f)
        F_bin, T_f = mag.shape[-2], mag.shape[-1]
        mag = mag.view(B, C, F_bin, T_f)   # (B, C, F, T_f)

        # 2) 频率尖峰通道：频率轴差分
        # pad 前后各一行，使 shape 不变
        pad_freq = F.pad(mag, (0,0,1,1), mode='replicate')
        diff_freq = pad_freq[:,:,2:,:] - pad_freq[:,:,:-2,:]  # (B,C,F,T_f)

        # 3) 快变通道：时间轴差分
        pad_time = F.pad(mag, (1,1,0,0), mode='replicate')
        diff_time = pad_time[:,:,:,2:] - pad_time[:,:,:,:-2]  # (B,C,F,T_f)

        # 4) 高频成分通道：Laplacian 高通滤波
        # 按通道卷积
        # laplacian: (1,1,3,3) -> (C,1,3,3)
        lap = self.laplacian.repeat(C,1,1,1)
        high_freq = F.conv2d(mag, weight=lap, padding=1, groups=C)  # (B,C,F,T_f)

        # 5) 当作“RGB”三通道，拼在一起： (B,3C,F,T_f)
        rgb = torch.cat([mag, diff_freq, diff_time, high_freq], dim=1)

        # 6) CNN 提取特征
        y = self.conv1(rgb)    # (B, C, F, T_f)
        y = self.bn1(y)
        y = self.act(y)

        y = self.conv2(y)      # (B, C, F, T_f)
        y = self.bn2(y)
        y = self.act(y)

        # 7) 沿频率维度池化 → (B, C, T_f)
        y = y.mean(dim=2)

        # 8) 插值对齐到原始窗口长度 T
        if T_f != T:
            y = F.interpolate(y, size=T, mode='linear', align_corners=False)

        return y  # (B, C, T)

######################################
###########################################
class SlidingWindowProcessor(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride, dropout=0.):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        # 每个窗口归一化，调整为 (B, window_size, model_dim)
        self.layer_norm = nn.LayerNorm([window_size, model_dim])

        #### 支路 A：标准多头注意力 + TCN_block
        self.standard_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm_std = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=32)  # 输出通道 32

        #### 支路 B：串联简化版频域特征提取 -> 因果多头注意力模块
        self.freq_branch = FrequencyBranchSimp(n_fft=64, hop_length=32, win_length=64)
        self.ct_msa = CT_MSA(
            dim=model_dim,
            depth=1,
            heads=num_heads,
            window_size=window_size,
            mlp_dim=2 * model_dim,
            num_time=window_size,
            dropout=dropout
        )
        self.highpass    = FrequencyDetailBranch(channels=model_dim)

        self.layer_norm_ct = nn.LayerNorm(model_dim)

        #### 融合层：融合支路 A 与支路 B
        # 支路 A 输出 (B, 32, window_size)，支路 B 输出 (B, model_dim, window_size)
        self.fuse_linear = nn.Linear(32 + model_dim, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        # 最后融合所有窗口输出的卷积层（保持原有设计），输出通道 32
        self.fusion_conv = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入 x: (B, model_dim, L)
        输出: 融合后的特征，形状 (B, 32, fused_length)
        """
        batch_size, model_dim, length = x.shape
        window_outputs = []

        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            # 分割窗口，形状 (B, model_dim, window_size)
            window = x[:, :, window_start:window_end]
            # 调整为 (B, window_size, model_dim) 便于归一化
            window_norm = window.permute(0, 2, 1)
            window_norm = self.layer_norm(window_norm)

            #### 支路 A：标准多头注意力 + TCN_block
            std_attn_out, _ = self.standard_attn(window_norm, window_norm, window_norm)
            std_attn_out = self.layer_norm_std(std_attn_out + window_norm)  # (B, window_size, model_dim)
            branchA = self.tcn_block(std_attn_out.permute(0, 2, 1))  # (B, 32, window_size)

            #### 支路 B：串联频域特征提取 -> 因果多头注意力模块
            # 统一输入形状 (B, model_dim, window_size)
            window_for_B = window  # (B, model_dim, window_size)
            branchB_inter_f = self.freq_branch(window_for_B)  # (B, model_dim, window_size)
            branchB_inter_h = self.highpass(branchB_inter_f)
            branchB_inter_f = self.layer_norm_ct(branchB_inter_f.permute(0, 2, 1)).permute(0, 2, 1)
            branchB_inter_h = self.layer_norm_ct(branchB_inter_h.permute(0, 2, 1)).permute(0, 2, 1)
            branchB_inter = branchB_inter_h * 0.03 + branchB_inter_f
            branchB_inter = self.layer_norm_ct(branchB_inter.permute(0, 2, 1)).permute(0, 2, 1)
            # branchB_inter = F.gelu(branchB_inter)
            # 调整形状供 CT_MSA 使用：CT_MSA 需要 (B, model_dim, 1, window_size)
            branchB_inter = branchB_inter.unsqueeze(2)  # (B, model_dim, 1, window_size)
            ct_out = self.ct_msa(branchB_inter)  # (B, model_dim, 1, window_size)
            ct_out = ct_out.squeeze(2).permute(0, 2, 1)  # (B, window_size, model_dim)
            ct_out = self.layer_norm_ct(ct_out + window_norm)  # 残差连接
            branchB_out = ct_out.permute(0, 2, 1)  # (B, model_dim, window_size)

            #### 融合支路 A 与支路 B
            fused = torch.cat([branchA, branchB_out], dim=1)  # (B, 32 + model_dim, window_size)
            fused = fused.permute(0, 2, 1)  # (B, window_size, 32 + model_dim)
            fused = self.fuse_linear(fused)  # (B, window_size, model_dim)
            window_outputs.append(fused)

        # 堆叠所有窗口，得到 (B, window_size, num_windows, model_dim)
        stacked_outputs = torch.stack(window_outputs, dim=2) #(1,100,23,32)
        # 调整形状以适应最终融合卷积层
        # 假设支路A（TCN_block）输出通道为 32，reshape 为 (B, 32, -1)
        stacked_outputs = stacked_outputs.permute(0, 1, 3, 2).reshape(batch_size, 32, -1) #(1,32,2300)
        fused_output = self.fusion_conv(stacked_outputs)
        return fused_output

########################################################################################

class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))

class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values

        return torch.mean(x, dim=dim)
