import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ==================== 共享组件 ====================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    时间步嵌入

    类用于将时间步（如扩散模型中的 t）编码为高维向量，便于神经网络处理。
    其核心思想是用正弦和余弦函数对时间步进行位置编码，类似于 Transformer 的位置编码。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """
    简化的UNet，用于噪声预测

    该UNet包含编码器、解码器和跳跃连接，适用于图像数据的噪声预测任务。
    通过时间步嵌入，模型可以根据不同的时间步调整其预测。
    结构包括：
    - 时间嵌入层：将时间步编码为高维向量。
    - 编码器：多个卷积层和下采样层，提取图像特征。
    - 中间层：进一步处理编码器输出的特征。
    - 解码器：多个卷积层和上采样层，重建图像。
    - 输出层：生成与输入图像相同尺寸的噪声预测。

    下采样：
    - 使用卷积层实现，减小空间尺寸。
    上采样：
    - 使用转置卷积层实现，增大空间尺寸。
        - 步骤1：输入特征图扩展
        - 步骤2：卷积核滑动计算
        - 步骤3：重叠部分相加，生成更大尺寸的输出特征图
    - 其他上采样方法还包括：
        - 最近邻插值: 直接复制最近的像素值
        - 双线性插值: 在水平和垂直方向分别进行线性插值
        - 双三次插值: 使用立方卷积核进行插值，效果更平滑
        - 反池化: 通过记录池化索引位置进行上采样
    """

    def __init__(self, image_channels=1, hidden_dims=[32, 64, 128], time_emb_dim=32):
        super().__init__()

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # 编码器
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        in_channels = image_channels
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
            ))
            self.downsample.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1))
            in_channels = hidden_dim

        # 中间层
        self.mid_block = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 3, padding=1),
            nn.ReLU(),
        )

        # 解码器
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(len(hidden_dims) - 1, 0, -1):
            self.upsample.append(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i], 4, stride=2, padding=1))
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i] + hidden_dims[i - 1], hidden_dims[i - 1], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dims[i - 1], hidden_dims[i - 1], 3, padding=1),
                nn.ReLU(),
            ))

        # 输出层
        self.final_upsample = nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0], 4, stride=2, padding=1)
        self.final_conv = nn.Conv2d(hidden_dims[0] + image_channels, image_channels, 3, padding=1)

    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 编码路径
        encoder_features = []
        for block, downsample in zip(self.encoder_blocks, self.downsample):
            x = block(x)
            encoder_features.append(x)
            x = downsample(x)

        # 中间层
        x = self.mid_block(x)

        # 解码路径
        for i, (upsample, block) in enumerate(zip(self.upsample, self.decoder_blocks)):
            x = upsample(x)
            x = torch.cat([x, encoder_features[-(i + 1)]], dim=1)
            x = block(x)

        # 最终输出
        x = self.final_upsample(x)
        x = torch.cat([x, encoder_features[0]], dim=1)
        x = self.final_conv(x)

        return x


class DiffusionModel:
    """扩散模型基类"""

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, image_size=28, device="cuda"):
        self.timesteps = timesteps
        self.image_size = image_size
        self.device = device

        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # 初始化模型
        self.model = SimpleUNet(image_channels=1).to(device)

    def q_sample(self, x_start, t, noise=None):
        """前向扩散过程：q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_step(self, x_start, optimizer):
        """训练步骤"""
        optimizer.zero_grad()

        batch_size = x_start.shape[0]

        # 随机采样时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()

        # 前向加噪
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        # 预测噪声
        noise_pred = self.model(x_noisy, t)

        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()

        return loss.item()


# ==================== DDPM 实现 ====================

class DDPM(DiffusionModel):
    """Denoising Diffusion Probabilistic Models"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # DDPM 特有的参数计算
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 后验方差计算
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def p_sample(self, x, t, t_index):
        """DDPM 反向过程单步采样: p(x_{t-1} | x_t)"""

        # 关键区别点 1: DDPM 使用固定的后验方差
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1, 1, 1)

        # 预测 x0 和噪声
        pred_noise = self.model(x, t)
        pred_x0 = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        # 关键区别点 2: DDPM 在每一步都添加随机噪声
        if t_index > 0:
            # 计算均值和方差
            model_mean = (
                    (self.alphas_cumprod_prev[t].sqrt() * self.betas[t] / (1. - self.alphas_cumprod[t])) * pred_x0 +
                    (self.alphas[t].sqrt() * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])) * x
            )

            # 关键区别点 3: 随机采样噪声
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t]) * noise
            x_prev = model_mean + variance
        else:
            x_prev = pred_x0

        return x_prev

    def sample(self, num_samples=16):
        """DDPM 完整采样过程"""
        self.model.eval()

        with torch.no_grad():
            # 从纯噪声开始
            img = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)

            imgs = []

            # 关键区别点 4: DDPM 必须从 T 到 0 完整遍历所有时间步
            for i in range(self.timesteps - 1, -1, -1):
                t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
                img = self.p_sample(img, t, i)

                if i % (self.timesteps // 10) == 0:
                    imgs.append(img.cpu())

        self.model.train()
        return imgs


# ==================== DDIM 实现 ====================

class DDIM(DiffusionModel):
    """Denoising Diffusion Implicit Models"""

    def __init__(self, *args, sampling_timesteps=50, eta=0.0, **kwargs):
        super().__init__(*args, **kwargs)

        # 关键区别点 1: DDIM 可以自定义采样步数
        self.sampling_timesteps = sampling_timesteps
        self.eta = eta  # η=0 表示确定性生成，η=1 表示类似DDPM的随机性

        # 创建采样时间序列
        step_ratio = self.timesteps // self.sampling_timesteps
        self.sampling_timesteps_seq = np.asarray(list(range(0, self.timesteps, step_ratio)))
        self.sampling_timesteps_seq = list(reversed(self.sampling_timesteps_seq))

    def p_sample(self, x, t, t_prev):
        """DDIM 反向过程单步采样"""

        # 预测噪声
        pred_noise = self.model(x, t)

        # 关键区别点 2: DDIM 使用不同的 x0 预测公式
        sqrt_recip_alphas_cumprod_t = torch.sqrt(1.0 / self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        pred_x0 = sqrt_recip_alphas_cumprod_t * (x - sqrt_one_minus_alphas_cumprod_t * pred_noise)
        pred_x0 = torch.clamp(pred_x0, -1., 1.)

        # 关键区别点 3: 计算方向
        dir_xt = torch.sqrt(1. - self.alphas_cumprod[t_prev] - self.eta ** 2 * self.posterior_variance[t]) * pred_noise

        # 关键区别点 4: DDIM 的确定性/随机性控制
        noise = torch.randn_like(x) if self.eta > 0 else 0

        x_prev = (
                torch.sqrt(self.alphas_cumprod[t_prev]) * pred_x0 +
                dir_xt +
                self.eta * torch.sqrt(self.posterior_variance[t]) * noise
        )

        return x_prev

    def sample(self, num_samples=16):
        """DDIM 完整采样过程"""
        self.model.eval()

        with torch.no_grad():
            # 从纯噪声开始
            img = torch.randn((num_samples, 1, self.image_size, self.image_size), device=self.device)

            imgs = []

            # 关键区别点 5: DDIM 可以跳步采样
            for i in range(len(self.sampling_timesteps_seq) - 1):
                t = torch.full((num_samples,), self.sampling_timesteps_seq[i], device=self.device, dtype=torch.long)
                t_prev = torch.full((num_samples,), self.sampling_timesteps_seq[i + 1], device=self.device, dtype=torch.long)

                img = self.p_sample(img, t, t_prev)

                if i % (len(self.sampling_timesteps_seq) // 5) == 0:
                    imgs.append(img.cpu())

        self.model.train()
        return imgs


# ==================== 训练和比较 ====================

def compare_ddpm_ddim():
    """比较 DDPM 和 DDIM 的示例"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建示例数据（这里用随机数据代替真实数据集）
    def create_dummy_data(batch_size=64, image_size=28):
        # 创建类似MNIST的随机数据
        data = torch.randn(batch_size * 100, 1, image_size, image_size) * 0.5 + 0.5
        data = torch.clamp(data, 0, 1)
        return data

    # 初始化模型
    ddpm = DDPM(timesteps=1000, image_size=28, device=device)
    ddim = DDIM(timesteps=1000, sampling_timesteps=50, eta=0.0, image_size=28, device=device)

    # 为了公平比较，让DDIM使用训练好的DDPM模型权重
    ddim.model.load_state_dict(ddpm.model.state_dict())

    # 创建虚拟数据加载器
    dummy_data = create_dummy_data()
    dataloader = DataLoader(dummy_data, batch_size=64, shuffle=True)

    # 训练DDPM（简化版训练循环）
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=1e-3)

    print("训练DDPM中...")
    for epoch in range(2):  # 简化的训练轮数
        for batch in dataloader:
            batch = batch.to(device)
            loss = ddpm.train_step(batch, optimizer)

        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # 采样比较
    print("\n生成样本...")

    # DDPM 采样（完整1000步）
    print("DDPM 采样（1000步）...")
    ddpm_samples = ddpm.sample(num_samples=4)

    # DDIM 采样（仅50步）
    print("DDIM 采样（50步）...")
    ddim_samples = ddim.sample(num_samples=4)

    # 可视化结果
    fig, axes = plt.subplots(2, len(ddpm_samples), figsize=(15, 6))

    # 显示DDPM生成过程
    for i, sample in enumerate(ddpm_samples):
        axes[0, i].imshow(sample[0].squeeze(), cmap='gray')
        axes[0, i].set_title(f'DDPM t={1000 - i * 100}')
        axes[0, i].axis('off')

    # 显示DDIM生成过程
    for i, sample in enumerate(ddim_samples):
        axes[1, i].imshow(sample[0].squeeze(), cmap='gray')
        axes[1, i].set_title(f'DDIM step={i * 10}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # 性能比较
    import time

    # DDPM 采样时间
    start_time = time.time()
    ddpm.sample(num_samples=1)
    ddpm_time = time.time() - start_time

    # DDIM 采样时间
    start_time = time.time()
    ddim.sample(num_samples=1)
    ddim_time = time.time() - start_time

    print(f"\n性能比较:")
    print(f"DDPM 采样时间 ({ddpm.timesteps}步): {ddpm_time:.2f}秒")
    print(f"DDIM 采样时间 ({ddim.sampling_timesteps}步): {ddim_time:.2f}秒")
    print(f"加速比: {ddpm_time / ddim_time:.1f}x")


if __name__ == "__main__":
    compare_ddpm_ddim()
