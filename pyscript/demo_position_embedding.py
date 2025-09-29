def get_sinusoid_encoding_table(n_position, d_pos_vec):
    """
    Sinusoid position encoding table
    """

    import numpy as np
    import torch

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


if __name__ == '__main__':
    n_position = 200  # maximum position
    d_pos_vec = 512  # dimension of position encoding vector

    pe = get_sinusoid_encoding_table(n_position, d_pos_vec)
    print(pe.shape)  # torch.Size([200, 512])
    print(pe)

    # 使用非交互式后端绘图并保存
    import matplotlib

    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Sinusoidal Position Encoding')
    plt.xlabel('Dimension')
    plt.ylabel('Position')
    plt.tight_layout()
    plt.savefig('position_encoding.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 position_encoding.png")
