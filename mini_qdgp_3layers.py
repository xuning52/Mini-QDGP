import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # 用这个替代 torchvision
import os

# 1. 环境配置 (纯 CPU)
device = torch.device("cpu")
output_dir = "qdgp_results"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 自定义输入图片处理
# ==========================================
def load_custom_image(path):
    if not os.path.exists(path):
        print(f"未发现图片 {path}，正在生成默认圆环...")
        img = np.zeros((28, 28))
        y, x = np.ogrid[:28, :28]
        mask_circle = (x-14)**2 + (y-14)**2
        img[(mask_circle > 60) & (mask_circle < 120)] = 1.0
        return torch.FloatTensor(img).view(1, 1, 28, 28)
    
    # 手动实现 torchvision 的功能：加载 -> 灰度 -> 缩放 -> 归一化
    img = Image.open(path).convert('L') # 灰度
    img = img.resize((28, 28)) # 缩放
    img_np = np.array(img).astype(np.float32) / 255.0 # 归一化到 [0, 1]
    return torch.from_numpy(img_np).view(1, 1, 28, 28)

# 这里你可以换成你自己的图片路径，比如 "my_data.jpg"
target = load_custom_image("target.jpg")

# 创建遮罩 (Mask)：遮住左半边 (0代表遮住，1代表可见)
mask = torch.ones(1, 1, 28, 28)
mask[:, :, :, :14] = 0.0 
corrupted_img = target * mask

# ==========================================
# 3. 模型定义 (Generator & PQC)
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x).view(-1, 128, 7, 7)
        return self.conv(x)


# 升级版：三层强纠缠量子电路 (Inspired by Xiao Tailong)
n_qubits = 8
n_layers = 3  # 增加到三层深度
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def q_circuit(weights):
    # weights 的形状预期为 (n_layers, n_qubits, 2)
    for l in range(n_layers):
        # 1. 旋转层：每一层都有独立的旋转参数
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
        
        # 2. 纠缠层：使用环形 CNOT 增加比特间的关联
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
    # 返回每个比特的期望值
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# 更新参数初始化
# 我们给它 3 层，每层每个 qubit 2 个旋转角
q_weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

# ==========================================
# 4. 优化过程
# ==========================================
gen = Generator(latent_dim=n_qubits).to(device)
q_weights = torch.nn.Parameter(torch.randn(n_qubits, 2) * 0.5)
optimizer = torch.optim.Adam([{"params": gen.parameters()}, {"params": q_weights}], lr=0.01)

print("开始优化...")
for step in range(301):
    optimizer.zero_grad()
    z = torch.stack(q_circuit(q_weights)).float().unsqueeze(0)
    generated_img = gen(z)
    
    # 损失函数：只对比 Mask 为 1 的区域
    loss = torch.mean(((generated_img - target) * mask)**2)
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step} | Loss: {loss.item():.6f}")
        # 离线保存对比图
        plt.figure(figsize=(10, 3))
        plt.subplot(131); plt.imshow(target[0,0], cmap='gray'); plt.title("Target")
        plt.subplot(132); plt.imshow(corrupted_img[0,0], cmap='gray'); plt.title("Corrupted")
        plt.subplot(133); plt.imshow(generated_img[0,0].detach(), cmap='gray'); plt.title(f"QDGP Step {step}")
        plt.savefig(f"{output_dir}/comparison_{step}.png")
        plt.close() # 释放内存，不弹窗

print(f"🎉 任务完成！结果已保存至 {output_dir} 文件夹。")