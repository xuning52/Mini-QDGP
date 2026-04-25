import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torchvision.models as models # 必须引入

# 1. 环境配置 (建议 CPU，若显存够可切 CUDA)
device = torch.device("cpu")
output_dir = "qdgp_vgg_results"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 引入预训练 VGG16 提取感知特征
# ==========================================
# 加载预训练的 VGG16 模型
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False # 冻结参数，我们只用它来计算 Loss

def get_perceptual_loss(input_img, target_img):
    # VGG16 期望 224x224 的 3 通道输入
    def preprocess(img):
        img_3ch = img.repeat(1, 3, 1, 1) # 1通道转3通道
        return nn.functional.interpolate(img_3ch, size=(224, 224), mode='bilinear', align_corners=False)
    
    features_input = vgg(preprocess(input_img))
    features_target = vgg(preprocess(target_img))
    return torch.mean((features_input - features_target)**2)

# ==========================================
# 3. 图像处理与模型定义
# ==========================================
# def load_custom_image(path):
#     if not os.path.exists(path):
#         img = np.zeros((28, 28))
#         y, x = np.ogrid[:28, :28]
#         mask_circle = (x-14)**2 + (y-14)**2
#         img[(mask_circle > 60) & (mask_circle < 120)] = 1.0
#         return torch.FloatTensor(img).view(1, 1, 28, 28)
#     img = Image.open(path).convert('L').resize((28, 28))
#     return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).view(1, 1, 28, 28)

def load_custom_image(path):
    # 1. 自动获取脚本所在的文件夹绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 拼接出 target.jpg 的绝对路径
    # 这样无论你在哪个目录下启动 Python，它都会去脚本旁边找图片
    target_path = os.path.join(current_dir, "target.jpg")

    # 3. 使用新路径加载图片
    target = load_custom_image(target_path).to(device)


target = load_custom_image("target.jpg").to(device)
mask = torch.ones(1, 1, 28, 28).to(device)
mask[:, :, :, :14] = 0.0 
corrupted_img = target * mask

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

# 三层量子电路
n_qubits = 8
n_layers = 3
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def q_circuit(weights):
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

# ==========================================
# 4. 优化过程 (关键修正)
# ==========================================
gen = Generator(latent_dim=n_qubits).to(device)
# 修正：权重维度必须与电路一致 (3, 8, 2)
q_weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.1)

# 使用稍微大一点的学习率来跳出局部最优
optimizer = torch.optim.Adam([{"params": gen.parameters()}, {"params": q_weights}], lr=0.01)

print("🚀 开始感知驱动的 QDGP 优化...")
for step in range(501): # 步数增加，让 VGG 充分引导
    optimizer.zero_grad()
    z = torch.stack(q_circuit(q_weights)).float().unsqueeze(0)
    generated_img = gen(z)
    
    # 损失函数组合：像素损失 (已知区域) + 感知损失 (全局结构)
    pixel_loss = torch.mean(((generated_img - target) * mask)**2)
    # 感知损失强迫生成器“补全”左半边的结构特征
    #vgg_loss = 0.5 * get_perceptual_loss(generated_img, target)
    vgg_loss = 0.5 * get_perceptual_loss(generated_img * mask, corrupted_img * mask) 
    
    loss = pixel_loss + vgg_loss
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step} | Pixel Loss: {pixel_loss.item():.6f} | VGG Loss: {vgg_loss.item():.6f}")
        plt.figure(figsize=(10, 3))
        plt.subplot(131); plt.imshow(target[0,0].cpu(), cmap='gray'); plt.title("Target")
        plt.subplot(132); plt.imshow(corrupted_img[0,0].cpu(), cmap='gray'); plt.title("Corrupted")
        plt.subplot(133); plt.imshow(generated_img[0,0].detach().cpu(), cmap='gray'); plt.title(f"Step {step}")
        plt.savefig(f"{output_dir}/vgg_recovery_{step}.png")
        plt.close()

print(f"🎉 任务完成！感知引导的结果已保存。")