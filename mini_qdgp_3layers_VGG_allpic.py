import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torchvision.models as models

# 1. 环境配置
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "qdgp_high_res_results"
os.makedirs(output_dir, exist_ok=True)
IMG_SIZE = 64  # 提升到 64x64 像素

# ==========================================
# 2. VGG 感知损失 (适配大图)
# ==========================================
vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def get_perceptual_loss(input_img, target_img):
    def preprocess(img):
        img_3ch = img.repeat(1, 3, 1, 1) # 1通道转3通道
        return nn.functional.interpolate(img_3ch, size=(224, 224), mode='bilinear', align_corners=False)
    features_input = vgg(preprocess(input_img))
    features_target = vgg(preprocess(target_img))
    return torch.mean((features_input - features_target)**2)

# ==========================================
# 3. 图像加载逻辑 (修正路径定位)
# ==========================================
def load_custom_image(abs_path):
    print(f"🔍 正在尝试加载图片: {abs_path}")
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"❌ 找不到图片：{abs_path}")
    img = Image.open(abs_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).view(1, 1, IMG_SIZE, IMG_SIZE)

# 自动获取脚本同目录下的 target.jpg
current_script_dir = os.path.dirname(os.path.abspath(__file__))
final_target_path = os.path.join(current_script_dir, "target.jpg")
target = load_custom_image(final_target_path).to(device)

# --- 遮罩设置：在中间挖掉一个小方块 ---
mask = torch.ones(1, 1, IMG_SIZE, IMG_SIZE).to(device)
box_size = 10  # 方块大小
start_y = (IMG_SIZE - box_size) // 2  # 居中起始行
start_x = (IMG_SIZE - box_size) // 2  # 居中起始列
mask[:, :, start_y:start_y+box_size, start_x:start_x+box_size] = 0.0 

corrupted_img = target * mask

# ==========================================
# 4. 增强版 Generator (适配 64x64)
# ==========================================
class Generator64(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2), # 8x8
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 16x16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 32x32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # 64x64
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc(x).view(-1, 256, 4, 4)
        return self.conv(x)

# 量子部分维持 3 层深度
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
# 5. 优化
# ==========================================
gen = Generator64(latent_dim=n_qubits).to(device)
q_weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.5) 
optimizer = torch.optim.Adam([{"params": gen.parameters()}, {"params": q_weights}], lr=0.005)

print(f"🚀 启动 64x64 感知驱动修复 (方块大小: {box_size}x{box_size})...")
for step in range(801): # 步数增加一点，让细节更完美
    optimizer.zero_grad()
    z = torch.stack(q_circuit(q_weights)).float().unsqueeze(0).to(device)
    generated_img = gen(z)
    
    # 像素损失 (只看没遮住的地方)
    pixel_loss = torch.mean(((generated_img - target) * mask)**2)
    # 感知损失 (全局审美引导)
    #vgg_loss = 1.5 * get_perceptual_loss(generated_img, target) 
    vgg_loss = 1.5 * get_perceptual_loss(generated_img * mask, corrupted_img * mask)
    
    loss = pixel_loss + vgg_loss
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step} | Pixel Loss: {pixel_loss.item():.6f} | VGG Loss: {vgg_loss.item():.6f}")
        plt.figure(figsize=(10, 3))
        plt.subplot(131); plt.imshow(target[0,0].cpu(), cmap='gray'); plt.title("Target")
        plt.subplot(132); plt.imshow(corrupted_img[0,0].cpu(), cmap='gray'); plt.title("Corrupted")
        plt.subplot(133); plt.imshow(generated_img[0,0].detach().cpu(), cmap='gray'); plt.title(f"Step {step}")
        plt.savefig(f"{output_dir}/high_res_{step}.png")
        plt.close()

print(f"🎉 高清修复完成！快去 {output_dir} 看看吧！")