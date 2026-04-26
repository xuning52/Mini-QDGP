# Mini-QDGP: Quantum Deep Generative Prior (Pytorch + PennyLane)

本项目是基于论文 **"Quantum Deep Generative Prior with Programmable Quantum Circuits" (Xiao et al.)** 的简化实现与实验记录。通过结合参数化量子电路 (PQC) 与深度卷积生成网络，探索量子变分电路作为生成器先验在图像修复（Image Inpainting）任务中的潜力。

## 🚀 演进历程 (Milestones)

本项目记录了从基础模型到感知驱动模型的完整演进过程：

1. **V1: 2-Layer PQC (`mini_qdgp_2layers.py`)**
   - 基础两层量子线路，尝试修复 28x28 破损圆环。
   - 痛点：量子表达能力不足，难以拟合复杂的几何形状。
2. **V2: 3-Layer Strong Entanglement (`mini_qdgp_3layers.py`)**
   - 引入强纠缠层（Strongly Entangling Layers），增加至三层深度。
   - 痛点：虽然表达能力提升，但在纯像素损失（MSE）引导下，模型容易陷入局部最优（出现全黑图像）。
3. **V3: Perceptual Guide (`mini_qdgp_3layers_VGG.py`)**
   - **突破点**：引入预训练 **VGG16 感知损失**。利用深度特征空间引导量子先验补全缺失区域。
4. **V4: High-Res 64x64 (`mini_qdgp_3layers_VGG_allpic.py`)**
   - 优化生成器架构，支持 64x64 分辨率。
   - 改进路径引用逻辑，适配更灵活的数据集输入。

## 🛠️ 技术栈
- **Quantum**: PennyLane (default.qubit)
- **Deep Learning**: PyTorch, Torchvision
- **Vision**: VGG16 (Perceptual Loss)

## 📖 参考文献与致谢
- **Original Paper**: [Quantum Deep Generative Prior with Programmable Quantum Circuits](https://doi.org/10.1038/s42005-024-01765-9) by Tailong Xiao, et al.
- **Inspiration**: 本项目受 Xiao Tailong 老师的研究启发，旨在通过简化代码实现（Mini-version）帮助初学者理解 QDGP 的核心机制。

## 👤 作者
Xu Ning (GitHub: [xuning52](https://github.com/xuning52))

---

## [2026-04-25] -- v0.2.0(Perceptual Update)
**Changed**

- 修正感知损失逻辑：将 VGG Perceptual Loss 的对比对象从原图 (target) 修改为受损图像 (corrupted_img) 的掩模区域。此举解决了训练过程中的“标签泄露（Label Leakage）”问题，确保模型在推理阶段完全基于图像先验进行修复。

**Added**

- 硬件自适应支持：增加了对 CUDA 设备的自动检测逻辑，优先调用 NVIDIA GPU (如 RTX 5060 Ti) 进行感知特征提取加速。