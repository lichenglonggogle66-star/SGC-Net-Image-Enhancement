import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

# ==========================================
# 0. 全局配置与工具函数
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32" # 使用公开的CLIP权重
FEATURE_DIM = 512 # CLIP-Vit-Base 的默认维度
CURVE_PIECES = 3 # 您提及的二次曲线，预测3个参数：a, b, c

def load_and_preprocess_image(img_path, size=224):
    """加载并预处理图像，适用于CLIP和网络输入"""
    img = Image.open(img_path).convert('RGB')
    
    # 1. CLIP 预处理
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip_inputs = processor(images=img, return_tensors="pt").to(device)
    
    # 2. 图像增强网络输入 (0-1 range)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    net_input = transform(img).unsqueeze(0).to(device) # Add batch dim
    
    return img, clip_inputs, net_input

def show_results(imgs, titles, text_desc=None):
    """显示增强结果对比"""
    n = len(imgs)
    plt.figure(figsize=(6 * n, 5))
    if text_desc:
        plt.suptitle(f"Semantic Prior: \"{text_desc}\"", fontsize=12, y=0.98)
        
    for i in range(n):
        plt.subplot(1, n, i + 1)
        # Convert to numpy for matplotlib
        img_np = imgs[i].detach().cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1) # Ensure valid range
        plt.imshow(img_np)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ==========================================
# 1. 离线多模态特征提取 (Section 2.1)
# ==========================================
class OfflineFeatureExtractor(nn.Module):
    """
    模拟论文公式 (1) 和 (2)。
    由于无法离线连接GPT，我们直接提供针对输入图像设计的语义文本 P。
    """
    def __init__(self):
        super().__init__()
        # 加载预训练的 CLIP 模型用于图像和文本编码
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self.eval() # 冻结 CLIP 权重

    @torch.no_grad()
    def forward(self, clip_inputs, text_priors):
        """
        输入: CLIP预处理后的图像, 语义文本列表
        输出: 视觉特征 Fv (B, 1, D), 语义特征 Ft (B, N, D)
        """
        batch_size = clip_inputs['pixel_values'].shape[0]
        
        # 2.1.2 提取视觉特征 Fv (使用图像编码器Ev)
        image_embeds = self.clip.get_image_features(**clip_inputs)
        Fv = image_embeds.unsqueeze(1) # (B, 1, FEATURE_DIM)

        # 2.1.2 提取语义特征 Ft (使用文本编码器Et)
        # 对文本进行预处理和编码
        text_inputs = self.processor(text=text_priors, padding=True, return_tensors="pt").to(device)
        text_embeds = self.clip.get_text_features(**text_inputs)
        
        # 模拟可能有多个文本或token的情况，将所有文本视为一个整体语义锚点
        num_texts = len(text_priors)
        Ft = text_embeds.unsqueeze(0).repeat(batch_size, 1, 1) # (B, N_texts, FEATURE_DIM)

        return Fv, Ft


# ==========================================
# 2. 跨模态校准 (CMC) 模块 (Section 2.2, Fig. 2)
# ==========================================
class CrossModalCalibration(nn.Module):
    """
    严格按照图 2 实现的双向注意力校准机制。
    包含：S2V、V2S 和 门控融合。
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        
        # 2.2.3 Semantic-to-Visual Stream (S2V)
        # Q: Fv, K/V: Ft
        self.s2v_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # 2.2.3 Visual-to-Semantic Stream (V2S)
        # Q: Ft, K/V: Fv
        self.v2s_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # 2.2.4 门控融合机制 (Gated Fusion)
        # 学习一个可训练的权重参数 λ
        self.lambda_gate = nn.Parameter(torch.zeros(1))
        
        # 2.2.4 前馈网络 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, Fv, Ft):
        """
        输入: Fv (B, 1, D), Ft (B, N, D)
        输出: 融合特征 Ffused (B, 1, D)
        """
        # A. Semantic-to-Visual (Q: V, K/V: T) -> F_s2v (B, 1, D)
        F_s2v, _ = self.s2v_attn(query=Fv, key=Ft, value=Ft)
        
        # B. Visual-to-Semantic (Q: T, K/V: V) -> F_v2s (B, N, D)
        F_v2s, _ = self.v2s_attn(query=Ft, key=Fv, value=Fv)
        
        # C. 聚合 Visual Context (对 Ft 维度进行池化，得到聚合的上下文矢量)
        # 这确保 F_v2s 具有与 Fv 相同的 (B, 1, D) 形状用于融合
        F_v2s_pooled = F_v2s.mean(dim=1, keepdim=True) # (B, 1, D)

        # D. 2.2.4 门控融合 Formula (4)
        # F_calibrated = F_s2v + λ * F_v2s_pooled
        F_calibrated = F_s2v + self.lambda_gate * F_v2s_pooled
        
        # E. FFN + 残差连接
        F_fused = self.norm(F_calibrated + self.ffn(F_calibrated))
        
        return F_fused


# ==========================================
# 3. 语义驱动的曲线映射 (Section 2.3)
# ==========================================
class CurveMappingNet(nn.Module):
    """
    根据融合特征 Ffused 预测逐像素增强曲线的参数 A。
    """
    def __init__(self, input_dim):
        super().__init__()
        # 预测二次曲线参数 A: a, b, c
        # 输出维度: Batch * Height * Width * (3 params * 3 channels)
        # 为了演示，我们简化输出，只预测全局或低分辨率的参数，然后上采样
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 3 * 3), # 3 params (a,b,c) for 3 RGB channels
            nn.Tanh() # 限制参数范围在 -1 到 1
        )

    def forward(self, Ffused, h, w):
        """
        输入: Ffused (B, 1, D), 输入图像的H, W
        输出: 曲线参数 A (B, 3, H, W, 3)
        """
        B = Ffused.shape[0]
        # 提取参数: (B, 1, D) -> (B, 9)
        A_flat = self.projection_head(Ffused.squeeze(1))
        
        # 重塑为 (B, 1, 1, 3 params, 3 channels) 用于广播
        A_global = A_flat.view(B, 1, 1, 3, 3)
        
        # 将全局参数上采样到图像尺寸 (B, H, W, 3, 3)
        A = A_global.expand(B, h, w, 3, 3)
        return A


# ==========================================
# 4. 完整的 SGC-Net 增强系统
# ==========================================
class SGCNetEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = OfflineFeatureExtractor()
        self.cmc = CrossModalCalibration(FEATURE_DIM)
        self.curve_mapping = CurveMappingNet(FEATURE_DIM)

    def apply_curve(self, image, A):
        """
        应用二次曲线: Iout = a*I^2 + b*I + c
        image: (B, 3, H, W) -> range [0, 1]
        A: (B, H, W, 3, 3) -> params [a, b, c] for [R, G, B]
        """
        B, C, H, W = image.shape
        # 重塑图像用于计算: (B, H, W, C)
        I = image.permute(0, 2, 3, 1)
        
        # 提取参数 (a, b, c)
        a = A[..., 0] # (B, H, W, C)
        b = A[..., 1] # (B, H, W, C)
        c = A[..., 2] # (B, H, W, C)
        
        # 计算二次曲线 transformation
        Iout = a * torch.pow(I, 2) + b * I + c
        
        # 转换回 (B, C, H, W)
        Iout = Iout.permute(0, 3, 1, 2)
        return Iout

    def forward(self, clip_inputs, net_input, text_priors):
        # 2.1 离线/静态特征提取
        Fv, Ft = self.feature_extractor(clip_inputs, text_priors)
        
        # 2.2 跨模态校准 (CMC)
        Ffused = self.cmc(Fv, Ft)
        
        # 2.3 曲线预测
        _, _, H, W = net_input.shape
        A = self.curve_mapping(Ffused, H, W)
        
        # 2.3 应用曲线
        Iout = self.apply_curve(net_input, A)
        return Iout


# ==========================================
# 5. 测试演示 (Demo)
# ==========================================
def run_demo():
    print(f"Running SGC-Net Demo on {device}...")
    
    # 1. 创建符合论文场景的测试数据
    # A. 创建一个低光、逆光的图像（模拟天空和逆光前景）
    test_img = Image.new('RGB', (400, 300), color=(20, 20, 30)) # 暗色基底
    # 模拟天空（较亮）
    for y in range(100):
        for x in range(400):
            test_img.putpixel((x, y), (100, 100, 130))
    # 保存测试图
    test_img_path = "demo_input.png"
    test_img.save(test_img_path)
    
    # B. 设计符合 prompt P 要求的语义描述 (Offline process simulation)
    # 此描述针对上述逆光场景
    semantic_prior = "A dark image with a dimly lit sky and an underexposed foreground, potentially a backlit scene."
    text_priors = [semantic_prior]

    # 2. 加载和预处理数据
    print("Loading image and initializing models...")
    original_img, clip_inputs, net_input = load_and_preprocess_image(test_img_path)

    # 3. 初始化 SGC-Net
    model = SGCNetEnhancer().to(device)
    # 由于没有训练，我们将 CMC 的 λ 设为正值以演示 Ft 的影响
    model.cmc.lambda_gate.data.fill_(1.0) 
    model.eval()

    # 4. 执行推理 (Inference)
    print("Performing Semantically-Guided Image Enhancement...")
    with torch.no_grad():
        enhanced_tensor = model(clip_inputs, net_input, text_priors)

    # 5. 显示结果
    print("Displaying results...")
    show_results([net_input, enhanced_tensor], ["Input (Low Light)", "SGC-Net Enhanced"], text_desc=semantic_prior)

if __name__ == "__main__":
    run_demo()
