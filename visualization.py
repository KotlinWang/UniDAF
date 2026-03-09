import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from PIL import Image
import argparse
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model.unidaf3.change import Change
# from model.ablation.baseline_sie.change import Change
# from model.ablation.baseline.change import Change
# from model.ablation.baseline_tad.change import Change

from einops import rearrange
from pathlib import Path
from torchvision.utils import save_image
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt
import torch.nn.functional as F

def img_loader(path):
    """加载图像并转换为numpy数组"""
    img = np.array(Image.open(path), np.float32)
    return img

def preprocess_image(root_dir, pre_img_path, post_opt_img_path, post_sar_img_path):
    """预处理输入图像"""
    # 加载图像
    pre_img = img_loader(os.path.join(root_dir, pre_img_path))[:,:,0:3]  # 只取RGB通道
    post_opt_img = img_loader(os.path.join(root_dir, post_opt_img_path))[:,:,0:3]  # 只取RGB通道
    post_sar_img = img_loader(os.path.join(root_dir, post_sar_img_path))
    
    # SAR图像转换为3通道
    post_sar_img = np.stack((post_sar_img,)*3, axis=-1)
    
    # 应用与验证集相同的预处理
    val_transform = A.Compose([
        A.Normalize(),
        ToTensorV2(),
    ], additional_targets={
        'image1': 'image',
        'image2': 'image'
    })
    
    # 创建一个占位符标签，实际推理中不会使用
    dummy_label = np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=np.float32)
    
    # 应用变换
    augmented = val_transform(image=pre_img, image1=post_opt_img, image2=post_sar_img, mask=dummy_label)
    
    return augmented['image'], augmented['image1'], augmented['image2']


# 定义最小-最大归一化函数
def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)  # 加上一个小值防止除以零

def viz(feats, output_dir):
    for i, feat in enumerate(feats):
        B, C, H, W = feat.shape
        N = B * H * W
        E = rearrange(feat, "B C H W -> (B H W) C")  # 形状为 (N, D)

        # 直接对所有像素进行PCA分析，投影到前三个主成分
        _, _, V = torch.pca_lowrank(E)
        E_pca3 = E @ V[:, :3]  # 投影到前三个主成分，形状为 (N, 3)
        
        # 极简的颜色处理，保持原始数据分布
        # 1. 最基础的百分位数裁剪，仅移除极端异常值
        # 2. 简单的归一化，保留原始PCA分布特性
        # 3. 避免任何复杂的色彩增强和变换
        
        # 使用更宽范围的百分位数裁剪，保留绝大部分原始分布
        # 只移除极端异常值
        lower_percentile = 1
        upper_percentile = 99
        lower_bound = torch.quantile(E_pca3, lower_percentile / 100, dim=0)
        upper_bound = torch.quantile(E_pca3, upper_percentile / 100, dim=0)
        
        # 裁剪极端异常值
        E_pca3_clipped = torch.clamp(E_pca3, lower_bound, upper_bound)
        
        # 简单的最小-最大归一化到0-1范围
        # 这是最接近原始数据分布的处理方式
        E_pca3_final = minmax_norm(E_pca3_clipped)
        
        # 重塑为 (B, H, W, 3)
        I_draw = rearrange(E_pca3_final, "(B H W) C -> B H W C", B=B, H=H, W=W)
        
        # 转换为通道优先格式
        I_draw = rearrange(I_draw, "B H W C -> B C H W")
        
        # 调整大小以匹配原始图像尺寸
        I_draw_resized = resize(I_draw, (1024, 1024))

        save_image(I_draw_resized.cpu(), str(os.path.join(output_dir, f"{i}.png")))


def viz_feat(feats, output_dir):
    for i, feat in enumerate(feats):
        # feat = F.interpolate(feat, size=(1024, 1024), mode='bilinear')

        plt.axis('off')
        # plt.imshow(feat.sum(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='twilight_shifted')
        # plt.imshow(feat.sum(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='RdBu')
        # plt.imshow(feat.mean(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='RdYlBu')
        # plt.imshow(feat.sum(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='hsv')
        plt.imshow(feat.mean(dim=1).contiguous().data.cpu().numpy()[0, :, :], cmap='twilight')
        plt.savefig(os.path.join(output_dir, f'feat_{i}.jpg'), bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close()
        

def infer_single_image(args):
    """对单张图像进行推理"""
    # 设置模型
    model = Change('resnet18.fb_swsl_ig1b_ft_in1k', 2, 4, 128)
    model = model.cuda()
    model.eval()
    
    # 加载预训练权重
    if args.existing_weight_path is not None:
        if not os.path.isfile(args.existing_weight_path):
            raise RuntimeError(f"没有找到模型权重文件: {args.existing_weight_path}")
        checkpoint = torch.load(args.existing_weight_path, weights_only=True)
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
        print(f'已加载预训练权重: {args.existing_weight_path}')
    
    # 颜色映射
    color_map = {
        0: (255, 255, 255),   # No damage - 白色
        1: (35,77,161),       # Intact - 蓝色
        2: (249,202,94),      # Damaged - 橙色
        3: (218,80,37)        # Destroyed - 红色
    }
    
    # 创建输出目录
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, now_str)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 预处理图像
    pre_img, post_opt_img, post_sar_img = preprocess_image(
        args.root_dir,
        args.pre_event_img,
        args.post_event_opt_img,
        args.post_event_sar_img
    )
    
    # 添加批次维度
    pre_img = pre_img.unsqueeze(0).cuda()
    post_opt_img = post_opt_img.unsqueeze(0).cuda()
    post_sar_img = post_sar_img.unsqueeze(0).cuda()
    
    # 合并输入
    img_data = torch.cat([pre_img, post_opt_img, post_sar_img], dim=0)
    
    # 推理
    with torch.no_grad():
        feats = model(img_data)

        # viz(feats, output_dir)
        viz_feat(feats, output_dir)

if __name__ == "__main__":
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 根据需要修改
    
    parser = argparse.ArgumentParser(description="对单张图像进行建筑物损伤评估推理")
    
    # 输入图像路径
    parser.add_argument('--root_dir', type=str, default='/shared/kotlin/DATASET/AegisDA/DisasterSet2')
    parser.add_argument('--pre_event_img', type=str, default='pre-event/hawaii-wildfire_00000030_pre_disaster.tif')
    parser.add_argument('--post_event_opt_img', type=str, default='post-event-opt/hawaii-wildfire_00000030_post_disaster_opt.tif')
    parser.add_argument('--post_event_sar_img', type=str, default='post-event-sar/hawaii-wildfire_00000030_post_disaster_sar.tif')
    
    # 模型和输出设置
    parser.add_argument('--existing_weight_path', type=str,
                        default='./saved_weights/AegisDA/unidaf3_resnet18_cross/best_sk_model.pth',
                        help='预训练模型权重路径')
    parser.add_argument('--output_dir', type=str, default='./inference_results_single',
                        help='预测结果保存目录')
    parser.add_argument('--use_tta', action='store_true', help='启用测试时增强(TTA)', default=False)
    
    args = parser.parse_args()
    
    # 运行推理
    infer_single_image(args)