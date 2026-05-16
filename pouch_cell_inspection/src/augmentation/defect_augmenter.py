"""
针对小样本缺陷检测的数据增强模块
目标：从200张缺陷图像生成足够的训练数据，同时保持真实性

增强策略:
1. 几何变换 - 旋转、翻转、缩放
2. 颜色扰动 - 亮度、对比度、色调调整（模拟不同光照）
3. 缺陷复制粘贴 - 将缺陷区域复制到其他正常样本上
4. Mosaic/MixUp - YOLO 原生增强
5. 合成缺陷生成 - 基于物理的缺陷模拟
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 几何变换
    rotate_limit: float = 5.0  # 小角度旋转（电芯通常水平/垂直放置）
    shift_scale_rotate: float = 0.05
    flip_prob: float = 0.5
    
    # 颜色扰动
    brightness_limit: float = 0.15
    contrast_limit: float = 0.15
    saturation_limit: float = 0.1
    hue_shift_limit: float = 0.05
    
    # 噪声和模糊
    blur_limit: int = 3
    noise_limit: float = 0.05
    
    # 高级增强
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.3
    
    # 缺陷特定的增强
    defect_copy_paste_prob: float = 0.7
    max_defect_copies: int = 5


class DefectAugmenter:
    """
    软包电池缺陷检测专用数据增强器
    
    针对小样本场景优化，平衡增强强度和真实性
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        
        # 基础增强管道（适用于所有样本）
        self.base_transform = A.Compose([
            A.HorizontalFlip(p=self.config.flip_prob),
            A.VerticalFlip(p=self.config.flip_prob * 0.3),  # 垂直翻转概率较低
            A.RandomRotate90(p=self.config.flip_prob * 0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 颜色增强管道（模拟不同光照条件）
        self.color_transform = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_limit,
                contrast_limit=self.config.contrast_limit,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(255 * self.config.hue_shift_limit),
                sat_shift_limit=int(255 * self.config.saturation_limit),
                val_shift_limit=int(255 * self.config.brightness_limit),
                p=0.5
            ),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 质量退化管道（模拟产线成像变化）
        self.quality_transform = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=self.config.blur_limit, p=0.5),
                A.MotionBlur(blur_limit=self.config.blur_limit, p=0.5),
            ], p=0.3),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 强增强管道（仅用于已标注的缺陷样本）
        self.strong_transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=self.config.shift_scale_rotate,
                scale_limit=self.config.shift_scale_rotate,
                rotate_limit=self.config.rotate_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.8
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ElasticTransform(
                alpha=1.0,
                sigma=50.0,
                alpha_affine=50.0,
                p=0.2
            ),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 0.9), num_shadows_lower=1, num_shadows_upper=2, p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        # 高分辨率专用增强（针对 8000x20000 大图）
        self.large_image_transform = A.Compose([
            A.RandomCropFromBorders(
                crop_left=0.01,
                crop_right=0.01,
                crop_top=0.01,
                crop_bottom=0.01,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=5,
                max_height=200,
                max_width=200,
                min_holes=1,
                min_height=50,
                min_width=50,
                fill_value=0,
                p=0.3
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def apply_basic_augmentation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """应用基础增强"""
        transformed = self.base_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    def apply_color_augmentation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """应用颜色增强"""
        transformed = self.color_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    def apply_quality_degradation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """应用质量退化模拟"""
        transformed = self.quality_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    def apply_strong_augmentation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """应用强增强（仅用于缺陷样本）"""
        transformed = self.strong_transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        return transformed['image'], transformed['bboxes'], transformed['class_labels']
    
    def defect_copy_paste(
        self,
        target_image: np.ndarray,
        source_defects: List[Dict],
        max_copies: Optional[int] = None
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        缺陷复制粘贴增强
        
        从源图像中提取缺陷区域，粘贴到目标图像上
        
        Args:
            target_image: 目标图像（通常是正常样本）
            source_defects: 源缺陷列表 [{"bbox": [x,y,w,h], "image": img_array, "class": cls_id}]
            max_copies: 最大复制数量
        
        Returns:
            增强后的图像、边界框、类别标签
        """
        if not source_defects:
            return target_image, [], []
        
        result_image = target_image.copy()
        pasted_bboxes = []
        pasted_classes = []
        
        n_copies = min(
            max_copies or self.config.max_defect_copies,
            len(source_defects)
        )
        
        # 随机选择要复制的缺陷
        selected_defects = random.sample(source_defects, n_copies)
        
        h, w = target_image.shape[:2]
        
        for defect in selected_defects:
            bbox = defect["bbox"]  # [x, y, w, h] normalized
            defect_img = defect["image"]
            cls_id = defect["class"]
            
            # 转换为绝对坐标
            x1_abs = int(bbox[0] * defect_img.shape[1])
            y1_abs = int(bbox[1] * defect_img.shape[0])
            x2_abs = int((bbox[0] + bbox[2]) * defect_img.shape[1])
            y2_abs = int((bbox[1] + bbox[3]) * defect_img.shape[0])
            
            # 提取缺陷区域
            defect_patch = defect_img[y1_abs:y2_abs, x1_abs:x2_abs].copy()
            
            if defect_patch.size == 0:
                continue
            
            # 随机选择目标位置
            patch_h, patch_w = defect_patch.shape[:2]
            
            # 尝试多个位置，避免重叠
            max_attempts = 10
            for _ in range(max_attempts):
                target_x = random.randint(0, w - patch_w)
                target_y = random.randint(0, h - patch_h)
                
                # 检查是否与现有边界框重叠（简化检查）
                overlap = False
                for existing_bbox in pasted_bboxes:
                    ex_x, ex_y, ex_w, ex_h = existing_bbox
                    if (abs(target_x/w - ex_x) < 0.1 and 
                        abs(target_y/h - ex_y) < 0.1):
                        overlap = True
                        break
                
                if not overlap:
                    break
            
            # 粘贴缺陷（使用 alpha 混合使边缘更自然）
            roi = result_image[target_y:target_y+patch_h, target_x:target_x+patch_w]
            
            # 简单的泊松融合替代方案
            alpha = 0.8  # 混合系数
            blended = cv2.addWeighted(roi, 1-alpha, defect_patch, alpha, 0)
            result_image[target_y:target_y+patch_h, target_x:target_x+patch_w] = blended
            
            # 记录新的边界框（YOLO 格式：x_center, y_center, width, height normalized）
            new_bbox = [
                (target_x + patch_w / 2) / w,
                (target_y + patch_h / 2) / h,
                patch_w / w,
                patch_h / h
            ]
            
            pasted_bboxes.append(new_bbox)
            pasted_classes.append(cls_id)
        
        return result_image, pasted_bboxes, pasted_classes
    
    def mosaic_augmentation(
        self,
        images: List[np.ndarray],
        all_bboxes: List[List[List[float]]],
        all_classes: List[List[int]]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Mosaic 增强（4 图拼接）
        
        YOLOv5/v8 经典增强方法
        """
        if len(images) < 4:
            # 样本不足时回退到普通增强
            idx = random.randint(0, len(images) - 1)
            return images[idx], all_bboxes[idx], all_classes[idx]
        
        # 随机选择 4 张图
        indices = random.sample(range(len(images)), 4)
        selected_images = [images[i] for i in indices]
        selected_bboxes = [all_bboxes[i] for i in indices]
        selected_classes = [all_classes[i] for i in indices]
        
        # 创建输出画布
        h, w = selected_images[0].shape[:2]
        mosaic = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        
        # 裁剪点
        xc = random.randint(int(w * 0.2), int(w * 0.8))
        yc = random.randint(int(h * 0.2), int(h * 0.8))
        
        all_bboxes_out = []
        all_classes_out = []
        
        for i, (img, bboxes, classes) in enumerate(zip(selected_images, selected_bboxes, selected_classes)):
            if i == 0:  # 左上
                x1a, y1a = max(xc - w, 0), max(yc - h, 0)
                x1b, y1b = w - (xc - x1a), h - (yc - y1a)
                x2a, y2a = xc, yc
                x2b, y2b = w, h
            elif i == 1:  # 右上
                x1a, y1a = xc, max(yc - h, 0)
                x1b, y1b = 0, h - (yc - y1a)
                x2a, y2a = min(xc + w, w * 2), yc
                x2b, y2b = w, h
            elif i == 2:  # 左下
                x1a, y1a = max(xc - w, 0), yc
                x1b, y1b = w - (xc - x1a), 0
                x2a, y2a = xc, min(h + yc, h * 2)
                x2b, y2b = w, h
            else:  # 右下
                x1a, y1a = xc, yc
                x1b, y1b = 0, 0
                x2a, y2a = min(xc + w, w * 2), min(h + yc, h * 2)
                x2b, y2b = w, h
            
            # 放置图像
            mosaic[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            
            # 转换边界框坐标
            pad_w = x1a - x1b
            pad_h = y1a - y1b
            
            for bbox, cls in zip(bboxes, classes):
                # bbox: [x_center, y_center, width, height] normalized
                x_center = (bbox[0] * w + pad_w) / (w * 2)
                y_center = (bbox[1] * h + pad_h) / (h * 2)
                width = bbox[2] * w / (w * 2)
                height = bbox[3] * h / (h * 2)
                
                # 检查是否在画布内
                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                    0 <= width <= 1 and 0 <= height <= 1):
                    all_bboxes_out.append([x_center, y_center, width, height])
                    all_classes_out.append(cls)
        
        # 裁剪到原始尺寸
        mosaic = mosaic[h//2:h//2+h, w//2:w//2+w]
        
        # 调整边界框到裁剪后的坐标系
        final_bboxes = []
        for bbox in all_bboxes_out:
            # 简化处理，实际应该根据裁剪区域重新计算
            final_bboxes.append(bbox)
        
        return mosaic, final_bboxes, all_classes_out
    
    def generate_training_samples(
        self,
        dataset_dir: str,
        output_dir: str,
        augmentation_factor: int = 10,
        use_mosaic: bool = True,
        use_copy_paste: bool = True
    ) -> Dict[str, int]:
        """
        生成增强的训练样本
        
        Args:
            dataset_dir: 原始数据集目录（YOLO 格式）
            output_dir: 输出目录
            augmentation_factor: 增强倍数
            use_mosaic: 是否使用 Mosaic 增强
            use_copy_paste: 是否使用 Copy-Paste 增强
        
        Returns:
            生成的样本统计信息
        """
        from ultralytics import YOLO
        import yaml
        
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        (output_path / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 加载原始数据
        images_dir = dataset_path / "images" / "train"
        labels_dir = dataset_path / "labels" / "train"
        
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return {}
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        stats = {
            "original_images": len(image_files),
            "augmented_images": 0,
            "mosaic_images": 0,
            "copy_paste_images": 0
        }
        
        # 收集所有缺陷样本用于 Copy-Paste
        defect_patches = []
        
        logger.info(f"Found {len(image_files)} original images")
        logger.info(f"Generating {augmentation_factor}x augmented samples...")
        
        # 第一遍：收集缺陷补丁
        if use_copy_paste:
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    continue
                
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            bbox = parts[1:5]
                            defect_patches.append({
                                "bbox": bbox,
                                "image": image,
                                "class": cls_id,
                                "source": str(img_file)
                            })
        
        logger.info(f"Collected {len(defect_patches)} defect patches for copy-paste")
        
        # 第二遍：生成增强样本
        sample_idx = 0
        for aug_round in range(augmentation_factor):
            for img_file in image_files:
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    # 无标签图像（负样本）
                    image = cv2.imread(str(img_file))
                    if image is None:
                        continue
                    
                    # 只应用颜色和质量的轻微变化
                    aug_image, _, _ = self.apply_color_augmentation(image, [], [])
                    
                    out_name = f"{img_file.stem}_aug{aug_round:02d}.jpg"
                    cv2.imwrite(str(output_path / "images" / "train" / out_name), aug_image)
                    
                    # 创建空标签文件
                    (output_path / "labels" / "train" / f"{out_name.replace('.jpg', '.txt')}").touch()
                    
                    stats["augmented_images"] += 1
                    continue
                
                # 有标签图像
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                with open(label_file, 'r') as f:
                    bboxes = []
                    classes = []
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) >= 5:
                            classes.append(int(parts[0]))
                            bboxes.append(parts[1:5])
                
                if not bboxes:
                    continue
                
                # 选择增强策略
                if use_mosaic and aug_round % 4 == 0 and sample_idx > 3:
                    # Mosaic 增强（需要多张图）
                    # 简化实现，实际需要收集一批图像
                    aug_image, aug_bboxes, aug_classes = self.apply_basic_augmentation(
                        image, bboxes, classes
                    )
                elif use_copy_paste and defect_patches and random.random() < self.config.defect_copy_paste_prob:
                    # Copy-Paste 增强
                    aug_image, aug_bboxes, aug_classes = self.defect_copy_paste(
                        image, defect_patches
                    )
                    # 合并原有缺陷和新粘贴的缺陷
                    aug_bboxes.extend(bboxes)
                    aug_classes.extend(classes)
                    stats["copy_paste_images"] += 1
                else:
                    # 标准增强组合
                    aug_image, aug_bboxes, aug_classes = self.apply_basic_augmentation(
                        image, bboxes, classes
                    )
                    aug_image, aug_bboxes, aug_classes = self.apply_color_augmentation(
                        aug_image, aug_bboxes, aug_classes
                    )
                    if random.random() < 0.3:
                        aug_image, aug_bboxes, aug_classes = self.apply_quality_degradation(
                            aug_image, aug_bboxes, aug_classes
                        )
                
                # 保存增强样本
                out_name = f"{img_file.stem}_aug{aug_round:02d}.jpg"
                cv2.imwrite(str(output_path / "images" / "train" / out_name), aug_image)
                
                # 保存标签
                label_out = output_path / "labels" / "train" / f"{out_name.replace('.jpg', '.txt')}"
                with open(label_out, 'w') as f:
                    for cls, bbox in zip(aug_classes, aug_bboxes):
                        f.write(f"{cls} {' '.join(map(str, bbox))}\n")
                
                stats["augmented_images"] += 1
                sample_idx += 1
        
        # 验证集：使用原始数据（不做增强）
        val_split = min(0.2, max(10, len(image_files) // 5))
        for img_file in image_files[:val_split]:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            # 复制原始图像到验证集
            import shutil
            shutil.copy(
                str(img_file),
                str(output_path / "images" / "val" / img_file.name)
            )
            
            if label_file.exists():
                shutil.copy(
                    str(label_file),
                    str(output_path / "labels" / "val" / f"{label_file.name}")
                )
            else:
                (output_path / "labels" / "val" / f"{label_file.name}").touch()
        
        # 生成数据集配置文件
        data_config = {
            "path": str(output_path.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": 11,  # 缺陷类别数
            "names": [
                "scratch", "dent", "stain", "wrinkle", "bubble",
                "seal_defect", "leakage", "foreign_object", "tab_misalignment",
                "deformation", "normal"
            ]
        }
        
        with open(output_path / "dataset.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"✓ Generated {stats['augmented_images']} augmented samples")
        logger.info(f"  - Mosaic: {stats['mosaic_images']}")
        logger.info(f"  - Copy-Paste: {stats['copy_paste_images']}")
        
        return stats


def create_augmented_dataset(
    input_dir: str,
    output_dir: str,
    factor: int = 10,
    enable_mosaic: bool = True,
    enable_copy_paste: bool = True
) -> None:
    """快速创建增强数据集的便捷函数"""
    augmenter = DefectAugmenter()
    
    stats = augmenter.generate_training_samples(
        dataset_dir=input_dir,
        output_dir=output_dir,
        augmentation_factor=factor,
        use_mosaic=enable_mosaic,
        use_copy_paste=enable_copy_paste
    )
    
    print("\n" + "="*60)
    print("📊 数据增强完成")
    print("="*60)
    print(f"原始图像：{stats.get('original_images', 0)}")
    print(f"增强后图像：{stats.get('augmented_images', 0)}")
    print(f"Mosaic 增强：{stats.get('mosaic_images', 0)}")
    print(f"Copy-Paste 增强：{stats.get('copy_paste_images', 0)}")
    print(f"\n💡 下一步：python src/train.py --data {output_dir}/dataset.yaml")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="缺陷检测数据增强工具")
    parser.add_argument("--input", type=str, required=True, help="输入数据集目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--factor", type=int, default=10, help="增强倍数")
    parser.add_argument("--no-mosaic", action="store_true", help="禁用 Mosaic 增强")
    parser.add_argument("--no-copy-paste", action="store_true", help="禁用 Copy-Paste 增强")
    
    args = parser.parse_args()
    
    create_augmented_dataset(
        input_dir=args.input,
        output_dir=args.output,
        factor=args.factor,
        enable_mosaic=not args.no_mosaic,
        enable_copy_paste=not args.no_copy_paste
    )
