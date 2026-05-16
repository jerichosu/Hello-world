"""
数据集处理模块
Dataset handling for pouch cell inspection
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from loguru import logger


class PouchCellDataset(Dataset):
    """
    软包电池电芯数据集类
    支持 YOLO 格式和 COCO 格式的数据集
    """
    
    # 缺陷类别定义
    CLASSES = [
        'normal',           # 0: 正常
        'scratch',          # 1: 划痕
        'dent',             # 2: 凹坑
        'stain',            # 3: 污渍
        'wrinkle',          # 4: 褶皱
        'bubble',           # 5: 气泡
        'poor_sealing',     # 6: 封口不良
        'leakage',          # 7: 漏液
        'foreign_object',   # 8: 异物
        'tab_misalignment', # 9: 极耳偏移
        'deformation'       # 10: 变形
    ]
    
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = False,
        format: str = 'yolo'  # 'yolo' or 'coco'
    ):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录
            image_size: 图像尺寸 (width, height)
            augment: 是否使用数据增强
            format: 标注格式 ('yolo' or 'coco')
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.augment = augment
        self.format = format
        
        self.images = []
        self.annotations = []
        
        self._load_dataset()
        self._build_transforms()
        
        logger.info(f"Dataset loaded: {len(self.images)} images")
    
    def _load_dataset(self) -> None:
        """加载数据集"""
        if self.format == 'yolo':
            self._load_yolo_format()
        elif self.format == 'coco':
            self._load_coco_format()
        else:
            raise ValueError(f"Unsupported format: {self.format}")
    
    def _load_yolo_format(self) -> None:
        """加载 YOLO 格式数据集"""
        images_dir = self.root_dir / 'images'
        labels_dir = self.root_dir / 'labels'
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f'*{ext}'))
        
        for img_path in sorted(image_files):
            label_path = labels_dir / f'{img_path.stem}.txt'
            
            if not label_path.exists():
                # 没有标注文件的视为正常样本
                annotations = []
            else:
                annotations = self._parse_yolo_label(label_path)
            
            self.images.append(str(img_path))
            self.annotations.append(annotations)
    
    def _parse_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        解析 YOLO 格式的标注文件
        
        YOLO 格式：class_id x_center y_center width height
        坐标已归一化到 [0, 1]
        """
        annotations = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为中心点 + 宽高格式到左上角 + 右下角格式
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max],  # 归一化坐标
                    'category': self.CLASSES[class_id] if class_id < len(self.CLASSES) else 'unknown'
                })
        
        return annotations
    
    def _load_coco_format(self) -> None:
        """加载 COCO 格式数据集"""
        annotation_file = self.root_dir / 'annotations.json'
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # 构建图像 ID 到路径的映射
        image_dict = {img['id']: img for img in coco_data.get('images', [])}
        
        # 构建图像 ID 到标注的映射
        annotations_dict = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in annotations_dict:
                annotations_dict[img_id] = []
            
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # 获取图像尺寸用于归一化
            img_info = image_dict[img_id]
            img_w, img_h = img_info['width'], img_info['height']
            
            annotations_dict[img_id].append({
                'class_id': ann['category_id'],
                'bbox': [x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h],
                'category': ann.get('category_name', 'unknown')
            })
        
        # 填充数据和标注
        for img_id, img_info in sorted(image_dict.items()):
            self.images.append(str(self.root_dir / img_info['file_name']))
            self.annotations.append(annotations_dict.get(img_id, []))
    
    def _build_transforms(self) -> None:
        """构建数据增强变换"""
        if self.augment:
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    height=self.image_size[1],
                    width=self.image_size[0],
                    scale=(0.8, 1.0),
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=15,
                    p=0.3
                ),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.3
                ),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.CLAHE(clip_limit=2.0, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_ids']))
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size[1], width=self.image_size[0]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_ids']))
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本
        
        Returns:
            包含图像、标注和元数据的字典
        """
        img_path = self.images[idx]
        annotations = self.annotations[idx]
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            raise IOError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 准备标注
        if annotations:
            bboxes = [ann['bbox'] for ann in annotations]
            class_ids = [ann['class_id'] for ann in annotations]
        else:
            bboxes = []
            class_ids = []
        
        # 应用变换
        try:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_ids=class_ids
            )
        except Exception as e:
            logger.warning(f"Transform failed for {img_path}: {e}")
            # 回退到无 bbox 的变换
            transformed = self.transform(image=image)
            transformed['bboxes'] = []
            transformed['class_ids'] = []
        
        # 转换图像为 tensor
        image_tensor = transformed['image']
        image_tensor = np.transpose(image_tensor, (2, 0, 1))  # HWC -> CHW
        image_tensor = image_tensor.astype(np.float32)
        
        # 准备目标标注
        target = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': []
        }
        
        for i, bbox in enumerate(transformed.get('bboxes', [])):
            # Albumentations bbox 格式：[x_min, y_min, x_max, y_max] (归一化)
            x_min, y_min, x_max, y_max = bbox
            
            # 转换为像素坐标
            x_min *= self.image_size[0]
            y_min *= self.image_size[1]
            x_max *= self.image_size[0]
            y_max *= self.image_size[1]
            
            target['boxes'].append([x_min, y_min, x_max, y_max])
            target['labels'].append(transformed['class_ids'][i])
            target['area'].append((x_max - x_min) * (y_max - y_min))
            target['iscrowd'].append(0)
        
        # 转换为 numpy 数组
        if target['boxes']:
            target['boxes'] = np.array(target['boxes'], dtype=np.float32)
            target['labels'] = np.array(target['labels'], dtype=np.int64)
            target['area'] = np.array(target['area'], dtype=np.float32)
            target['iscrowd'] = np.array(target['iscrowd'], dtype=np.int64)
        else:
            target['boxes'] = np.zeros((0, 4), dtype=np.float32)
            target['labels'] = np.zeros((0,), dtype=np.int64)
            target['area'] = np.zeros((0,), dtype=np.float32)
            target['iscrowd'] = np.zeros((0,), dtype=np.int64)
        
        return {
            'image': image_tensor,
            'target': target,
            'image_path': img_path,
            'original_shape': image.shape[:2]
        }
    
    @staticmethod
    def get_class_names() -> List[str]:
        """获取类别名称列表"""
        return PouchCellDataset.CLASSES
    
    @staticmethod
    def get_num_classes() -> int:
        """获取类别数量"""
        return len(PouchCellDataset.CLASSES)


def create_dataloader(
    dataset: PouchCellDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        num_workers: 工作线程数
        shuffle: 是否打乱数据
        
    Returns:
        DataLoader 实例
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=lambda batch: tuple(zip(*batch)),
        pin_memory=True
    )
    
    logger.info(f"DataLoader created: batch_size={batch_size}, num_workers={num_workers}")
    return dataloader


def split_dataset(
    root_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> None:
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        root_dir: 数据集根目录（包含所有数据）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    import shutil
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    root_path = Path(root_dir)
    images_dir = root_path / 'images'
    labels_dir = root_path / 'labels'
    
    # 获取所有图像文件
    image_files = list(images_dir.glob('*.jpg')) + \
                  list(images_dir.glob('*.png')) + \
                  list(images_dir.glob('*.jpeg'))
    
    # 打乱并划分
    np.random.seed(seed)
    np.random.shuffle(image_files)
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # 创建目录结构
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        split_images_dir = root_path / split_name / 'images'
        split_labels_dir = root_path / split_name / 'labels'
        
        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in files:
            # 复制图像
            shutil.copy(img_path, split_images_dir / img_path.name)
            
            # 复制标注文件（如果存在）
            label_path = labels_dir / f'{img_path.stem}.txt'
            if label_path.exists():
                shutil.copy(label_path, split_labels_dir / f'{img_path.stem}.txt')
        
        logger.info(f"{split_name}: {len(files)} images")
    
    logger.info("Dataset split completed")
