"""
工具函数模块
Utility functions for pouch cell inspection
"""

import os
import yaml
import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from loguru import logger


def setup_logger(
    log_file: str = "inspection.log",
    level: str = "INFO",
    rotation: str = "500 MB"
) -> None:
    """
    设置日志记录器
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        rotation: 日志轮转大小
    """
    logger.remove()
    
    # 控制台输出
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # 文件输出
    logger.add(
        log_file,
        level=level,
        rotation=rotation,
        retention="10 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    
    logger.info("Logger initialized successfully")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    logger.info(f"Configuration saved to {save_path}")


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    图像预处理
    
    Args:
        image: 输入图像 (BGR 格式)
        target_size: 目标尺寸 (width, height)
        normalize: 是否归一化
        
    Returns:
        预处理后的图像和元数据
    """
    original_shape = image.shape[:2]
    
    # 计算缩放比例
    scale = min(target_size[0] / original_shape[1], target_size[1] / original_shape[0])
    new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))
    
    # 缩放图像
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    
    # 创建画布并粘贴图像
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y_offset = (target_size[1] - new_size[1]) // 2
    x_offset = (target_size[0] - new_size[0]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    
    # 归一化
    if normalize:
        canvas = canvas.astype(np.float32) / 255.0
    
    meta = {
        'original_shape': original_shape,
        'new_size': new_size,
        'scale': scale,
        'offset': (x_offset, y_offset),
        'target_size': target_size
    }
    
    return canvas, meta


def postprocess_detections(
    detections: List[Dict],
    meta: Dict[str, Any],
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    后处理检测结果，将坐标映射回原图
    
    Args:
        detections: 检测结果列表
        meta: 预处理元数据
        confidence_threshold: 置信度阈值
        
    Returns:
        映射后的检测结果
    """
    scale = meta['scale']
    offset = meta['offset']
    
    processed_detections = []
    
    for det in detections:
        if det.get('confidence', 0) < confidence_threshold:
            continue
        
        # 映射边界框坐标回原图
        bbox = det['bbox']
        x1 = (bbox[0] - offset[0]) / scale
        y1 = (bbox[1] - offset[1]) / scale
        x2 = (bbox[2] - offset[0]) / scale
        y2 = (bbox[3] - offset[1]) / scale
        
        # 限制在原图范围内
        x1 = max(0, min(x1, meta['original_shape'][1]))
        y1 = max(0, min(y1, meta['original_shape'][0]))
        x2 = max(0, min(x2, meta['original_shape'][1]))
        y2 = max(0, min(y2, meta['original_shape'][0]))
        
        processed_det = det.copy()
        processed_det['bbox'] = [x1, y1, x2, y2]
        processed_det['confidence'] = float(det.get('confidence', 0))
        processed_detections.append(processed_det)
    
    return processed_detections


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Optional[List[str]] = None,
    font_scale: float = 0.5,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制检测结果
    
    Args:
        image: 原始图像
        detections: 检测结果列表
        class_names: 类别名称列表
        font_scale: 字体大小
        thickness: 线条粗细
        
    Returns:
        绘制后的图像
    """
    result = image.copy()
    
    # 颜色映射（不同缺陷类型不同颜色）
    colors = [
        (0, 255, 0),    # 绿色 - 正常
        (0, 0, 255),    # 红色 - 严重缺陷
        (0, 165, 255),  # 橙色 - 中等缺陷
        (255, 255, 0),  # 青色 - 轻微缺陷
        (255, 0, 255),  # 紫色 - 其他
    ]
    
    for i, det in enumerate(detections):
        bbox = list(map(int, det['bbox']))
        x1, y1, x2, y2 = bbox
        
        # 获取颜色
        class_id = det.get('class_id', 0)
        color = colors[class_id % len(colors)]
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签
        label_parts = []
        if class_names and class_id < len(class_names):
            label_parts.append(class_names[class_id])
        
        if 'confidence' in det:
            label_parts.append(f"{det['confidence']:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # 计算文本大小
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # 绘制文本背景
            cv2.rectangle(
                result,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                result,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    return result


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    计算两个边界框的 IoU (Intersection over Union)
    
    Args:
        box1: 第一个边界框 [x1, y1, x2, y2]
        box2: 第二个边界框 [x1, y1, x2, y2]
        
    Returns:
        IoU 值
    """
    # 计算交集
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算并集
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def nms(
    detections: List[Dict],
    iou_threshold: float = 0.5
) -> List[Dict]:
    """
    非极大值抑制 (Non-Maximum Suppression)
    
    Args:
        detections: 检测结果列表
        iou_threshold: IoU 阈值
        
    Returns:
        NMS 后的检测结果
    """
    if not detections:
        return []
    
    # 按置信度排序
    detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        # 移除与最佳检测结果 IoU 过高的结果
        detections = [
            det for det in detections
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep


def get_defect_severity(defect_type: str) -> str:
    """
    获取缺陷严重程度等级
    
    Args:
        defect_type: 缺陷类型
        
    Returns:
        严重程度：critical, major, minor
    """
    severity_map = {
        # 严重缺陷
        'leakage': 'critical',
        'poor_sealing': 'critical',
        'foreign_object': 'critical',
        
        # 主要缺陷
        'dent': 'major',
        'wrinkle': 'major',
        'bubble': 'major',
        'tab_misalignment': 'major',
        
        # 次要缺陷
        'scratch': 'minor',
        'stain': 'minor',
        'deformation': 'minor'
    }
    
    return severity_map.get(defect_type.lower(), 'minor')


def create_output_directory(output_dir: str) -> Path:
    """
    创建输出目录
    
    Args:
        output_dir: 输出目录路径
        
    Returns:
        输出目录的 Path 对象
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_path / 'passed').mkdir(exist_ok=True)
    (output_path / 'failed').mkdir(exist_ok=True)
    (output_path / 'results').mkdir(exist_ok=True)
    
    return output_path
