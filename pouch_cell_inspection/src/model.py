"""
模型定义模块
Model definitions for pouch cell inspection
基于 YOLOv8 的缺陷检测模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available. Install with: pip install ultralytics")


class PouchCellDetector:
    """
    软包电池电芯检测器
    封装 YOLOv8 模型，提供训练和推理接口
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8m.pt',
        num_classes: int = 11,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5
    ):
        """
        初始化检测器
        
        Args:
            model_name: 预训练模型名称 (yolov8n/s/m/l/x.pt)
            num_classes: 类别数量
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self) -> None:
        """加载 YOLOv8 模型"""
        try:
            # 加载预训练模型
            self.model = YOLO(self.model_name)
            
            # 如果类别数不匹配，需要重新初始化分类头
            # 这将在训练时自动处理
            logger.info(f"Loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def train(
        self,
        data_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        save_dir: str = 'runs/train',
        name: str = 'pouch_cell_exp',
        **kwargs
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            data_path: 数据集路径（包含 data.yaml 配置文件）
            epochs: 训练轮数
            batch_size: 批次大小
            image_size: 图像尺寸
            save_dir: 保存目录
            name: 实验名称
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        # 训练参数
        train_args = {
            'data': data_path,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'device': self.device,
            'project': save_dir,
            'name': name,
            'exist_ok': True,
            'verbose': True,
            'save': True,
            'save_txt': True,
            'save_conf': True,
        }
        
        # 添加额外参数
        train_args.update(kwargs)
        
        # 开始训练
        results = self.model.train(**train_args)
        
        logger.info("Training completed!")
        
        return {
            'model_path': results.save_dir,
            'metrics': results.results_dict if hasattr(results, 'results_dict') else {}
        }
    
    def predict(
        self,
        source: Any,
        save: bool = False,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """
        推理预测
        
        Args:
            source: 输入源（图像路径、numpy 数组、或目录）
            save: 是否保存结果
            save_dir: 保存目录
            **kwargs: 其他预测参数
            
        Returns:
            检测结果列表
        """
        predict_args = {
            'source': source,
            'conf': self.confidence_threshold,
            'iou': self.iou_threshold,
            'device': self.device,
            'verbose': False,
            'save': save,
        }
        
        if save and save_dir:
            predict_args['project'] = save_dir
        
        predict_args.update(kwargs)
        
        results = self.model.predict(**predict_args)
        
        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                det = {
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                }
                detections.append(det)
        
        return detections
    
    def export(
        self,
        format: str = 'onnx',
        save_dir: str = 'models/exported',
        **kwargs
    ) -> str:
        """
        导出模型用于部署
        
        Args:
            format: 导出格式 ('onnx', 'tensorrt', 'openvino', etc.)
            save_dir: 保存目录
            **kwargs: 其他导出参数
            
        Returns:
            导出模型的路径
        """
        logger.info(f"Exporting model to {format} format...")
        
        export_args = {
            'format': format,
            'save_dir': save_dir,
        }
        
        export_args.update(kwargs)
        
        export_path = self.model.export(**export_args)
        
        logger.info(f"Model exported to: {export_path}")
        return export_path
    
    def load_weights(self, weights_path: str) -> None:
        """
        加载模型权重
        
        Args:
            weights_path: 权重文件路径
        """
        logger.info(f"Loading weights from: {weights_path}")
        self.model = YOLO(weights_path)
    
    def save_weights(self, save_path: str) -> None:
        """
        保存模型权重
        
        Args:
            save_path: 保存路径
        """
        # YOLOv8 会自动保存训练结果
        # 这里可以手动复制最佳权重
        import shutil
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果是训练后的模型，保存最佳权重
        if hasattr(self.model, 'ckpt'):
            torch.save(self.model.ckpt, save_path)
        else:
            # 否则复制当前权重文件
            shutil.copy(self.model_name, save_path)
        
        logger.info(f"Weights saved to: {save_path}")
    
    def set_thresholds(
        self,
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> None:
        """
        设置检测阈值
        
        Args:
            confidence: 置信度阈值
            iou: NMS IoU 阈值
        """
        if confidence is not None:
            self.confidence_threshold = confidence
        if iou is not None:
            self.iou_threshold = iou
        
        logger.info(
            f"Thresholds updated - confidence: {self.confidence_threshold}, "
            f"iou: {self.iou_threshold}"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'ultralytics_version': self.model.__version__ if hasattr(self.model, '__version__') else 'unknown'
        }


def create_model(
    model_type: str = 'yolov8',
    pretrained: bool = True,
    num_classes: int = 11,
    **kwargs
) -> PouchCellDetector:
    """
    创建检测模型工厂函数
    
    Args:
        model_type: 模型类型 ('yolov8')
        pretrained: 是否使用预训练权重
        num_classes: 类别数量
        **kwargs: 其他参数
        
    Returns:
        检测器实例
    """
    if model_type.lower() == 'yolov8':
        model_name = kwargs.get('model_size', 'yolov8m.pt')
        if not pretrained:
            # 如果没有预训练，创建一个空模型
            logger.info("Creating model without pretrained weights")
        
        return PouchCellDetector(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
