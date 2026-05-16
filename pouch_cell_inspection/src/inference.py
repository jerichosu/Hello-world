"""
推理模块
Inference module for pouch cell inspection
提供高效的实时推理接口，适用于产线部署
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from loguru import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from .utils import (
    preprocess_image,
    postprocess_detections,
    draw_detections,
    nms,
    get_defect_severity
)


class Detector:
    """
    软包电池电芯检测器（推理专用）
    优化用于产线实时检测
    """
    
    # 缺陷类别名称
    CLASS_NAMES = [
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
        model_path: str,
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        image_size: int = 640,
        use_half_precision: bool = True
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型权重路径 (.pt, .onnx, .engine)
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
            confidence_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            image_size: 输入图像尺寸
            use_half_precision: 是否使用半精度推理 (FP16)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.use_half_precision = use_half_precision
        
        # 自动选择设备
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # 加载模型
        self._load_model()
        
        # 性能统计
        self.inference_times = []
        self.total_inferences = 0
        
        logger.info(f"Detector initialized on {self.device}")
    
    def _load_model(self) -> None:
        """加载模型"""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        logger.info(f"Loading model from: {self.model_path}")
        
        # 根据文件类型加载模型
        suffix = model_path.suffix.lower()
        
        if suffix in ['.pt', '.pth']:
            # PyTorch 模型
            self.model = YOLO(self.model_path)
        elif suffix == '.onnx':
            # ONNX 模型
            self.model = YOLO(self.model_path, task='detect')
        elif suffix == '.engine':
            # TensorRT 模型
            self.model = YOLO(self.model_path, task='detect')
        else:
            raise ValueError(f"Unsupported model format: {suffix}")
        
        # 设置模型参数
        if self.device == 'cuda' and self.use_half_precision:
            self.model.fuse()  # 融合层以加速推理
        
        logger.info("Model loaded successfully")
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        return_visualized: bool = False
    ) -> Dict[str, Any]:
        """
        执行单次检测
        
        Args:
            image: 输入图像 (numpy 数组、路径字符串或 Path 对象)
            return_visualized: 是否返回可视化结果
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        # 加载图像
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise IOError(f"Failed to load image: {image}")
        elif not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        original_image = image.copy()
        original_shape = image.shape[:2]
        
        # 执行推理
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            save=False
        )
        
        # 解析结果
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes
            for i in range(len(boxes)):
                det = {
                    'bbox': boxes.xyxy[i].cpu().numpy().tolist(),
                    'confidence': float(boxes.conf[i].cpu().numpy()),
                    'class_id': int(boxes.cls[i].cpu().numpy()),
                    'class_name': self.CLASS_NAMES[int(boxes.cls[i].cpu().numpy())] 
                                  if int(boxes.cls[i].cpu().numpy()) < len(self.CLASS_NAMES) 
                                  else 'unknown',
                    'severity': get_defect_severity(
                        self.CLASS_NAMES[int(boxes.cls[i].cpu().numpy())]
                        if int(boxes.cls[i].cpu().numpy()) < len(self.CLASS_NAMES)
                        else 'unknown'
                    )
                }
                detections.append(det)
        
        # 应用 NMS
        detections = nms(detections, self.iou_threshold)
        
        # 计算推理时间
        inference_time = (time.time() - start_time) * 1000  # 毫秒
        self.inference_times.append(inference_time)
        self.total_inferences += 1
        
        # 保持最近的 100 次推理时间
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        # 判断是否合格
        is_passed = len(detections) == 0
        
        # 构建结果
        output = {
            'is_passed': is_passed,
            'detection_count': len(detections),
            'detections': detections,
            'inference_time_ms': inference_time,
            'original_shape': original_shape,
            'timestamp': time.time()
        }
        
        if return_visualized:
            vis_image = draw_detections(
                original_image,
                detections,
                class_names=self.CLASS_NAMES
            )
            output['visualized_image'] = vis_image
        
        return output
    
    def detect_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        batch_size: int = 8
    ) -> List[Dict[str, Any]]:
        """
        批量检测
        
        Args:
            images: 图像列表
            batch_size: 批次大小
            
        Returns:
            检测结果列表
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for image in batch:
                result = self.detect(image)
                results.append(result)
        
        return results
    
    def detect_from_camera(
        self,
        camera_id: int = 0,
        display: bool = True,
        save_dir: Optional[str] = None
    ) -> None:
        """
        从摄像头实时检测
        
        Args:
            camera_id: 摄像头 ID
            display: 是否显示结果
            save_dir: 保存目录（可选）
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open camera {camera_id}")
        
        logger.info(f"Camera opened. Press 'q' to quit.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 执行检测
            result = self.detect(frame, return_visualized=True)
            
            # 绘制结果
            vis_frame = result.get('visualized_image', frame)
            
            # 添加状态信息
            status_color = (0, 255, 0) if result['is_passed'] else (0, 0, 255)
            status_text = "PASS" if result['is_passed'] else "FAIL"
            
            cv2.putText(
                vis_frame,
                f"Status: {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                2
            )
            
            cv2.putText(
                vis_frame,
                f"Time: {result['inference_time_ms']:.1f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # 显示
            if display:
                cv2.imshow('Pouch Cell Inspection', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存不合格品
            if save_dir and not result['is_passed']:
                save_path = Path(save_dir) / f"fail_{frame_count:06d}.jpg"
                cv2.imwrite(str(save_path), vis_frame)
                logger.warning(f"Saved failed sample: {save_path}")
            
            frame_count += 1
        
        cap.release()
        if display:
            cv2.destroyAllWindows()
        
        logger.info(f"Total frames processed: {frame_count}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取推理统计信息
        
        Returns:
            统计信息字典
        """
        if not self.inference_times:
            return {
                'total_inferences': 0,
                'avg_time_ms': 0,
                'min_time_ms': 0,
                'max_time_ms': 0,
                'fps': 0
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        
        return {
            'total_inferences': self.total_inferences,
            'avg_time_ms': avg_time,
            'min_time_ms': min(self.inference_times),
            'max_time_ms': max(self.inference_times),
            'fps': 1000 / avg_time if avg_time > 0 else 0
        }
    
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
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.inference_times = []
        self.total_inferences = 0
        logger.info("Statistics reset")


def create_detector(
    model_path: str,
    **kwargs
) -> Detector:
    """
    创建检测器工厂函数
    
    Args:
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        Detector 实例
    """
    return Detector(model_path, **kwargs)
