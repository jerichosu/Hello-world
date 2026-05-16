#!/usr/bin/env python3
"""
测试模块
Test module for pouch cell inspection
"""

import unittest
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUtils(unittest.TestCase):
    """工具函数测试"""
    
    def test_preprocess_image(self):
        """测试图像预处理"""
        from src.utils import preprocess_image
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # 测试预处理
        processed, meta = preprocess_image(test_image, target_size=(640, 640))
        
        self.assertEqual(processed.shape[0], 640)
        self.assertEqual(processed.shape[1], 640)
        self.assertIn('original_shape', meta)
        self.assertIn('scale', meta)
    
    def test_nms(self):
        """测试非极大值抑制"""
        from src.utils import nms
        
        # 创建测试检测结果
        detections = [
            {'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class_id': 1},
            {'bbox': [12, 12, 52, 52], 'confidence': 0.8, 'class_id': 1},
            {'bbox': [100, 100, 150, 150], 'confidence': 0.7, 'class_id': 2},
        ]
        
        # 应用 NMS
        result = nms(detections, iou_threshold=0.5)
        
        # 应该保留 2 个检测结果（重叠的只保留一个）
        self.assertEqual(len(result), 2)
    
    def test_iou_calculation(self):
        """测试 IoU 计算"""
        from src.utils import calculate_iou
        
        # 相同框
        box1 = [0, 0, 10, 10]
        box2 = [0, 0, 10, 10]
        self.assertAlmostEqual(calculate_iou(box1, box2), 1.0)
        
        # 不重叠框
        box3 = [0, 0, 10, 10]
        box4 = [20, 20, 30, 30]
        self.assertAlmostEqual(calculate_iou(box3, box4), 0.0)


class TestDataset(unittest.TestCase):
    """数据集测试"""
    
    def test_class_names(self):
        """测试类别名称"""
        from src.dataset import PouchCellDataset
        
        class_names = PouchCellDataset.get_class_names()
        num_classes = PouchCellDataset.get_num_classes()
        
        self.assertEqual(len(class_names), num_classes)
        self.assertIn('scratch', class_names)
        self.assertIn('dent', class_names)


class TestDetector(unittest.TestCase):
    """检测器测试"""
    
    def test_detector_initialization(self):
        """测试检测器初始化（需要模型文件）"""
        # 这是一个占位测试，实际需要模型文件才能运行
        pass


if __name__ == '__main__':
    unittest.main()
