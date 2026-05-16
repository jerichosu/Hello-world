#!/usr/bin/env python3
"""
训练脚本
Training script for pouch cell inspection
"""

import argparse
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PouchCellDetector, create_model
from src.utils import setup_logger, load_config
from loguru import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train pouch cell defect detection model')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    
    # 数据集
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset (overrides config)')
    
    # 模型
    parser.add_argument('--model', type=str, default='yolov8m.pt',
                        help='Pretrained model name or path')
    parser.add_argument('--num-classes', type=int, default=11,
                        help='Number of classes')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--image-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    # 保存
    parser.add_argument('--save-dir', type=str, default='runs/train',
                        help='Directory to save results')
    parser.add_argument('--name', type=str, default='pouch_cell_exp',
                        help='Experiment name')
    
    # 其他
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    """主训练函数"""
    args = parse_args()
    
    # 设置日志
    setup_logger(log_file=f'{args.save_dir}/{args.name}/training.log')
    
    logger.info("=" * 50)
    logger.info("Pouch Cell Defect Detection - Training")
    logger.info("=" * 50)
    
    # 加载配置（如果有）
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    
    # 合并配置和命令行参数
    data_path = args.data or config.get('data', 'data/pouch_cell')
    epochs = args.epochs or config.get('epochs', 100)
    batch_size = args.batch_size or config.get('batch_size', 16)
    image_size = args.image_size or config.get('image_size', 640)
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Image size: {image_size}")
    
    # 创建模型
    try:
        detector = create_model(
            model_type='yolov8',
            model_name=args.model,
            num_classes=args.num_classes,
            device=args.device
        )
        
        logger.info("Model created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise
    
    # 开始训练
    try:
        results = detector.train(
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            save_dir=args.save_dir,
            name=args.name,
            workers=args.workers,
            resume=args.resume if args.resume else False
        )
        
        logger.info("=" * 50)
        logger.info("Training completed!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info("=" * 50)
        
        # 导出模型用于部署
        logger.info("Exporting model to ONNX format...")
        try:
            onnx_path = detector.export(
                format='onnx',
                save_dir=f"{results['model_path']}/exported"
            )
            logger.info(f"ONNX model exported to: {onnx_path}")
        except Exception as e:
            logger.warning(f"Failed to export model: {e}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
