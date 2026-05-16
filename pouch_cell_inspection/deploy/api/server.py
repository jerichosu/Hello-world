#!/usr/bin/env python3
"""
API 服务
FastAPI-based REST API for pouch cell inspection
用于产线集成的 HTTP API 服务
"""

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import uvicorn

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference import Detector
from src.utils import setup_logger
from loguru import logger

# 创建 FastAPI 应用
app = FastAPI(
    title="Pouch Cell Inspection API",
    description="软包电池电芯缺陷检测 API 服务",
    version="1.0.0"
)

# 全局检测器实例
detector: Optional[Detector] = None


@app.on_event("startup")
async def startup_event():
    """启动时初始化检测器"""
    global detector
    
    model_path = getattr(app.state, 'model_path', 'models/best.pt')
    
    if Path(model_path).exists():
        detector = Detector(
            model_path=model_path,
            confidence_threshold=getattr(app.state, 'confidence_threshold', 0.5),
            iou_threshold=getattr(app.state, 'iou_threshold', 0.5)
        )
        logger.info(f"Detector initialized with model: {model_path}")
    else:
        logger.warning(f"Model not found: {model_path}. Detection will be unavailable.")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Pouch Cell Inspection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect_image": "/detect/image",
            "detect_batch": "/detect/batch",
            "statistics": "/statistics"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    if detector is None:
        return {"status": "unavailable", "message": "Detector not initialized"}
    
    return {
        "status": "healthy",
        "device": detector.device,
        "model_path": detector.model_path
    }


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    return_visualized: bool = False
):
    """
    检测单张图像
    
    Args:
        file: 上传的图像文件
        return_visualized: 是否返回可视化结果
        
    Returns:
        检测结果 JSON
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    try:
        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image.convert('RGB'))
        
        # 执行检测
        result = detector.detect(image_np, return_visualized=return_visualized)
        
        # 如果需要返回可视化图像
        if return_visualized and 'visualized_image' in result:
            vis_image = result.pop('visualized_image')
            
            # 将可视化图像转换为 base64
            pil_image = Image.fromarray(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            buffer.seek(0)
            
            result['visualized_image_base64'] = base64.b64encode(buffer.getvalue()).decode()
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    批量检测多张图像
    
    Args:
        files: 上传的图像文件列表
        
    Returns:
        检测结果列表
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    try:
        results = []
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_np = np.array(image.convert('RGB'))
            
            result = detector.detect(image_np)
            result['filename'] = file.filename
            results.append(result)
        
        return JSONResponse(content={
            "total": len(results),
            "passed": sum(1 for r in results if r['is_passed']),
            "failed": sum(1 for r in results if not r['is_passed']),
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """获取推理统计信息"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    stats = detector.get_statistics()
    return JSONResponse(content=stats)


@app.post("/thresholds")
async def set_thresholds(
    confidence: Optional[float] = None,
    iou: Optional[float] = None
):
    """
    设置检测阈值
    
    Args:
        confidence: 置信度阈值
        iou: NMS IoU 阈值
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not available")
    
    detector.set_thresholds(confidence=confidence, iou=iou)
    
    return {
        "message": "Thresholds updated",
        "confidence_threshold": detector.confidence_threshold,
        "iou_threshold": detector.iou_threshold
    }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Pouch Cell Inspection API Server')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Path to model weights')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IoU threshold')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker processes')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logger(log_file='api_server.log')
    
    logger.info("=" * 50)
    logger.info("Pouch Cell Inspection API Server")
    logger.info("=" * 50)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Model: {args.model}")
    
    # 保存配置到 app.state
    app.state.model_path = args.model
    app.state.confidence_threshold = args.confidence
    app.state.iou_threshold = args.iou
    
    # 启动服务器
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == '__main__':
    # 需要导入 cv2 用于图像处理
    import cv2
    main()
