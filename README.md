# 软包电池电芯缺陷检测系统

## 项目概述
本项目是一个基于深度学习的软包电池电芯缺陷检测算法，可部署到产线上进行实时缺陷检测。

## 主要功能
- 表面缺陷检测（划痕、凹坑、污渍等）
- 封装缺陷检测（封口不良、褶皱等）
- 尺寸测量与偏差检测
- 极耳位置检测
- 实时推理与产线集成

## 目录结构
```
pouch_cell_inspection/
├── data/                  # 数据目录
├── models/                # 模型文件
├── configs/               # 配置文件
├── src/                   # 源代码
│   ├── __init__.py
│   ├── dataset.py         # 数据集处理
│   ├── model.py           # 模型定义
│   ├── train.py           # 训练脚本
│   ├── inference.py       # 推理脚本
│   └── utils.py           # 工具函数
├── requirements.txt       # 依赖包
├── deploy/                # 部署相关
│   ├── docker/            # Docker配置
│   └── api/               # API服务
└── tests/                 # 测试文件
```

## 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
python src/train.py --config configs/default.yaml
```

### 推理检测
```bash
python src/inference.py --model_path models/best_model.pth --image_path data/test/sample.jpg
```

### 启动API服务
```bash
python deploy/api/server.py --port 8000
```

## 支持的缺陷类型
1. **表面缺陷**
   - 划痕 (Scratches)
   - 凹坑 (Dents)
   - 污渍 (Stains)
   - 异物 (Foreign objects)

2. **封装缺陷**
   - 封口不良 (Poor sealing)
   - 褶皱 (Wrinkles)
   - 气泡 (Bubbles)
   - 漏液 (Leakage)

3. **结构缺陷**
   - 极耳偏移 (Tab misalignment)
   - 尺寸超差 (Dimension out of tolerance)
   - 变形 (Deformation)

## 性能指标
- 检测精度：>99%
- 推理速度：<50ms/张 (GPU)
- 支持分辨率：最高 4096x4096

## 许可证
MIT License
