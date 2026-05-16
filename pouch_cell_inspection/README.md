# 🔋 软包电池电芯缺陷检测系统

针对软包电池产线的高精度缺陷检测解决方案，基于 YOLOv8 实现 Recall=100%、Precision>95% 的检测目标。

## 📋 核心特性

- **图像拼接**: 将 18 张多角度图像拼接成完整电芯图 (8000x20000)
- **主动学习**: 智能选择最有价值的样本进行标注
- **数据增强**: Copy-Paste、Mosaic 等专业增强技术
- **高召回率优化**: 专为 Recall=100% 设计的训练和推理策略
- **产线部署**: REST API + Docker，支持 GPU 加速

## 🎯 适用场景

| 特点 | 本方案解决 |
|------|-----------|
| 小样本缺陷 (~200 张) | ✅ 数据增强 + 主动学习 |
| 超大分辨率图像 | ✅ 图像拼接 + 分块处理 |
| 高召回率要求 | ✅ 低阈值检测 + 两级验证 |
| 仅有分类标签 | ✅ 标注优先级推荐 |

## 🚀 快速开始

### 1. 环境准备

```bash
cd pouch_cell_inspection
pip install -r requirements.txt
```

### 2. 一键启动（推荐）

```bash
# 配置数据路径
export RAW_DATA=/path/to/your/cell_images
export OUTPUT_DIR=./output

# 运行快速启动脚本
bash scripts/quick_start.sh
```

### 3. 分步执行

#### 步骤 1: 图像拼接

```bash
python src/stitching/cell_stitcher.py \
  --input /path/to/raw_images \
  --output ./stitched_data \
  --rows 3 --cols 6 \
  --width 20000 --height 8000
```

#### 步骤 2: 创建标注项目

```bash
python src/active_learning/annotation_manager.py \
  --working-dir ./al_project \
  --unlabeled-dir ./stitched_data \
  --batch-size 30
```

查看优先标注列表：`./al_project/priority_queue/cell_defect_phase1_priority.json`

#### 步骤 3: 数据增强

```bash
python src/augmentation/defect_augmenter.py \
  --input ./labeled_data \
  --output ./augmented_data \
  --factor 10
```

#### 步骤 4: 训练模型

```bash
python src/train.py \
  --data ./augmented_data/dataset.yaml \
  --model yolov8l.pt \
  --epochs 300 \
  --batch 8 \
  --name cell_defect_v1
```

## 📁 目录结构

```
pouch_cell_inspection/
├── src/
│   ├── stitching/           # 图像拼接模块
│   │   └── cell_stitcher.py
│   ├── active_learning/     # 主动学习标注
│   │   └── annotation_manager.py
│   ├── augmentation/        # 数据增强
│   │   └── defect_augmenter.py
│   ├── model.py             # 模型管理
│   ├── train.py             # 训练脚本
│   └── inference.py         # 推理接口
├── scripts/
│   └── quick_start.sh       # 快速启动脚本
├── docs/
│   └── IMPLEMENTATION_PLAN.md  # 详细实施方案
├── configs/                 # 配置文件
└── deploy/                  # 部署相关
    └── api/
        └── server.py        # REST API 服务
```

## 🎓 缺陷类别

支持 11 类缺陷检测：

1. **scratch** - 划痕
2. **dent** - 凹坑
3. **stain** - 污渍
4. **wrinkle** - 褶皱
5. **bubble** - 气泡
6. **seal_defect** - 封口不良
7. **leakage** - 漏液
8. **foreign_object** - 异物
9. **tab_misalignment** - 极耳偏移
10. **deformation** - 变形
11. **normal** - 正常区域

## 📊 性能提升路径

| 阶段 | 数据量 | Recall | Precision |
|------|--------|--------|-----------|
| 初始 | 200 张 | 70% | 85% |
| + 拼接 | 200 张 | 75% | 87% |
| + 主动学习 | 400 张 | 85% | 90% |
| + 数据增强 | 4000 张 | 92% | 93% |
| + 第二轮 AL | 600 张 | 96% | 94% |
| **最终** | **~800 张** | **~100%** | **~96%** |

## 🛠️ 产线部署

### REST API

```bash
python deploy/api/server.py \
  --model ./runs/detect/cell_defect_v1/weights/best.pt \
  --port 8000
```

调用示例：

```python
import requests

response = requests.post(
    "http://localhost:8000/detect",
    files={"image": open("cell.jpg", "rb")},
    params={"conf": 0.1, "iou": 0.7}
)
results = response.json()
```

### Docker 部署

```bash
docker-compose up -d
```

## 💡 关键技巧

### 达到 Recall=100%

1. **低置信度阈值**: 推理时设置 `conf=0.1`
2. **高 NMS 阈值**: 设置 `iou=0.7` 避免漏检
3. **两级检测**: 
   ```python
   # 第一级低阈值检测
   detections = model(image, conf=0.1)
   # 第二级分类器验证
   verified = [d for d in detections if classifier.verify(d)]
   ```

### 小样本优化

1. **Copy-Paste 增强**: 缺陷区域复制粘贴
2. **Focal Loss**: 关注难例样本
3. **主动学习**: 优先标注高不确定性样本

## 📖 文档

- [详细实施方案](docs/IMPLEMENTATION_PLAN.md)
- [快速入门指南](docs/QUICKSTART.md)

## ⚙️ 配置示例

```yaml
# configs/recall_optimized.yaml
train:
  conf_threshold: 0.15
  iou_threshold: 0.7
  fl_gamma: 2.0
  fl_alpha: 0.75
  pos_weight: 3.0

inference:
  conf: 0.1
  iou: 0.7
  max_det: 100
  augment: true
```

## 🤝 技术支持

如有问题请查看 [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) 或提交 Issue。

## 📄 License

MIT License
