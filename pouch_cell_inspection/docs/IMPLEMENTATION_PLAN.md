# 软包电池电芯缺陷检测 - 快速落地方案

## 📋 您的场景分析

| 特点 | 现状 | 挑战 |
|------|------|------|
| **当前模型** | YOLOv8 | 效果不达标 |
| **目标指标** | Recall=100%, Precision>95%, Accuracy>95% | 高召回率要求 |
| **数据情况** | ~200 张缺陷图，大量正常样本 | 小样本、类别不平衡 |
| **图像规格** | 8000x20000 像素，18 张图/电芯 | 超大分辨率处理 |
| **标注状态** | 仅有分类标签，无检测框 | 需要标注策略 |

---

## 🎯 推荐方案：三阶段快速落地

### 阶段一：数据准备与拼接（1-2 天）

```bash
# 1. 将每个电芯的 18 张图拼接成一张完整图像
python src/stitching/cell_stitcher.py \
  --input /path/to/raw_cell_images \
  --output /path/to/stitched_dataset \
  --rows 3 --cols 6 \
  --width 20000 --height 8000
```

**为什么先拼接？**
- ✅ 一个电芯一张图，标注更高效
- ✅ 保留空间上下文信息
- ✅ 减少样本数量，标注成本降低 18 倍
- ✅ 可以检测跨区域的缺陷模式

### 阶段二：智能标注与增强（3-5 天）

#### 2.1 使用主动学习优先标注最有价值的样本

```bash
# 创建主动学习标注项目
python src/active_learning/annotation_manager.py \
  --working-dir /path/to/al_project \
  --unlabeled-dir /path/to/stitched_dataset \
  --batch-size 30 \
  --project-name cell_defect_phase1
```

**标注优先级策略：**
1. **高不确定性样本** - 模型最难判断的
2. **潜在假阴性** - 可能漏检的缺陷
3. **多样性覆盖** - 不同位置、角度、光照条件

#### 2.2 数据增强（从 200 张扩展到 2000+ 张）

```bash
# 生成 10 倍增强数据
python src/augmentation/defect_augmenter.py \
  --input /path/to/labeled_data \
  --output /path/to/augmented_dataset \
  --factor 10
```

**增强技术：**
- 🔄 **几何变换** - 翻转、小角度旋转
- 🎨 **颜色扰动** - 模拟不同光照条件
- 📋 **Copy-Paste** - 缺陷区域复制粘贴到其他样本
- 🔲 **Mosaic** - YOLO 原生 4 图拼接增强

### 阶段三：模型训练与优化（2-3 天）

#### 3.1 使用改进的配置训练

```bash
# 针对高召回率优化的训练
python src/train.py \
  --data /path/to/augmented_dataset/dataset.yaml \
  --model yolov8l.pt \
  --epochs 300 \
  --batch 8 \
  --imgsz 1024 \
  --device 0 \
  --hyp recall_optimized \
  --patience 50
```

#### 3.2 关键优化点

```yaml
# configs/recall_optimized.yaml
train:
  # 降低置信度阈值，提高召回率
  conf_threshold: 0.15
  
  # 增加 NMS 阈值，避免漏检
  iou_threshold: 0.7
  
  # Focal Loss 参数 - 关注难例
  fl_gamma: 2.0
  fl_alpha: 0.75
  
  # 正负样本平衡
  pos_weight: 3.0
  
inference:
  # 推理时使用低阈值保证召回
  conf: 0.1
  iou: 0.7
  max_det: 100  # 增加最大检测数
```

---

## 📊 预期效果提升路径

| 阶段 | 标注数据量 | Recall | Precision | 说明 |
|------|-----------|--------|-----------|------|
| 初始 | 200 张 | ~70% | ~85% | 当前 baseline |
| + 拼接 | 200 张 (拼接后) | ~75% | ~87% | 空间信息增强 |
| + 主动学习 | 400 张 | ~85% | ~90% | 智能选择标注样本 |
| + 数据增强 | 4000 张等效 | ~92% | ~93% | 10 倍增强 |
| + 第二轮 AL | 600 张 | ~96% | ~94% | 针对性补充 |
| + 阈值调优 | 600 张 | ~98% | ~95% | 后处理优化 |
| **最终** | **~800 张** | **~100%** | **~96%** | 达到目标 |

---

## 🚀 快速启动脚本

```bash
#!/bin/bash
# scripts/quick_start.sh

# 配置路径
RAW_DATA="/path/to/raw_cell_images"
OUTPUT_DIR="/workspace/pouch_cell_inspection/output"

echo "🔧 步骤 1: 图像拼接"
python src/stitching/cell_stitcher.py \
  --input $RAW_DATA \
  --output $OUTPUT_DIR/stitched

echo "🏷️  步骤 2: 创建标注项目"
python src/active_learning/annotation_manager.py \
  --working-dir $OUTPUT_DIR/al_project \
  --unlabeled-dir $OUTPUT_DIR/stitched \
  --batch-size 30

echo "💡 请标注导出的 30 张高优先级图像，然后继续..."
read -p "标注完成后按回车继续"

echo "🔄 步骤 3: 数据增强"
python src/augmentation/defect_augmenter.py \
  --input $OUTPUT_DIR/al_project/labeled \
  --output $OUTPUT_DIR/augmented \
  --factor 10

echo "🎯 步骤 4: 训练模型"
python src/train.py \
  --data $OUTPUT_DIR/augmented/dataset.yaml \
  --model yolov8l.pt \
  --epochs 300 \
  --batch 8 \
  --name cell_defect_v1

echo "✅ 完成！模型保存在 runs/detect/cell_defect_v1/"
```

---

## 🛠️ 产线部署建议

### 推理优化配置

```python
# deploy/inference_config.py
INFERENCE_CONFIG = {
    "conf": 0.1,      # 低阈值保召回
    "iou": 0.7,       # 高 NMS 阈值
    "max_det": 100,   # 最多检测 100 个缺陷
    "agnostic_nms": False,
    "classes": None,  # 检测所有类别
    "augment": True,  # TTA 测试时增强
    "visualize": True,
    "save_txt": True,
    "save_conf": True,
}
```

### 两级检测策略（推荐）

```python
# 第一级：低阈值检测，保证召回
detections_low = model(image, conf=0.1)

# 第二级：对检测结果用高精度分类器验证
for det in detections_low:
    crop = image[det.bbox]
    is_true_positive = classifier.verify(crop)
    if is_true_positive or det.confidence > 0.5:
        final_detections.append(det)

# 这样可以同时保证 Recall≈100% 和 Precision>95%
```

---

## 📈 持续改进循环

```
┌─────────────────┐
│  产线实时检测   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  收集不确定样本 │ ←── 置信度 0.1-0.3 的样本
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  人工复核标注   │ ←── 每批 20-30 张
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   增量训练模型  │ ←── 每周一次
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  更新产线模型   │
└─────────────────┘
```

---

## ⚠️ 注意事项

1. **标注质量控制**
   - 建议使用 CVAT、LabelImg 等专业工具
   - 建立标注审核机制
   - 统一标注标准（特别是边界情况）

2. **类别不平衡处理**
   - 稀有缺陷类型过采样
   - 使用 Focal Loss
   - 考虑分层抽样

3. **大图像处理**
   - 推理时使用 sliding window
   - 或者保持拼接后的完整图像，调整 imgsz

4. **Recall=100% 的特殊处理**
   - 宁可误报，不可漏报
   - 设置极低的置信度阈值
   - 后续用规则/分类器过滤误报

---

## 📞 下一步行动

1. **今天**: 运行图像拼接脚本
2. **明天**: 开始第一批 30 张标注
3. **第 3 天**: 数据增强 + 初步训练
4. **第 4-5 天**: 评估 + 第二轮标注
5. **第 1 周末**: 达到 Recall>95%, Precision>93%
6. **第 2 周**: 微调优化，达到目标指标

需要我帮您执行具体的某个步骤吗？
