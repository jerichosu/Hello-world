#!/bin/bash
# 软包电池电芯缺陷检测 - 快速启动脚本
# 使用方法：bash scripts/quick_start.sh

set -e

echo "=========================================="
echo "🔋 软包电池电芯缺陷检测 - 快速启动"
echo "=========================================="

# 配置路径（请根据实际情况修改）
RAW_DATA="${RAW_DATA:-/path/to/raw_cell_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./output}"
BATCH_SIZE="${BATCH_SIZE:-30}"
AUG_FACTOR="${AUG_FACTOR:-10}"

echo ""
echo "📁 配置信息:"
echo "  原始数据目录：$RAW_DATA"
echo "  输出目录：$OUTPUT_DIR"
echo "  每批标注数量：$BATCH_SIZE"
echo "  数据增强倍数：$AUG_FACTOR"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ==========================================
# 步骤 1: 图像拼接
# ==========================================
echo "=========================================="
echo "🔧 步骤 1: 图像拼接"
echo "=========================================="
echo "将每个电芯的 18 张图拼接成一张完整图像..."
echo ""

python src/stitching/cell_stitcher.py \
  --input "$RAW_DATA" \
  --output "$OUTPUT_DIR/stitched" \
  --rows 3 \
  --cols 6 \
  --width 20000 \
  --height 8000

echo ""
echo "✅ 拼接完成！结果保存在：$OUTPUT_DIR/stitched"
echo ""

# ==========================================
# 步骤 2: 创建主动学习标注项目
# ==========================================
echo "=========================================="
echo "🏷️  步骤 2: 创建标注项目"
echo "=========================================="
echo "使用主动学习选择最有价值的样本进行标注..."
echo ""

python src/active_learning/annotation_manager.py \
  --working-dir "$OUTPUT_DIR/al_project" \
  --unlabeled-dir "$OUTPUT_DIR/stitched" \
  --batch-size "$BATCH_SIZE" \
  --project-name cell_defect_phase1

echo ""
echo "✅ 标注项目已创建！"
echo ""
echo "📋 下一步操作:"
echo "  1. 查看优先标注列表：$OUTPUT_DIR/al_project/priority_queue/cell_defect_phase1_priority.json"
echo "  2. 使用标注工具（如 CVAT、LabelImg）标注这 $BATCH_SIZE 张图像"
echo "  3. 将标注后的 YOLO 格式标签文件放入：$OUTPUT_DIR/al_project/labeled/labels/"
echo "  4. 将对应的图像复制到：$OUTPUT_DIR/al_project/labeled/images/"
echo ""
read -p "💡 标注完成后按回车继续..."

# ==========================================
# 步骤 3: 数据增强
# ==========================================
echo ""
echo "=========================================="
echo "🔄 步骤 3: 数据增强"
echo "=========================================="
echo "从标注数据生成增强样本..."
echo ""

python src/augmentation/defect_augmenter.py \
  --input "$OUTPUT_DIR/al_project/labeled" \
  --output "$OUTPUT_DIR/augmented" \
  --factor "$AUG_FACTOR"

echo ""
echo "✅ 数据增强完成！"
echo ""

# ==========================================
# 步骤 4: 训练模型
# ==========================================
echo ""
echo "=========================================="
echo "🎯 步骤 4: 训练模型"
echo "=========================================="
echo "使用增强后的数据训练 YOLOv8 模型..."
echo ""

python src/train.py \
  --data "$OUTPUT_DIR/augmented/dataset.yaml" \
  --model yolov8l.pt \
  --epochs 300 \
  --batch 8 \
  --imgsz 1024 \
  --name cell_defect_v1 \
  --patience 50

echo ""
echo "✅ 训练完成！"
echo ""

# ==========================================
# 步骤 5: 评估模型
# ==========================================
echo ""
echo "=========================================="
echo "📊 步骤 5: 模型评估"
echo "=========================================="
echo "在验证集上评估模型性能..."
echo ""

python src/evaluate.py \
  --model "$OUTPUT_DIR/runs/detect/cell_defect_v1/weights/best.pt" \
  --data "$OUTPUT_DIR/augmented/dataset.yaml" \
  --target-recall 1.0 \
  --target-precision 0.95

echo ""
echo "=========================================="
echo "🎉 全部完成！"
echo "=========================================="
echo ""
echo "📦 产出物:"
echo "  - 训练好的模型：$OUTPUT_DIR/runs/detect/cell_defect_v1/weights/best.pt"
echo "  - 评估报告：$OUTPUT_DIR/runs/detect/cell_defect_v1/results.json"
echo ""
echo "🚀 下一步:"
echo "  1. 如果指标未达标，重复步骤 2-5（增量标注 + 训练）"
echo "  2. 部署到产线：python deploy/api/server.py --model <模型路径>"
echo "  3. 或者使用 Docker：docker-compose up -d"
echo ""
echo "💡 提示：要达到 Recall=100%，建议:"
echo "  - 推理时使用低置信度阈值 (conf=0.1)"
echo "  - 增加一轮主动学习标注"
echo "  - 考虑使用两级检测策略（检测 + 分类验证）"
echo ""
