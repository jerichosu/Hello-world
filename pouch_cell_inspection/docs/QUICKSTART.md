# 快速开始指南

## 环境要求

- Python >= 3.8
- CUDA >= 11.8 (可选，用于 GPU 加速)
- NVIDIA GPU (可选，推荐 RTX 3060 或更高)

## 安装步骤

### 1. 克隆仓库
```bash
git clone <repository-url>
cd pouch_cell_inspection
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 准备数据集

将数据集组织为以下结构：
```
data/
└── pouch_cell/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

`data.yaml` 示例：
```yaml
path: data/pouch_cell
train: train/images
val: val/images
nc: 11
names:
  - normal
  - scratch
  - dent
  - stain
  - wrinkle
  - bubble
  - poor_sealing
  - leakage
  - foreign_object
  - tab_misalignment
  - deformation
```

### 5. 训练模型
```bash
python src/train.py --data data/pouch_cell/data.yaml --epochs 100 --batch-size 16
```

### 6. 推理检测
```bash
python -c "
from src.inference import Detector
detector = Detector('runs/train/pouch_cell_exp/weights/best.pt')
result = detector.detect('test_image.jpg', return_visualized=True)
print(f'Detected {result[\"detection_count\"]} defects')
"
```

### 7. 启动 API 服务
```bash
python deploy/api/server.py --port 8000 --model runs/train/pouch_cell_exp/weights/best.pt
```

### 8. 使用 Docker 部署
```bash
cd deploy/docker
docker-compose up -d
```

## API 使用示例

### curl 请求
```bash
# 健康检查
curl http://localhost:8000/health

# 单张图像检测
curl -X POST http://localhost:8000/detect/image \
  -F "file=@test_image.jpg" \
  -F "return_visualized=true"

# 批量检测
curl -X POST http://localhost:8000/detect/batch \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

### Python 客户端
```python
import requests

# 上传图像进行检测
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/detect/image', files=files)
    result = response.json()
    print(result)
```

## 性能优化建议

1. **GPU 加速**: 使用 NVIDIA GPU 可获得最佳性能
2. **模型量化**: 导出为 TensorRT 引擎以获得更快推理
3. **批处理**: 批量处理图像可提高吞吐量
4. **图像尺寸**: 根据实际需求调整输入图像尺寸

## 常见问题

**Q: CUDA out of memory?**
A: 减小 batch_size 或使用更小的模型 (yolov8n)

**Q: 检测精度不够？**
A: 
- 增加训练数据量
- 调整置信度阈值
- 使用更大的模型 (yolov8l/x)

**Q: 推理速度太慢？**
A:
- 导出为 ONNX 或 TensorRT 格式
- 使用 GPU 推理
- 减小输入图像尺寸
