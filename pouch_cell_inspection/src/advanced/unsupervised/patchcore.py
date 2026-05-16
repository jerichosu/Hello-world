"""
无监督异常检测模块：PatchCore 算法实现
适用于仅有正常样本的缺陷检测场景

核心思想:
- 只用正常图像训练，无需任何缺陷标注
- 提取多尺度特征构建正常模式库
- 测试时计算与正常模式的距离，发现异常

优势:
- 零标注成本
- 可检测未知缺陷类型
- 快速部署 (1 周内)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import cv2
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from tqdm import tqdm


class PatchCoreDetector:
    """
    PatchCore 无监督异常检测器
    
    论文: "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022)
    GitHub: https://github.com/amazon-science/patchcore-inspection
    """
    
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers_to_use: List[str] = None,
        patch_size: int = 3,
        stride: int = 1,
        n_neighbors: int = 9,
        pca_ratio: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.patch_size = patch_size
        self.stride = stride
        
        # 加载预训练 backbone
        try:
            import timm
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True
            )
            self.backbone.to(device)
            self.backbone.eval()
            
            # 获取特征通道数
            feature_info = self.backbone.feature_info
            if layers_to_use is None:
                # 默认使用最后两层
                self.layers_to_use = [feature_info[-1]['module'], feature_info[-2]['module']]
            else:
                self.layers_to_use = layers_to_use
            
            self.feature_channels = [
                info['num_chs'] for info in feature_info 
                if info['module'] in self.layers_to_use
            ]
            
        except ImportError:
            raise ImportError("请安装 timm: pip install timm")
        
        # 最近邻模型
        self.n_neighbors = n_neighbors
        self.pca_ratio = pca_ratio
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.pca = None
        
        # 正常特征库
        self.normal_features = None
        self.feature_locations = None  # 记录每个特征的 (image_id, x, y)
    
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取多尺度特征
        
        Args:
            image: HxWx3 numpy array
            
        Returns:
            dict: {layer_name: feature_map}
        """
        # 预处理
        h, w = image.shape[:2]
        input_tensor = cv2.resize(image, (224, 224))  # Resize to model input
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # 标准化 (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        input_tensor = (input_tensor - mean) / std
        
        # 前向传播
        with torch.no_grad():
            features = self.backbone(input_tensor)
        
        # 整理特征
        feature_dict = {}
        for i, layer_name in enumerate(self.layers_to_use):
            feat = features[i]
            # BxCxHxW -> CxHxW
            feat = feat[0].cpu().numpy()
            feature_dict[layer_name] = feat
        
        return feature_dict
    
    def extract_patches_from_features(
        self, 
        feature_dict: Dict[str, np.ndarray],
        image_id: int
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        从特征图中提取局部 patch 特征
        
        Returns:
            patches: NxD 特征矩阵
            locations: [(image_id, layer_idx, spatial_idx), ...]
        """
        all_patches = []
        all_locations = []
        
        for layer_idx, (layer_name, feat) in enumerate(feature_dict.items()):
            C, H, W = feat.shape
            
            # 将特征图 reshape 为空间位置 x 通道
            feat_flat = feat.reshape(C, -1).T  # (H*W) x C
            
            # 提取局部 patch (可选，增强局部性)
            if self.patch_size > 1:
                # TODO: 实现 patch 提取
                patches = feat_flat
            else:
                patches = feat_flat
            
            all_patches.append(patches)
            
            # 记录位置
            for spatial_idx in range(patches.shape[0]):
                all_locations.append((image_id, layer_idx, spatial_idx))
        
        # 拼接所有层的特征
        all_patches = np.vstack(all_patches)
        
        return all_patches, all_locations
    
    def fit(self, normal_images: List[np.ndarray], verbose: bool = True):
        """
        训练：构建正常样本特征库
        
        Args:
            normal_images: 正常图像列表 (只需正常图像！)
        """
        print("🔧 开始构建正常特征库...")
        all_features = []
        all_locations = []
        
        for img_id, image in enumerate(tqdm(normal_images, disable=not verbose)):
            # 提取特征
            features = self.extract_features(image)
            
            # 提取 patch
            patches, locations = self.extract_patches_from_features(features, img_id)
            
            all_features.append(patches)
            
            # 调整 location 的 image_id
            for loc in locations:
                all_locations.append((img_id, loc[1], loc[2]))
        
        # 合并所有特征
        all_features = np.vstack(all_features)
        self.feature_locations = all_locations
        
        print(f"📊 提取了 {all_features.shape[0]} 个特征向量")
        
        # PCA 降维 (加速最近邻搜索)
        if self.pca_ratio < 1.0:
            n_components = max(1, int(all_features.shape[1] * self.pca_ratio))
            print(f"🔽 PCA 降维：{all_features.shape[1]} -> {n_components}")
            self.pca = PCA(n_components=n_components, random_state=42)
            all_features = self.pca.fit_transform(all_features)
        
        self.normal_features = all_features
        
        # 构建最近邻索引
        print("🔍 构建最近邻索引...")
        self.nn_model.fit(all_features)
        
        print("✅ 训练完成!")
    
    def predict(
        self, 
        image: np.ndarray, 
        return_heatmap: bool = True
    ) -> Dict:
        """
        推理：检测异常
        
        Returns:
            {
                "anomaly_score": float,  # 整图异常分数
                "anomaly_map": np.array,  # 异常热力图 (HxW)
                "defect_locations": [...],  # 缺陷位置列表
                "inference_time": float
            }
        """
        start_time = time.time()
        
        if self.normal_features is None:
            raise RuntimeError("请先调用 fit() 训练模型")
        
        # 提取特征
        features = self.extract_features(image)
        patches, locations = self.extract_patches_from_features(features, 0)
        
        # PCA 变换
        if self.pca is not None:
            patches = self.pca.transform(patches)
        
        # 计算异常分数 (到最近邻的平均距离)
        distances, indices = self.nn_model.kneighbors(patches)
        anomaly_scores_per_patch = distances.mean(axis=1)  # N
        
        # 归一化到 [0, 1]
        anomaly_scores_per_patch = (
            anomaly_scores_per_patch - anomaly_scores_per_patch.min()
        ) / (anomaly_scores_per_patch.max() - anomaly_scores_per_patch.min() + 1e-8)
        
        # 生成异常热力图
        h, w = image.shape[:2]
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        
        # 将 patch 分数映射回原图
        for patch_idx, score in enumerate(anomaly_scores_per_patch):
            img_id, layer_idx, spatial_idx = locations[patch_idx]
            
            # 估算空间位置 (简化版)
            # 实际需要根据 layer 的分辨率进行映射
            x = (spatial_idx % 14) * w // 14
            y = (spatial_idx // 14) * h // 14
            
            # 累加到热力图
            patch_h, patch_w = h // 14, w // 14
            y1, y2 = y, min(y + patch_h, h)
            x1, x2 = x, min(x + patch_w, w)
            anomaly_map[y1:y2, x1:x2] = np.maximum(
                anomaly_map[y1:y2, x1:x2], 
                score
            )
        
        # 平滑热力图
        anomaly_map = cv2.GaussianBlur(anomaly_map, (21, 21), sigmaX=10)
        
        # 全局异常分数
        global_anomaly_score = anomaly_map.max()
        
        # 检测缺陷区域
        defect_locations = []
        threshold = 0.6  # 可调节
        
        # 寻找连通区域
        binary_map = (anomaly_map > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 最小面积阈值
                x, y, w_box, h_box = cv2.boundingRect(contour)
                max_score = anomaly_map[y:y+h_box, x:x+w_box].max()
                
                defect_locations.append({
                    "bbox": [x, y, x + w_box, y + h_box],
                    "score": float(max_score),
                    "area": int(area)
                })
        
        inference_time = time.time() - start_time
        
        result = {
            "anomaly_score": float(global_anomaly_score),
            "anomaly_map": anomaly_map if return_heatmap else None,
            "defect_locations": defect_locations,
            "inference_time": inference_time
        }
        
        return result
    
    def save(self, save_path: str):
        """保存模型"""
        import pickle
        
        save_data = {
            "normal_features": self.normal_features,
            "feature_locations": self.feature_locations,
            "pca": self.pca,
            "nn_model": self.nn_model,
            "config": {
                "patch_size": self.patch_size,
                "stride": self.stride,
                "n_neighbors": self.n_neighbors,
                "pca_ratio": self.pca_ratio
            }
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"💾 模型已保存到 {save_path}")
    
    def load(self, load_path: str):
        """加载模型"""
        import pickle
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.normal_features = save_data["normal_features"]
        self.feature_locations = save_data["feature_locations"]
        self.pca = save_data["pca"]
        self.nn_model = save_data["nn_model"]
        
        config = save_data.get("config", {})
        self.patch_size = config.get("patch_size", self.patch_size)
        self.n_neighbors = config.get("n_neighbors", self.n_neighbors)
        self.pca_ratio = config.get("pca_ratio", self.pca_ratio)
        
        print(f"📂 模型已从 {load_path} 加载")


def train_patchcore(
    normal_data_dir: str,
    save_path: str = "models/patchcore_normal.pkl",
    **kwargs
):
    """
    训练 PatchCore 模型的便捷函数
    
    Args:
        normal_data_dir: 正常图像目录
        save_path: 模型保存路径
    """
    # 加载正常图像
    normal_images = []
    normal_dir = Path(normal_data_dir)
    
    for ext in ['*.jpg', '*.png', '*.bmp']:
        for img_path in normal_dir.glob(ext):
            img = cv2.imread(str(img_path))
            if img is not None:
                normal_images.append(img)
    
    print(f"📁 加载了 {len(normal_images)} 张正常图像")
    
    if len(normal_images) == 0:
        raise ValueError(f"未在 {normal_data_dir} 中找到图像")
    
    # 创建并训练检测器
    detector = PatchCoreDetector(**kwargs)
    detector.fit(normal_images)
    
    # 保存模型
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    detector.save(save_path)
    
    return detector


def evaluate_patchcore(
    model_path: str,
    test_data_dir: str,
    threshold: float = 0.6
):
    """
    评估 PatchCore 模型
    
    Args:
        model_path: 模型路径
        test_data_dir: 测试数据目录 (包含正常和缺陷图像)
        threshold: 异常阈值
    """
    # 加载模型
    detector = PatchCoreDetector()
    detector.load(model_path)
    
    # 加载测试图像
    test_images = []
    test_labels = []  # 0=正常，1=缺陷
    
    test_dir = Path(test_data_dir)
    
    # 假设目录结构：test_data/normal/, test_data/defect/
    for split in ['normal', 'defect']:
        split_dir = test_dir / split
        if split_dir.exists():
            label = 0 if split == 'normal' else 1
            for ext in ['*.jpg', '*.png', '*.bmp']:
                for img_path in split_dir.glob(ext):
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        test_images.append(img)
                        test_labels.append(label)
    
    print(f"📊 测试集：{len(test_images)} 张图像 ({sum(test_labels)} 缺陷, {len(test_images)-sum(test_labels)} 正常)")
    
    # 推理
    predictions = []
    inference_times = []
    
    for img in tqdm(test_images, desc="推理中"):
        result = detector.predict(img, return_heatmap=False)
        predictions.append(result["anomaly_score"])
        inference_times.append(result["inference_time"])
    
    # 计算指标
    from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
    
    predictions = np.array(predictions)
    test_labels = np.array(test_labels)
    
    # AUROC
    auroc = roc_auc_score(test_labels, predictions)
    
    # 在指定阈值下的指标
    pred_labels = (predictions > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_labels, pred_labels).ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / len(test_labels)
    
    print("\n" + "="*50)
    print("📈 评估结果")
    print("="*50)
    print(f"AUROC: {auroc:.4f}")
    print(f"Recall: {recall:.4f} ({tp}/{tp+fn})")
    print(f"Precision: {precision:.4f} ({tp}/{tp+fp})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"平均推理时间：{np.mean(inference_times)*1000:.1f}ms")
    print("="*50)
    
    return {
        "auroc": auroc,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
        "inference_time_ms": np.mean(inference_times) * 1000
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PatchCore 无监督异常检测")
    parser.add_argument("--mode", choices=["train", "eval", "demo"], required=True)
    parser.add_argument("--data-dir", type=str, help="数据目录")
    parser.add_argument("--model-path", type=str, default="models/patchcore_normal.pkl")
    parser.add_argument("--threshold", type=float, default=0.6)
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if not args.data_dir:
            raise ValueError("请指定 --data-dir")
        
        detector = train_patchcore(
            normal_data_dir=args.data_dir,
            save_path=args.model_path
        )
        
    elif args.mode == "eval":
        if not args.data_dir:
            raise ValueError("请指定 --data-dir")
        
        evaluate_patchcore(
            model_path=args.model_path,
            test_data_dir=args.data_dir,
            threshold=args.threshold
        )
    
    elif args.mode == "demo":
        # 简单演示
        print("🎬 PatchCore 演示模式")
        print("用法:")
        print("  训练：python unsupervised_detector.py --mode train --data-dir ./normal_images")
        print("  评估：python unsupervised_detector.py --mode eval --data-dir ./test_images")
        print("\n特点:")
        print("  ✅ 只需正常图像，无需标注")
        print("  ✅ 可检测未知缺陷")
        print("  ✅ 快速部署 (1 周)")
