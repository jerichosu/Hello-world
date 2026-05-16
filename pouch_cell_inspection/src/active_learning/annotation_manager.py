"""
主动学习模块 - 针对小样本缺陷检测优化
目标：用最少的人工标注达到 Recall=100%, Precision>95%

核心策略:
1. 不确定性采样 - 优先标注模型最不确定的样本
2. 多样性采样 - 覆盖不同缺陷类型和场景
3. 错误分析驱动 - 重点标注模型误判的样本
4. 半监督学习 - 利用大量无标签数据
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationPriority:
    """标注优先级信息"""
    image_path: str
    priority_score: float  # 0-1, 越高越优先
    uncertainty: float  # 模型不确定性
    predicted_defects: List[str]  # 预测的缺陷类型
    confidence_scores: List[float]  # 对应置信度
    reason: str  # 优先标注的原因
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ActiveLearningManager:
    """
    主动学习管理器
    
    针对软包电池缺陷检测的特殊优化：
    - 小样本场景（~200张缺陷图）
    - 高召回率要求（Recall=100%）
    - 多类别不平衡
    """
    
    def __init__(
        self,
        working_dir: str,
        initial_labeled_count: int = 200,
        target_recall: float = 1.0,
        target_precision: float = 0.95
    ):
        """
        Args:
            working_dir: 工作目录
            initial_labeled_count: 初始标注数量
            target_recall: 目标召回率
            target_precision: 目标精确率
        """
        self.working_dir = Path(working_dir)
        self.initial_labeled_count = initial_labeled_count
        self.target_recall = target_recall
        self.target_precision = target_precision
        
        # 创建目录结构
        (self.working_dir / "pool").mkdir(parents=True, exist_ok=True)  # 未标注池
        (self.working_dir / "labeled").mkdir(parents=True, exist_ok=True)  # 已标注
        (self.working_dir / "priority_queue").mkdir(parents=True, exist_ok=True)  # 优先队列
        (self.working_dir / "model_outputs").mkdir(parents=True, exist_ok=True)  # 模型输出
        
        self.priority_queue: List[AnnotationPriority] = []
    
    def compute_prediction_uncertainty(
        self,
        predictions: List[Dict],
        method: str = "entropy"
    ) -> float:
        """
        计算预测的不确定性
        
        Args:
            predictions: 模型预测结果列表
            method: 计算方法 ["entropy", "margin", "variance"]
        
        Returns:
            不确定性分数 (0-1)
        """
        if len(predictions) == 0:
            return 1.0
        
        confidences = [p.get("confidence", 0.5) for p in predictions]
        
        if method == "entropy":
            # 基于熵的不确定性
            entropy = 0.0
            for conf in confidences:
                if conf > 0 and conf < 1:
                    entropy -= conf * np.log2(conf) + (1 - conf) * np.log2(1 - conf)
            # 归一化到 [0, 1]
            max_entropy = len(confidences) * 1.0  # 最大可能熵
            uncertainty = entropy / max_entropy if max_entropy > 0 else 1.0
            
        elif method == "margin":
            # 基于间隔的不确定性（最高和次高置信度的差）
            if len(confidences) < 2:
                uncertainty = 1.0 - (confidences[0] if confidences else 0.5)
            else:
                sorted_conf = sorted(confidences, reverse=True)
                margin = sorted_conf[0] - sorted_conf[1]
                uncertainty = 1.0 - margin
                
        elif method == "variance":
            # 基于方差的不确定性
            uncertainty = np.std(confidences) if len(confidences) > 1 else 1.0
            
        else:
            uncertainty = 1.0 - np.mean(confidences)
        
        return min(1.0, max(0.0, uncertainty))
    
    def analyze_false_negatives(
        self,
        model_predictions: List[Dict],
        ground_truth: Optional[List[Dict]] = None,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        分析可能的假阴性样本
        
        对于追求 Recall=100% 的场景，这是最关键的步骤
        """
        potential_fn = []
        
        # 低置信度但检测到某些特征的样本
        for pred in model_predictions:
            if pred.get("confidence", 1.0) < threshold:
                potential_fn.append({
                    "image_path": pred.get("image_path"),
                    "reason": "low_confidence_detection",
                    "confidence": pred.get("confidence"),
                    "predicted_class": pred.get("class"),
                    "priority_boost": 0.3  # 优先级提升
                })
        
        # 特定区域的弱响应（如边缘、角落等易漏检区域）
        # 这里可以加入基于位置的启发式规则
        
        return potential_fn
    
    def sample_diverse_cases(
        self,
        unlabeled_pool: List[str],
        feature_vectors: Optional[np.ndarray] = None,
        n_samples: int = 20,
        method: str = "kmeans"
    ) -> List[str]:
        """
        采样多样化的未标注样本
        
        Args:
            unlabeled_pool: 未标注图像路径列表
            feature_vectors: 特征向量（可选，使用图像嵌入）
            n_samples: 采样数量
            method: 采样方法 ["random", "kmeans", "farthest"]
        
        Returns:
            选中的样本路径列表
        """
        if len(unlabeled_pool) <= n_samples:
            return unlabeled_pool
        
        if method == "random":
            return list(np.random.choice(unlabeled_pool, n_samples, replace=False))
        
        elif method == "kmeans" and feature_vectors is not None:
            # 使用 K-Means 聚类，从每个簇中选取代表性样本
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_samples, random_state=42)
            labels = kmeans.fit_predict(feature_vectors)
            
            selected = []
            for cluster_id in range(n_samples):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 0:
                    # 选择离簇中心最近的样本
                    center = kmeans.cluster_centers_[cluster_id]
                    cluster_features = feature_vectors[cluster_indices]
                    distances = np.linalg.norm(cluster_features - center, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected.append(unlabeled_pool[closest_idx])
            
            return selected
        
        elif method == "farthest" and feature_vectors is not None:
            # 最远点采样
            selected = [unlabeled_pool[0]]
            selected_features = [feature_vectors[0]]
            
            for _ in range(n_samples - 1):
                # 找到距离已选样本最远的点
                distances = []
                for i, feat in enumerate(feature_vectors):
                    if unlabeled_pool[i] in selected:
                        continue
                    min_dist = min(np.linalg.norm(feat - sf) for sf in selected_features)
                    distances.append((i, min_dist))
                
                if distances:
                    farthest_idx = max(distances, key=lambda x: x[1])[0]
                    selected.append(unlabeled_pool[farthest_idx])
                    selected_features.append(feature_vectors[farthest_idx])
            
            return selected
        
        else:
            # 默认随机采样
            return self.sample_diverse_cases(unlabeled_pool, None, n_samples, "random")
    
    def prioritize_annotation_batch(
        self,
        unlabeled_images: List[str],
        model_predictions: Dict[str, List[Dict]],
        n_priority: int = 30
    ) -> List[AnnotationPriority]:
        """
        为一批未标注图像生成标注优先级
        
        Args:
            unlabeled_images: 未标注图像路径
            model_predictions: 模型预测结果 {image_path: [predictions]}
            n_priority: 返回的优先样本数
        
        Returns:
            按优先级排序的标注任务列表
        """
        priorities = []
        
        for img_path in unlabeled_images:
            preds = model_predictions.get(img_path, [])
            
            # 计算不确定性
            uncertainty = self.compute_prediction_uncertainty(preds, method="entropy")
            
            # 检查是否可能是假阴性
            fn_analysis = self.analyze_false_negatives(preds, threshold=0.6)
            fn_boost = max([f.get("priority_boost", 0) for f in fn_analysis], default=0)
            
            # 综合优先级分数
            base_score = uncertainty
            priority_score = min(1.0, base_score + fn_boost)
            
            # 确定原因
            reasons = []
            if uncertainty > 0.7:
                reasons.append("high_uncertainty")
            if fn_boost > 0:
                reasons.append("potential_false_negative")
            if len(preds) == 0:
                reasons.append("no_detection")
                priority_score = max(priority_score, 0.5)  # 无检测的样本也需关注
            
            priority = AnnotationPriority(
                image_path=img_path,
                priority_score=priority_score,
                uncertainty=uncertainty,
                predicted_defects=[p.get("class", "unknown") for p in preds],
                confidence_scores=[p.get("confidence", 0) for p in preds],
                reason=", ".join(reasons) if reasons else "normal_sample"
            )
            
            priorities.append(priority)
        
        # 按优先级排序
        priorities.sort(key=lambda x: x.priority_score, reverse=True)
        
        self.priority_queue = priorities[:n_priority]
        return self.priority_queue
    
    def estimate_labeling_effort(
        self,
        current_metrics: Dict[str, float],
        target_metrics: Dict[str, float],
        current_labeled_count: int
    ) -> Dict[str, any]:
        """
        估算达到目标指标所需的标注工作量
        
        基于学习曲线的经验估计
        """
        current_recall = current_metrics.get("recall", 0.8)
        current_precision = current_metrics.get("precision", 0.9)
        
        # 简化的学习曲线模型：性能 ~ log(样本数)
        # 实际应用中应基于历史数据拟合
        
        recall_gap = target_metrics.get("recall", 1.0) - current_recall
        precision_gap = target_metrics.get("precision", 0.95) - current_precision
        
        # 估算需要的额外样本
        if recall_gap > 0:
            # 召回率提升通常需要更多正样本
            additional_for_recall = int(current_labeled_count * (recall_gap / (1 - current_recall + 0.01)) * 1.5)
        else:
            additional_for_recall = 0
        
        if precision_gap > 0:
            # 精确率提升需要更多负样本和难例
            additional_for_precision = int(current_labeled_count * (precision_gap / (1 - current_precision + 0.01)) * 1.2)
        else:
            additional_for_precision = 0
        
        total_additional = max(additional_for_recall, additional_for_precision)
        
        return {
            "current_labeled": current_labeled_count,
            "estimated_additional_labels": total_additional,
            "estimated_total_labels": current_labeled_count + total_additional,
            "recall_gap": recall_gap,
            "precision_gap": precision_gap,
            "recommendation": self._generate_recommendation(recall_gap, precision_gap)
        }
    
    def _generate_recommendation(self, recall_gap: float, precision_gap: float) -> str:
        """生成标注建议"""
        recommendations = []
        
        if recall_gap > 0.1:
            recommendations.append("⚠️ 召回率差距较大，建议优先标注含缺陷样本，特别是难以识别的缺陷类型")
        elif recall_gap > 0:
            recommendations.append("📌 小幅提升召回率，重点关注模型漏检的边缘案例")
        
        if precision_gap > 0.05:
            recommendations.append("⚠️ 精确率有待提升，建议增加背景/正常样本的标注，减少误报")
        elif precision_gap > 0:
            recommendations.append("📌 小幅提升精确率，可考虑调整置信度阈值或增加难例挖掘")
        
        if recall_gap <= 0 and precision_gap <= 0:
            recommendations.append("✅ 当前指标已达到目标，可考虑停止标注或进行模型微调")
        
        recommendations.append("\n💡 建议采用迭代式标注策略：每标注20-30张图后重新训练并评估")
        
        return "\n".join(recommendations)
    
    def export_priority_list(
        self,
        output_path: str,
        format: str = "json"
    ) -> None:
        """导出优先标注列表"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            data = [
                {
                    "image_path": p.image_path,
                    "priority_score": p.priority_score,
                    "uncertainty": p.uncertainty,
                    "predicted_defects": p.predicted_defects,
                    "confidence_scores": p.confidence_scores,
                    "reason": p.reason,
                    "timestamp": p.timestamp
                }
                for p in self.priority_queue
            ]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "image_path", "priority_score", "uncertainty",
                    "predicted_defects", "reason", "timestamp"
                ])
                for p in self.priority_queue:
                    writer.writerow([
                        p.image_path, p.priority_score, p.uncertainty,
                        ";".join(p.predicted_defects), p.reason, p.timestamp
                    ])
        
        logger.info(f"Exported {len(self.priority_queue)} priority items to {output_path}")
    
    def create_annotation_project(
        self,
        project_name: str,
        unlabeled_dir: str,
        model_output_dir: str,
        batch_size: int = 30
    ) -> Dict[str, any]:
        """
        创建标注项目
        
        集成主动学习策略，自动生成优先标注队列
        """
        unlabeled_path = Path(unlabeled_dir)
        model_output_path = Path(model_output_dir)
        
        # 收集未标注图像
        unlabeled_images = [
            str(p) for p in unlabeled_path.glob("*.jpg") + 
                       list(unlabeled_path.glob("*.png"))
        ]
        
        # 加载模型预测结果
        model_predictions = {}
        for pred_file in model_output_path.glob("*.json"):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                img_path = pred_data.get("image_path", pred_file.stem)
                model_predictions[img_path] = pred_data.get("predictions", [])
        
        # 生成优先级队列
        priorities = self.prioritize_annotation_batch(
            unlabeled_images,
            model_predictions,
            n_priority=batch_size
        )
        
        # 导出标注任务
        export_path = self.working_dir / "priority_queue" / f"{project_name}_priority.json"
        self.export_priority_list(str(export_path), format="json")
        
        # 估算工作量
        effort_estimate = self.estimate_labeling_effort(
            current_metrics={"recall": 0.85, "precision": 0.90},  # 示例值
            target_metrics={"recall": self.target_recall, "precision": self.target_precision},
            current_labeled_count=self.initial_labeled_count
        )
        
        return {
            "project_name": project_name,
            "total_unlabeled": len(unlabeled_images),
            "priority_batch_size": batch_size,
            "priority_queue_exported": str(export_path),
            "effort_estimate": effort_estimate,
            "next_steps": [
                f"1. 标注导出的 {batch_size} 张高优先级图像",
                "2. 将标注后的图像移入 labeled/ 目录",
                "3. 重新训练模型：python src/train.py --data <新数据集>",
                "4. 运行主动学习循环：python src/active_learning/run_cycle.py",
                "5. 重复直到达到目标指标"
            ]
        }


def run_active_learning_cycle(
    working_dir: str,
    model_path: str,
    unlabeled_dir: str,
    output_dir: str,
    batch_size: int = 30
) -> None:
    """
    运行一轮主动学习循环
    
    1. 用当前模型预测未标注数据
    2. 计算优先级
    3. 导出待标注列表
    """
    manager = ActiveLearningManager(
        working_dir=working_dir,
        initial_labeled_count=200,
        target_recall=1.0,
        target_precision=0.95
    )
    
    # 这里应该调用模型推理，简化起见假设已有预测结果
    # 实际使用时需要集成推理模块
    
    print(f"✓ Active learning cycle initialized")
    print(f"📁 Working directory: {working_dir}")
    print(f"📊 Batch size: {batch_size}")
    print(f"🎯 Target: Recall=100%, Precision>95%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="主动学习标注管理工具")
    parser.add_argument("--working-dir", type=str, required=True, help="工作目录")
    parser.add_argument("--unlabeled-dir", type=str, required=True, help="未标注数据目录")
    parser.add_argument("--model-output", type=str, help="模型预测结果目录")
    parser.add_argument("--batch-size", type=int, default=30, help="每批标注数量")
    parser.add_argument("--project-name", type=str, default="cell_defect", help="项目名称")
    
    args = parser.parse_args()
    
    manager = ActiveLearningManager(
        working_dir=args.working_dir,
        initial_labeled_count=200,
        target_recall=1.0,
        target_precision=0.95
    )
    
    if args.model_output:
        result = manager.create_annotation_project(
            project_name=args.project_name,
            unlabeled_dir=args.unlabeled_dir,
            model_output_dir=args.model_output,
            batch_size=args.batch_size
        )
        
        print("\n" + "="*60)
        print("📋 主动学习标注项目已创建")
        print("="*60)
        print(f"项目名称：{result['project_name']}")
        print(f"未标注图像：{result['total_unlabeled']} 张")
        print(f"本批优先标注：{result['priority_batch_size']} 张")
        print(f"\n📈 工作量估算:")
        print(f"  当前标注数：{result['effort_estimate']['current_labeled']}")
        print(f"  预计还需标注：{result['effort_estimate']['estimated_additional_labels']} 张")
        print(f"  预计总标注量：{result['effort_estimate']['estimated_total_labels']} 张")
        print(f"\n💡 建议:\n{result['effort_estimate']['recommendation']}")
        print(f"\n📝 下一步:\n" + "\n".join(result['next_steps']))
    else:
        run_active_learning_cycle(
            working_dir=args.working_dir,
            model_path="",
            unlabeled_dir=args.unlabeled_dir,
            output_dir=args.working_dir,
            batch_size=args.batch_size
        )
