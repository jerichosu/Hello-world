"""
电芯图像拼接模块
将单个电芯的18张多角度图像拼接成一张完整图像 (8000x20000)
支持自动对齐、曝光补偿、无缝融合
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CellImageInfo:
    """单张电芯图像信息"""
    image_path: Path
    angle: str  # 拍摄角度标识
    position: Tuple[int, int]  # 在拼接图中的预期位置 (row, col)
    image: Optional[np.ndarray] = None


class CellStitcher:
    """
    电芯图像拼接器
    
    针对软包电池电芯的18张多角度图像拼接优化：
    - 自动检测图像重叠区域
    - 曝光一致性调整
    - 无缝融合处理
    - 支持手动位置校正
    """
    
    def __init__(
        self,
        overlap_threshold: float = 0.15,
        blend_width: int = 50,
        exposure_compensation: bool = True,
        use_gpu: bool = False
    ):
        """
        Args:
            overlap_threshold: 图像重叠比例阈值
            blend_width: 融合带宽度（像素）
            exposure_compensation: 是否启用曝光补偿
            use_gpu: 是否使用GPU加速（需要CUDA支持）
        """
        self.overlap_threshold = overlap_threshold
        self.blend_width = blend_width
        self.exposure_compensation = exposure_compensation
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        
        if self.use_gpu:
            logger.info("GPU acceleration enabled")
        else:
            logger.info("Using CPU mode")
    
    def load_images(self, image_paths: List[Path]) -> List[np.ndarray]:
        """加载图像列表"""
        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"Failed to load image: {path}")
                continue
            images.append(img)
        logger.info(f"Loaded {len(images)} images")
        return images
    
    def estimate_positions(
        self,
        images: List[np.ndarray],
        grid_rows: int = 3,
        grid_cols: int = 6
    ) -> List[Tuple[int, int]]:
        """
        估计图像在拼接图中的位置
        
        对于18张图，默认按3行6列排列
        可通过特征匹配自动调整
        """
        positions = []
        for i in range(len(images)):
            row = i // grid_cols
            col = i % grid_cols
            positions.append((row, col))
        return positions
    
    def compute_exposure_gain(self, ref_image: np.ndarray, target_image: np.ndarray) -> float:
        """计算曝光增益系数"""
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        ref_mean = np.mean(ref_gray)
        target_mean = np.mean(target_gray)
        
        if target_mean < 1e-6:
            return 1.0
        
        gain = ref_mean / target_mean
        # 限制增益范围避免过曝/欠曝
        gain = np.clip(gain, 0.5, 2.0)
        return gain
    
    def apply_exposure_compensation(
        self,
        images: List[np.ndarray],
        reference_idx: int = 0
    ) -> List[np.ndarray]:
        """对图像序列应用曝光补偿"""
        if not self.exposure_compensation:
            return images
        
        compensated = []
        ref_image = images[reference_idx]
        
        for i, img in enumerate(images):
            if i == reference_idx:
                compensated.append(img.copy())
                continue
            
            gain = self.compute_exposure_gain(ref_image, img)
            compensated_img = cv2.convertScaleAbs(img, alpha=gain, beta=0)
            compensated.append(compensated_img)
            
            logger.debug(f"Image {i}: exposure gain = {gain:.3f}")
        
        return compensated
    
    def find_overlap_region(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        direction: str = 'horizontal'
    ) -> Tuple[int, int, int, int]:
        """
        查找两张图像的重叠区域
        
        Returns:
            (x1, y1, x2, y2): img2相对于img1的偏移量
        """
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # 使用ORB特征检测
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.warning("Insufficient features for matching")
            return (0, 0, img2.shape[1], img2.shape[0])
        
        # 特征匹配
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        if len(matches) < 20:
            logger.warning(f"Few matches found: {len(matches)}")
            return (0, 0, img2.shape[1], img2.shape[0])
        
        # RANSAC估计单应性矩阵
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return (0, 0, img2.shape[1], img2.shape[0])
        
        # 计算偏移量
        h, w = img2.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        
        x_offset = int(np.mean(transformed[:, 0, 0]))
        y_offset = int(np.mean(transformed[:, 0, 1]))
        
        return (max(0, -x_offset), max(0, -y_offset), 
                img2.shape[1] + min(0, w - x_offset), 
                img2.shape[0] + min(0, h - y_offset))
    
    def blend_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_x: int,
        overlap_y: int
    ) -> np.ndarray:
        """融合两张有重叠的图像"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 创建输出画布
        out_h = max(h1, overlap_y + h2)
        out_w = max(w1, overlap_x + w2)
        
        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # 放置第一张图
        result[:h1, :w1] = img1
        
        # 创建第二张图的掩码
        mask = np.zeros((out_h, out_w), dtype=np.uint8)
        mask[overlap_y:overlap_y+h2, overlap_x:overlap_x+w2] = 255
        
        # 在重叠区域创建渐变权重
        if overlap_x > 0 and overlap_y == 0:  # 水平重叠
            for i in range(self.blend_width):
                x = overlap_x + i
                if x < out_w:
                    weight = i / self.blend_width
                    result[overlap_y:min(overlap_y+h2, h1), x] = \
                        (1 - weight) * result[overlap_y:min(overlap_y+h2, h1), x] + \
                        weight * img2[overlap_y:min(overlap_y+h2, h1), x - overlap_x]
        elif overlap_y > 0 and overlap_x == 0:  # 垂直重叠
            for i in range(self.blend_width):
                y = overlap_y + i
                if y < out_h:
                    weight = i / self.blend_width
                    result[y, overlap_x:min(overlap_x+w2, w1)] = \
                        (1 - weight) * result[y, overlap_x:min(overlap_x+w2, w1)] + \
                        weight * img2[y - overlap_y, overlap_x:min(overlap_x+w2, w1)]
        else:
            # 简单叠加（复杂情况）
            result[overlap_y:overlap_y+h2, overlap_x:overlap_x+w2] = \
                img2
        
        return result
    
    def stitch_grid(
        self,
        images: List[np.ndarray],
        grid_rows: int = 3,
        grid_cols: int = 6,
        output_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        按网格布局拼接图像
        
        Args:
            images: 图像列表
            grid_rows: 网格行数
            grid_cols: 网格列数
            output_size: 输出尺寸 (width, height)，如 (20000, 8000)
        
        Returns:
            拼接后的完整图像
        """
        if len(images) != grid_rows * grid_cols:
            logger.warning(f"Expected {grid_rows * grid_cols} images, got {len(images)}")
        
        # 曝光补偿
        images = self.apply_exposure_compensation(images)
        
        # 获取图像尺寸
        h, w = images[0].shape[:2]
        
        # 确定输出尺寸
        if output_size is None:
            out_w = w * grid_cols
            out_h = h * grid_rows
        else:
            out_w, out_h = output_size
        
        # 创建输出画布
        stitched = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # 按网格放置图像
        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            
            x = col * w
            y = row * h
            
            # 调整图像尺寸以适应网格
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            
            stitched[y:y+h, x:x+w] = img
        
        logger.info(f"Stitched image size: {stitched.shape[1]}x{stitched.shape[0]}")
        return stitched
    
    def stitch_with_feature_matching(
        self,
        images: List[np.ndarray],
        initial_positions: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        基于特征匹配的精确拼接
        
        适用于图像间有明确重叠区域的场景
        """
        if len(images) == 0:
            raise ValueError("No images to stitch")
        
        # 使用第一张图作为参考
        result = images[0].copy()
        
        for i in range(1, len(images)):
            # 估计与前一帧的重叠
            overlap = self.find_overlap_region(result, images[i], 'horizontal')
            
            # 融合图像
            result = self.blend_images(
                result, images[i],
                overlap_x=overlap[0],
                overlap_y=overlap[1]
            )
        
        return result
    
    def save_stitched_image(
        self,
        stitched: np.ndarray,
        output_path: Path,
        compression: int = 95
    ) -> None:
        """保存拼接后的图像"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 高质量保存
        cv2.imwrite(
            str(output_path),
            stitched,
            [cv2.IMWRITE_JPEG_QUALITY, compression]
        )
        logger.info(f"Saved stitched image to {output_path}")
    
    def process_cell_batch(
        self,
        cell_image_dir: Path,
        output_dir: Path,
        grid_rows: int = 3,
        grid_cols: int = 6,
        output_size: Tuple[int, int] = (20000, 8000)
    ) -> Dict[str, Path]:
        """
        批量处理电芯图像
        
        Args:
            cell_image_dir: 包含单个电芯18张图像的目录
            output_dir: 输出拼接图像的目录
            grid_rows: 网格行数
            grid_cols: 网格列数
            output_size: 输出尺寸
        
        Returns:
            电芯ID到拼接图像路径的映射
        """
        results = {}
        
        # 假设目录结构：cell_image_dir/{cell_id}/{angle_images}
        cell_dirs = [d for d in cell_image_dir.iterdir() if d.is_dir()]
        
        for cell_dir in cell_dirs:
            cell_id = cell_dir.name
            
            # 收集该电芯的所有图像
            image_paths = sorted([
                p for p in cell_dir.glob("*.jpg") + 
                       list(cell_dir.glob("*.png")) + 
                       list(cell_dir.glob("*.bmp"))
            ])
            
            if len(image_paths) == 0:
                logger.warning(f"No images found in {cell_dir}")
                continue
            
            logger.info(f"Processing cell {cell_id} with {len(image_paths)} images")
            
            # 加载图像
            images = self.load_images(image_paths)
            
            if len(images) == 0:
                continue
            
            # 拼接
            stitched = self.stitch_grid(
                images,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                output_size=output_size
            )
            
            # 保存
            output_path = output_dir / f"{cell_id}_stitched.jpg"
            self.save_stitched_image(stitched, output_path)
            
            results[cell_id] = output_path
        
        logger.info(f"Processed {len(results)} cells")
        return results


def create_stitched_dataset(
    raw_data_dir: str,
    output_dir: str,
    grid_config: Dict[str, int] = None
) -> None:
    """
    创建拼接后的数据集
    
    Args:
        raw_data_dir: 原始数据目录（每个电芯一个子目录）
        output_dir: 输出目录
        grid_config: 网格配置 {"rows": 3, "cols": 6}
    """
    stitcher = CellStitcher(
        overlap_threshold=0.15,
        blend_width=50,
        exposure_compensation=True
    )
    
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    grid_rows = grid_config.get("rows", 3) if grid_config else 3
    grid_cols = grid_config.get("cols", 6) if grid_config else 6
    
    stitcher.process_cell_batch(
        cell_image_dir=raw_path,
        output_dir=output_path,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        output_size=(20000, 8000)
    )
    
    print(f"✓ Dataset created at {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="电芯图像拼接工具")
    parser.add_argument("--input", type=str, required=True, help="输入目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--rows", type=int, default=3, help="网格行数")
    parser.add_argument("--cols", type=int, default=6, help="网格列数")
    parser.add_argument("--width", type=int, default=20000, help="输出宽度")
    parser.add_argument("--height", type=int, default=8000, help="输出高度")
    
    args = parser.parse_args()
    
    create_stitched_dataset(
        raw_data_dir=args.input,
        output_dir=args.output,
        grid_config={"rows": args.rows, "cols": args.cols}
    )
