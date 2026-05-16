"""
软包电池电芯缺陷检测系统
Pouch Cell Inspection System
"""

__version__ = "1.0.0"
__author__ = "Inspection Team"

from .model import PouchCellDetector
from .dataset import PouchCellDataset
from .inference import Detector
from .utils import setup_logger, load_config

__all__ = [
    "PouchCellDetector",
    "PouchCellDataset",
    "Detector",
    "setup_logger",
    "load_config"
]
