"""
Abstract base classes for all models.
All classifier and detector wrappers must implement these interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class BaseClassifier(ABC):
    """
    Unified interface for classification models.
    Implementations: DeiT, ViT, etc.
    """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> Dict:
        """
        Run inference on a batch of normalized image tensors.

        Args:
            x: [B, 3, H, W], normalized (ImageNet mean/std)

        Returns:
            dict with keys:
              - 'logits': [B, num_classes]
              - 'probs':  [B, num_classes]
              - 'top1_idx': [B]
        """
        pass

    @abstractmethod
    def get_logits_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that keeps the computation graph.
        Used by attack methods to compute gradients w.r.t. input.

        Args:
            x: [B, 3, H, W], requires_grad should be set by caller

        Returns:
            logits: [B, num_classes]
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "BaseClassifier":
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass


class BaseDetector(ABC):
    """
    Unified interface for detection models.
    Implementations: RTMDet, YOLOX, etc.
    """

    @abstractmethod
    def detect(self, x: torch.Tensor) -> List[Dict]:
        """
        Run inference on a batch of normalized image tensors.

        Args:
            x: [B, 3, H, W]

        Returns:
            list of dicts (one per image), each with:
              - 'boxes':  [N, 4] (x1, y1, x2, y2)
              - 'scores': [N]
              - 'labels': [N]
        """
        pass

    @abstractmethod
    def get_loss_with_grad(self, x: torch.Tensor, targets: List[Dict]) -> torch.Tensor:
        """
        Forward pass returning scalar loss for gradient computation.
        Used by DPatch attack.
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "BaseDetector":
        pass
