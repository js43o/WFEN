"""Frozen eDifFIQA-T reference selector using the OpenCV Zoo ONNX checkpoint.

Input frames must be aligned RGB tensors, already normalized to [-1, 1].
The model is loaded lazily by the caller (e.g. inside DataLoader workers).
"""

from pathlib import Path

import cv2
import numpy as np
import torch


class EDifFIQATSelector:
    def __init__(self, onnx_path):
        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing eDifFIQA-T weights: {onnx_path}")
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    @torch.no_grad()
    def score_sequence(self, frames, mask=None):
        """Score [T,3,112,112] RGB tensors in [-1,1]. Invalid frames = -inf.

        Returns torch.float32 [T] on CPU; higher scores indicate better FIQA.
        """
        if frames.ndim != 4 or frames.shape[1:] != (3, 112, 112):
            raise ValueError("Expected aligned RGB frames of shape [T,3,112,112]")
        count = frames.shape[0]
        if mask is None:
            mask = torch.ones(count, dtype=torch.bool)
        mask = mask.detach().cpu().bool().reshape(-1)
        if mask.numel() != count or not bool(mask.any()):
            raise ValueError("Mask must match frame count and contain valid frames")

        scores = torch.full((count,), -float("inf"), dtype=torch.float32)
        for i in range(count):
            if not mask[i]:
                continue
            # Frames are already RGB, 112x112, normalized as OpenCV Zoo expects.
            blob = frames[i].detach().cpu().unsqueeze(0).contiguous().numpy()
            blob = np.ascontiguousarray(blob, dtype=np.float32)
            self.net.setInput(blob)
            score = float(self.net.forward().reshape(-1)[0])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite eDifFIQA-T score for frame {i}")
            scores[i] = score
        return scores

    @torch.no_grad()
    def select_sequence(self, frames, mask=None):
        scores = self.score_sequence(frames, mask)
        return int(scores.argmax().item()), scores

    @torch.no_grad()
    def select_batch(self, frames, masks=None):
        """Convenience API for SR inference with [B,T,3,112,112] tensors."""
        if frames.ndim != 5:
            raise ValueError("Expected [B,T,3,112,112]")
        if masks is None:
            masks = torch.ones(frames.shape[:2], dtype=torch.bool)
        indices = [
            self.select_sequence(frames[b], masks[b])[0] for b in range(frames.shape[0])
        ]
        return torch.tensor(indices, dtype=torch.long)
