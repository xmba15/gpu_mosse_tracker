#!/usr/bin/env python
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torchvision

try:
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated

import logging

from .types import Array, BBoxXYWH

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s:%(message)s")
_logger = logging.getLogger(__name__)

__all__ = ["MosseTracker"]


class MosseTracker:
    def __init__(
        self,
        img_shape: Annotated[Tuple[int], 2],
        use_gpu: bool = False,
        sigma: float = 1.0,
        num_perturbations=128,
        min_psr: float = 8.0,
        seed=2021,
    ):
        self._device = torch.device("cpu")
        if use_gpu:
            if not torch.cuda.is_available():
                _logger.warning("Cuda not found. Fallback to cpu now")
            else:
                self._device = torch.device("cuda")

        self._sigma = sigma
        self._num_perturbations = num_perturbations
        self._min_psr = min_psr
        self._bbox_xywh: Optional[BBoxXYWH] = None
        torch.manual_seed(2021)

    def init(self, frame: Array[Tuple[int, int], np.uint8], bbox_xywh: BBoxXYWH) -> bool:
        assert frame.ndim == 2 and frame.dtype == np.uint8, "frame is invalid"

        self._bbox_xywh = BBoxXYWH(*bbox_xywh)
        assert self._bbox_xywh.width > 0 and self._bbox_xywh.height > 0

        self._G = torch.fft.fft2(self._get_gauss_response(self._bbox_xywh.width, self._bbox_xywh.height))
        x_min, y_min, width, height = bbox_xywh
        roi = frame[y_min : y_min + height, x_min : x_min + width]
        roi = self._preprocess(roi)
        fis = self._perturb_image(roi, self._num_perturbations)
        FIs = torch.fft.fft2(fis)
        FIs_conj = torch.conj(FIs)
        Ai = self._G[None, :, :].repeat((self._num_perturbations, 1, 1)) * FIs_conj
        Bi = FIs * FIs_conj

        self._Ai = torch.sum(Ai, dim=0)
        self._Bi = torch.sum(Bi, dim=0)

        return True

    def update(self, frame: Array[Tuple[int, int], np.uint8], rate=0.125, eps=1e-5) -> Optional[BBoxXYWH]:
        assert frame.ndim == 2 and frame.dtype == np.uint8, "frame is invalid"

        if not self._bbox_xywh:
            _logger.warning("failed to initialize the first bounding box")
            return None

        x_min, y_min, width, height = self._bbox_xywh
        roi = frame[y_min : y_min + height, x_min : x_min + width]
        fi = self._preprocess(roi)
        FI = torch.fft.fft2(fi)
        Hi = self._Ai / self._Bi
        response = torch.fft.ifft2(Hi * FI).real

        new_row_center, new_col_center = (response == torch.max(response)).nonzero().to("cpu").detach().numpy()[0]
        dx = int(new_col_center - width / 2.0)
        dy = int(new_row_center - height / 2.0)

        psr = (response[new_row_center, new_col_center] - response.mean()) / (response.std() + eps)
        if psr < self._min_psr:
            _logger.warning(f"low psr: {psr}")
        new_bbox = MosseTracker.correct_bbox_xywh(
            BBoxXYWH(x_min + dx, y_min + dy, width, height), frame.shape[1], frame.shape[0]
        )
        if new_bbox.width != self._bbox_xywh.width or new_bbox.height != new_bbox.height:
            self.init(frame, new_bbox)
        else:
            self._update_filters(frame, self._bbox_xywh, rate)

        self._bbox_xywh = new_bbox

        return self._bbox_xywh

    def _update_filters(self, frame: Array[Tuple[int, int], np.uint8], bbox_xywh, rate) -> None:
        x_min, y_min, width, height = bbox_xywh
        roi = frame[y_min : y_min + height, x_min : x_min + width]
        fi = self._preprocess(roi)
        FI = torch.fft.fft2(fi)
        FI_conj = torch.conj(FI)
        self._Ai = rate * self._G * FI_conj + (1 - rate) * self._Ai
        self._Bi = rate * FI * FI_conj + (1 - rate) * self._Bi

    def _get_gauss_response(self, width, height) -> torch.Tensor:
        x_center = width / 2.0
        y_center = height / 2.0
        yy, xx = torch.meshgrid(torch.arange(height, device=self._device), torch.arange(width, device=self._device))
        response = (torch.square(xx - x_center) + torch.square(yy - y_center)) / (2 * self._sigma)
        response = torch.exp(-response)
        response -= response.min()
        response /= response.max() - response.min()

        return response

    def _perturb_image(self, frame: torch.Tensor, num_samples: int, rot_degree: float = 180 / 10.0):
        transform = torchvision.transforms.RandomRotation(rot_degree)

        return transform(frame[None, :, :].repeat((num_samples, 1, 1)))

    def _preprocess(self, frame: Array[Tuple[int, int], np.uint8], eps: float = 1e-5):
        frame_tensor = torch.from_numpy(frame.astype(np.float32)).to(self._device)
        frame_tensor = torch.log(frame_tensor + 1)
        frame_tensor = (frame_tensor - frame_tensor.mean()) / (frame_tensor + eps)
        height, width = frame.shape[:2]
        return frame_tensor * MosseTracker.hann_window_2d(height, width, self._device)

    @staticmethod
    def hann_window_2d(height: int, width: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        mask_row, mask_col = torch.meshgrid(
            torch.hann_window(height, device=device), torch.hann_window(width, device=device)
        )
        return mask_row * mask_col

    @staticmethod
    def correct_bbox_xywh(bbox_xywh, width, height):
        x_min, y_min, w, h = bbox_xywh
        x_max = x_min + w
        y_max = y_min + h

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(width - 1, x_max)
        y_max = min(height - 1, y_max)

        return BBoxXYWH(x_min, y_min, x_max - x_min, y_max - y_min)
