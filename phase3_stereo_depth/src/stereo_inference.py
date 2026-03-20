"""Fast-FoundationStereo 推理封裝 — 載入模型、預測視差"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch


class StereoInference:
    """Fast-FoundationStereo 立體匹配推理"""

    def __init__(
        self,
        model_dir: str,
        max_disp: int = 256,
        valid_iters: int = 8,
        pad_multiple: int = 32,
    ):
        self._max_disp = max_disp
        self._valid_iters = valid_iters
        self._pad_multiple = pad_multiple
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加入 Fast-FoundationStereo 到 sys.path
        # model_dir 例: .../weights/23-36-37/model_best_bp2_serialize.pth → 上溯 3 層
        ffs_root = os.path.abspath(os.path.join(model_dir, "..", "..", ".."))
        if ffs_root not in sys.path:
            sys.path.insert(0, ffs_root)

        from core.utils.utils import InputPadder
        self._InputPadder = InputPadder

        # 載入序列化模型（torch.load 整個模型）
        model = torch.load(model_dir, map_location="cpu", weights_only=False)
        model.args.valid_iters = valid_iters
        model.args.max_disp = max_disp
        self._model = model.to(self._device).eval()
        print(f"[StereoInference] Model loaded on {self._device}")

    def predict_disparity(
        self, left_rect: np.ndarray, right_rect: np.ndarray,
    ) -> np.ndarray:
        """預測視差圖

        Args:
            left_rect: 校正後左圖 BGR uint8 (H,W,3)
            right_rect: 校正後右圖 BGR uint8 (H,W,3)

        Returns:
            disparity: float32 (H,W)，像素單位
        """
        H, W = left_rect.shape[:2]

        # BGR → RGB
        left_rgb = cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB)

        # HWC → NCHW float（保持 0-255 範圍，與官方 demo 一致）
        img0 = torch.as_tensor(left_rgb).float()[None].permute(0, 3, 1, 2).to(self._device)
        img1 = torch.as_tensor(right_rgb).float()[None].permute(0, 3, 1, 2).to(self._device)

        # Pad
        padder = self._InputPadder(img0.shape, divis_by=self._pad_multiple, force_square=False)
        img0, img1 = padder.pad(img0, img1)

        # 推理
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            disp = self._model.forward(
                img0, img1,
                iters=self._valid_iters,
                test_mode=True,
                optimize_build_volume="pytorch1",
            )

        # Unpad
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        return disp.astype(np.float32)

    def warmup(self) -> None:
        """空跑一次暖機 CUDA kernel（第一次含 torch.compile 編譯較慢）"""
        dummy_l = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_r = np.zeros((480, 640, 3), dtype=np.uint8)
        t0 = time.time()
        self.predict_disparity(dummy_l, dummy_r)
        print(f"[StereoInference] Warmup done in {time.time() - t0:.1f}s")

    def set_fast_mode(self, fast: bool) -> None:
        """即時模式用 valid_iters=4；擷取模式用 valid_iters=8"""
        self._valid_iters = 4 if fast else 8
        self._model.args.valid_iters = self._valid_iters

    @property
    def valid_iters(self) -> int:
        return self._valid_iters
