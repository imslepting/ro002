"""SAM3 目標物件分割技能封裝"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from shared.types import SAM3Result
from . import sam3_visualizer

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("skill_sam3", {})


class SAM3Skill:
    """SAM3 目標分割技能封裝"""

    def __init__(self, device: str = "cuda"):
        """載入 SAM3 模型到指定裝置"""
        self._cfg = _load_config()
        self._device = device

        logger.info("Loading SAM3 model on %s ...", device)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        checkpoint = self._cfg.get("checkpoint")
        checkpoint_path = None
        if checkpoint:
            checkpoint_path = str(_PROJECT_ROOT / checkpoint)
            logger.info("Using local checkpoint: %s", checkpoint_path)

        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=(checkpoint_path is None),
        )
        self._processor = Sam3Processor(model)

        # Image encoding 快取（同一場景可跳過 ViT 重複計算）
        self._cached_state = None
        self._cached_image_fp: int | None = None

        logger.info("SAM3 model loaded.")

    @staticmethod
    def _image_fingerprint(rgb_image: np.ndarray) -> int:
        """快速指紋：取樣少量像素做 hash，~0.01ms"""
        h, w = rgb_image.shape[:2]
        step_h = max(1, h // 4)
        step_w = max(1, w // 4)
        return hash(rgb_image[::step_h, ::step_w, :].tobytes())

    def encode_image(self, rgb_image: np.ndarray):
        """預編碼圖片 ViT（可與 LLM 調用並行），快取供後續 segment() 復用"""
        fp = self._image_fingerprint(rgb_image)
        if fp == self._cached_image_fp and self._cached_state is not None:
            logger.info("SAM3: image already cached, skip encode")
            return
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        self._cached_state = self._processor.set_image(pil_image)
        self._cached_image_fp = fp
        logger.info("SAM3: image encoded and cached (ViT)")

    def segment(
        self,
        rgb_image: np.ndarray,
        object_description: str,
        score_threshold: float | None = None,
        max_masks: int | None = None,
    ) -> SAM3Result:
        """執行分割推理

        Parameters
        ----------
        rgb_image : BGR uint8 (H, W, 3)
        object_description : 文字描述，例如 "紅色瓶子"
        score_threshold : 過濾低置信度 mask（None 使用 config 預設值）
        max_masks : 最多保留幾個 mask（None 使用 config 預設值）

        Returns
        -------
        SAM3Result
        """
        if score_threshold is None:
            score_threshold = self._cfg.get("score_threshold", 0.3)
        if max_masks is None:
            max_masks = self._cfg.get("max_masks", 5)
        alpha = self._cfg.get("overlay_alpha", 0.4)

        # BGR → RGB → PIL.Image
        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # 推理 — 若圖像相同則復用 ViT encoding（省 2-4 秒）
        fp = self._image_fingerprint(rgb_image)
        if fp == self._cached_image_fp and self._cached_state is not None:
            logger.info("SAM3: reusing cached image encoding (skip ViT)")
            state = self._cached_state
        else:
            state = self._processor.set_image(pil_image)
            self._cached_state = state
            self._cached_image_fp = fp
            logger.info("SAM3: image encoded (ViT)")

        output = self._processor.set_text_prompt(
            state=state, prompt=object_description,
        )

        # SAM3 returns tensors: masks (N,1,H,W), scores (N,), boxes (N,4)
        raw_masks = output["masks"].cpu()
        raw_scores = output["scores"].cpu()
        raw_boxes = output["boxes"].cpu()

        n = raw_scores.shape[0]

        # 按 score 排序（高→低）
        if n > 0:
            order = raw_scores.argsort(descending=True)
        else:
            order = []

        masks: list[np.ndarray] = []
        scores: list[float] = []
        boxes: list[np.ndarray] = []

        for idx in order:
            s = float(raw_scores[idx])
            if s < score_threshold:
                continue
            # squeeze (1, H, W) → (H, W)
            mask_np = raw_masks[idx].squeeze(0).numpy().astype(bool)
            masks.append(mask_np)
            scores.append(s)
            boxes.append(raw_boxes[idx].numpy())
            if len(masks) >= max_masks:
                break

        if not masks:
            logger.warning(
                "No masks above threshold %.2f for '%s'",
                score_threshold, object_description,
            )
            h, w = rgb_image.shape[:2]
            empty_mask = np.zeros((h, w), dtype=bool)
            return SAM3Result(
                masks=[],
                scores=[],
                boxes=[],
                best_mask=empty_mask,
                best_score=0.0,
                annotated_image=rgb_image.copy(),
                object_description=object_description,
            )

        best_mask = masks[0]
        best_score = scores[0]

        # 可視化標注
        annotated = sam3_visualizer.annotate_image(
            rgb_image, masks, scores, boxes,
            object_description, alpha=alpha,
        )

        return SAM3Result(
            masks=masks,
            scores=scores,
            boxes=boxes,
            best_mask=best_mask,
            best_score=best_score,
            annotated_image=annotated,
            object_description=object_description,
        )

    def warmup(self) -> None:
        """空跑一次暖機（640x480 黑圖）"""
        logger.info("SAM3 warmup ...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.segment(dummy, "warmup")
        logger.info("SAM3 warmup done.")
