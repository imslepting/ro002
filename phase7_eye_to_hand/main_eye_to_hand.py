"""Phase 7 - Eye-to-Hand calibration.

Usage:
    conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode all
    conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode capture
    conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode analyze
    conda run -n ro002 python phase7_eye_to_hand/main_eye_to_hand.py --mode solve

Modes:
- capture: Capture images + robot state (no analysis)
- analyze:  Analyze captured images (estimate Charuco pose)
- solve:    Solve hand-eye calibration from analyzed samples
- all:      Capture -> Analyze -> Solve (default)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase1_intrinsics.src.intrinsics_io import load_calib_result
from phase7_eye_to_hand.src.charuco_pose_estimator import CharucoPoseEstimator
from phase7_eye_to_hand.src.handeye_solver import solve_eye_to_hand
from phase7_eye_to_hand.src.io_utils import (
    load_sample_pairs_jsonl,
    save_capture_records_jsonl,
    load_capture_records_jsonl,
    now_iso,
    save_result_json,
    save_sample_pairs_jsonl,
    write_t_matrix_npy,
)
from phase7_eye_to_hand.src.validation import validate_translation_error
from phase7_eye_to_hand.src.robot_pose_parser import RobotPoseSample
from phase7_eye_to_hand.src.sample_capture import (
    capture_samples_only,
    analyze_samples,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_pairs_to_robot_samples(pairs) -> list[RobotPoseSample]:
    """Convert SamplePair objects with embedded robot state to RobotPoseSample."""
    robot_samples = []
    for pair in pairs:
        if pair.R_gripper2base is None or pair.t_gripper2base is None:
            raise ValueError(
                f"Sample {pair.sample_index} missing robot state. "
                "Ensure capture was done with real-time robot state fetching."
            )
        robot_samples.append(
            RobotPoseSample(
                index=pair.sample_index,
                raw_row=[],  # No CSV row for real-time captures
                t_gripper2base=pair.t_gripper2base.astype(np.float64),
                R_gripper2base=pair.R_gripper2base.astype(np.float64),
            )
        )
    return robot_samples


def _estimate_target_offset_gripper_m(
    T_cam2base: np.ndarray,
    robot_samples: list[RobotPoseSample],
    pairs,
) -> np.ndarray:
    """Estimate constant target offset in gripper frame by least squares.

    Model:
      R_g2b * off + t_g2b ~= R_c2b * t_t2c + t_c2b
    where `off` is target origin in gripper frame.
    """
    if len(robot_samples) != len(pairs):
        raise ValueError("robot_samples and pairs must have same length for offset estimation")
    if len(pairs) < 3:
        raise ValueError("need at least 3 samples to estimate target offset")

    A_blocks = []
    b_blocks = []
    for rs, pair in zip(robot_samples, pairs):
        p_from_cam = T_cam2base[:3, :3] @ np.asarray(pair.t_target2cam, dtype=np.float64).reshape(3)
        p_from_cam = p_from_cam + T_cam2base[:3, 3]
        A_blocks.append(np.asarray(rs.R_gripper2base, dtype=np.float64))
        b_blocks.append((p_from_cam - np.asarray(rs.t_gripper2base, dtype=np.float64).reshape(3)).reshape(3, 1))

    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)
    off, *_ = np.linalg.lstsq(A, b, rcond=None)
    return off.reshape(3).astype(np.float64)


def _evaluate_method(
    method: str,
    robot_samples: list[RobotPoseSample],
    pairs,
    configured_offset: np.ndarray,
) -> dict:
    """Solve hand-eye and return method metrics for comparison."""
    res = solve_eye_to_hand(
        robot_samples=robot_samples,
        sample_pairs=pairs,
        method=method,
    )

    uncompensated_offset = np.zeros(3, dtype=np.float64)
    val_stats_uncomp, errors_mm_uncomp = validate_translation_error(
        T_cam2base=res.T_cam2base,
        robot_samples=robot_samples,
        pairs=pairs,
        target_offset_gripper_m=uncompensated_offset,
    )

    target_offset = configured_offset
    offset_source = "configured"
    try:
        target_offset = _estimate_target_offset_gripper_m(res.T_cam2base, robot_samples, pairs)
        offset_source = "estimated_from_samples"
    except Exception:
        pass

    val_stats_comp, errors_mm_comp = validate_translation_error(
        T_cam2base=res.T_cam2base,
        robot_samples=robot_samples,
        pairs=pairs,
        target_offset_gripper_m=target_offset,
    )

    improvement_mm = val_stats_uncomp.mean_mm - val_stats_comp.mean_mm
    improvement_pct = (improvement_mm / val_stats_uncomp.mean_mm * 100.0) if val_stats_uncomp.mean_mm > 1e-9 else 0.0

    return {
        "method": method,
        "res": res,
        "target_offset": target_offset,
        "offset_source": offset_source,
        "val_stats_uncomp": val_stats_uncomp,
        "errors_mm_uncomp": errors_mm_uncomp,
        "val_stats_comp": val_stats_comp,
        "errors_mm_comp": errors_mm_comp,
        "improvement_mm": improvement_mm,
        "improvement_pct": improvement_pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase7 Eye-to-Hand calibration")
    parser.add_argument(
        "--mode",
        choices=["capture", "analyze", "solve", "all"],
        default="all",
        help="capture: only capture images+robot state | analyze: analyze from captures | solve: solve from samples | all: full pipeline"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    p7 = cfg.get("phase7_eye_to_hand", {})

    camera_name = p7.get("camera_name", "cam0")
    camera_idx = cfg["cameras"][camera_name]["index"]
    intrinsics_path = os.path.join(_ROOT, "phase1_intrinsics", "outputs", "intrinsics.json")

    out_base = os.path.join(_ROOT, "phase7_eye_to_hand", "outputs")
    captures_dir = os.path.join(out_base, "captures")
    captures_jsonl = os.path.join(out_base, "captures.jsonl")
    sample_jsonl = os.path.join(out_base, "samples", "sample_pairs.jsonl")
    sample_img_dir = os.path.join(out_base, "samples", "images")
    result_json = os.path.join(out_base, "eye_to_hand_result.json")
    result_npy = os.path.join(out_base, "T_cam2arm.npy")
    report_json = os.path.join(out_base, "reports", "validation_report.json")

    calib = load_calib_result(camera_name, intrinsics_path)
    if calib is None:
        raise ValueError(f"intrinsics for {camera_name} not found: {intrinsics_path}")

    estimator = CharucoPoseEstimator(
        charuco_cfg=cfg["calibration"]["charuco"],
        K=calib.K,
        D=calib.D,
        min_corners=int(p7.get("min_charuco_corners", 6)),
    )

    # === CAPTURE PHASE ===
    if args.mode in ("capture", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: CAPTURE (images + robot state only)")
        logger.info("=" * 60)
        server_url = p7.get("robot_server_url", "http://140.118.117.61:5000/get_status")
        records = capture_samples_only(
            camera_index=camera_idx,
            out_dir=captures_dir,
            warmup_sec=float(p7.get("camera_warmup_sec", 1.0)),
            server_url=server_url,
        )
        save_capture_records_jsonl(records, captures_jsonl)
        logger.info(f"✓ Captured: {len(records)} records -> {captures_jsonl}\n")

        if args.mode == "capture":
            return

    # === ANALYZE PHASE ===
    if args.mode in ("analyze", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: ANALYZE (estimate Charuco poses from captures)")
        logger.info("=" * 60)
        if not os.path.exists(captures_jsonl):
            raise ValueError(f"captures.jsonl not found: {captures_jsonl}\nRun with --mode capture first")
        
        pairs = analyze_samples(estimator=estimator, captures_path=captures_jsonl)
        save_sample_pairs_jsonl(pairs, sample_jsonl)
        logger.info(f"✓ Analyzed: {len(pairs)} pairs -> {sample_jsonl}\n")

        if args.mode == "analyze":
            return

    # === SOLVE PHASE ===
    if args.mode in ("solve", "all"):
        logger.info("=" * 60)
        logger.info("PHASE: SOLVE (hand-eye calibration)")
        logger.info("=" * 60)
        pairs = load_sample_pairs_jsonl(sample_jsonl)
        min_samples = int(p7.get("min_samples", 15))
        if len(pairs) < min_samples:
            raise ValueError(f"need at least {min_samples} pairs, got {len(pairs)}")

        # Convert sample pairs to robot samples
        robot_samples = _sample_pairs_to_robot_samples(pairs)
        configured_offset = np.asarray(p7.get("target_offset_gripper_m", [0.0, 0.0, 0.0]), dtype=np.float64)

        candidate_methods = ["park", "daniilidis", "horaud"]
        method_trials = []
        best_eval = None

        for method in candidate_methods:
            try:
                trial = _evaluate_method(method, robot_samples, pairs, configured_offset)
                method_trials.append(
                    {
                        "method": method,
                        "success": True,
                        "mean_mm_uncompensated": trial["val_stats_uncomp"].mean_mm,
                        "mean_mm_compensated": trial["val_stats_comp"].mean_mm,
                        "improvement_percent": trial["improvement_pct"],
                    }
                )
                logger.info(
                    "Method %s: mean error %.2f mm -> %.2f mm",
                    method,
                    trial["val_stats_uncomp"].mean_mm,
                    trial["val_stats_comp"].mean_mm,
                )

                if best_eval is None or trial["val_stats_comp"].mean_mm < best_eval["val_stats_comp"].mean_mm:
                    best_eval = trial
            except Exception as exc:
                method_trials.append({"method": method, "success": False, "error": str(exc)})
                logger.warning("Method %s failed: %s", method, exc)

        if best_eval is None:
            raise RuntimeError("all candidate methods failed: park, daniilidis, horaud")

        res = best_eval["res"]
        target_offset = best_eval["target_offset"]
        offset_source = best_eval["offset_source"]
        val_stats_uncomp = best_eval["val_stats_uncomp"]
        errors_mm_uncomp = best_eval["errors_mm_uncomp"]
        val_stats_comp = best_eval["val_stats_comp"]
        errors_mm_comp = best_eval["errors_mm_comp"]
        improvement_mm = best_eval["improvement_mm"]
        improvement_pct = best_eval["improvement_pct"]

        logger.info(
            "Selected best method: %s (compensated mean %.2f mm)",
            res.method,
            val_stats_comp.mean_mm,
        )
        logger.info(
            "Estimated target_offset_gripper_m: %s (norm=%.1f mm, source=%s)",
            np.round(target_offset, 6).tolist(),
            float(np.linalg.norm(target_offset) * 1000.0),
            offset_source,
        )

        payload = {
            "timestamp": now_iso(),
            "camera_name": camera_name,
            "camera_index": camera_idx,
            "method": res.method,
            "num_pairs": len(pairs),
            "T_cam2arm": res.T_cam2base,
            "method_candidates": candidate_methods,
            "method_trials": method_trials,
            "selected_method": res.method,
            "target_offset_gripper_m": target_offset,
            "target_offset_gripper_m_config": configured_offset,
            "offset_source": offset_source,
            "validation_uncompensated": {
                "mean_mm": val_stats_uncomp.mean_mm,
                "median_mm": val_stats_uncomp.median_mm,
                "p95_mm": val_stats_uncomp.p95_mm,
                "max_mm": val_stats_uncomp.max_mm,
                "count": val_stats_uncomp.count,
            },
            "validation_compensated": {
                "mean_mm": val_stats_comp.mean_mm,
                "median_mm": val_stats_comp.median_mm,
                "p95_mm": val_stats_comp.p95_mm,
                "max_mm": val_stats_comp.max_mm,
                "count": val_stats_comp.count,
            },
            "validation_improvement": {
                "mean_mm_delta": improvement_mm,
                "mean_mm_delta_percent": improvement_pct,
            },
        }

        save_result_json(result_json, payload)
        save_result_json(
            report_json,
            {
                "errors_mm_uncompensated": errors_mm_uncomp,
                "summary_uncompensated": payload["validation_uncompensated"],
                "errors_mm_compensated": errors_mm_comp,
                "summary_compensated": payload["validation_compensated"],
                "improvement": payload["validation_improvement"],
            },
        )
        write_t_matrix_npy(result_npy, res.T_cam2base)

        logger.info("✓ SOLVE complete")
        logger.info(f"  T_cam2arm: {result_json}")
        logger.info(f"  matrix npy: {result_npy}")
        logger.info(
            "  mean error (uncompensated -> compensated): %.2f mm -> %.2f mm (improve %.2f%%)",
            val_stats_uncomp.mean_mm,
            val_stats_comp.mean_mm,
            improvement_pct,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Phase7 pipeline complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
