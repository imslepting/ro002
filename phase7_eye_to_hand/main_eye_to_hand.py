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
from scipy.spatial.transform import Rotation

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
from phase7_eye_to_hand.src.validation import validate_target_in_gripper_consistency
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

VALID_EULER_ORDERS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def load_config(config_path: str = "config/settings.yaml") -> dict:
    path = os.path.join(_ROOT, config_path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_top_level_key_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return (not line.startswith((" ", "\t"))) and stripped.endswith(":")


def _write_t_cam2arm_to_config(config_path: str, T_cam2arm: np.ndarray) -> str:
    """Replace only arm.T_cam2arm block in YAML config to preserve other comments/format."""
    abs_path = os.path.join(_ROOT, config_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    arm_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "arm:":
            arm_idx = i
            break
    if arm_idx is None:
        raise ValueError(f"'arm:' section not found in config: {abs_path}")

    arm_end = len(lines)
    for i in range(arm_idx + 1, len(lines)):
        if _is_top_level_key_line(lines[i]):
            arm_end = i
            break

    t_idx = None
    for i in range(arm_idx + 1, arm_end):
        if lines[i].lstrip().startswith("T_cam2arm:"):
            t_idx = i
            break
    if t_idx is None:
        raise ValueError(f"'arm.T_cam2arm' not found in config: {abs_path}")

    key_indent = len(lines[t_idx]) - len(lines[t_idx].lstrip(" "))
    item_indent = key_indent + 2

    block_start = t_idx + 1
    block_end = block_start
    while block_end < arm_end:
        candidate = lines[block_end]
        stripped = candidate.strip()
        if not stripped:
            block_end += 1
            continue
        if candidate.startswith(" " * item_indent) and candidate.lstrip().startswith("-"):
            block_end += 1
            continue
        break

    T = np.asarray(T_cam2arm, dtype=np.float64).reshape(4, 4)
    new_block = []
    for row in T:
        a, b, c, d = [float(v) for v in row]
        new_block.append(
            f"{' ' * item_indent}- [{a:.16g}, {b:.16g}, {c:.16g}, {d:.16g}]\n"
        )

    lines[block_start:block_end] = new_block

    with open(abs_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return abs_path


def _normalize_euler_order(order: str) -> str:
    normalized = str(order).lower()
    if normalized not in VALID_EULER_ORDERS:
        raise ValueError(f"invalid euler_order='{order}', expected one of {VALID_EULER_ORDERS}")
    return normalized


def _sample_pairs_to_robot_samples(pairs, euler_order: str | None = None) -> list[RobotPoseSample]:
    """Convert SamplePair objects with embedded robot state to RobotPoseSample."""
    decoded_order = _normalize_euler_order(euler_order) if euler_order is not None else None
    robot_samples = []
    for pair in pairs:
        if pair.R_gripper2base is None or pair.t_gripper2base is None:
            raise ValueError(
                f"Sample {pair.sample_index} missing robot state. "
                "Ensure capture was done with real-time robot state fetching."
            )

        R_g2b = pair.R_gripper2base.astype(np.float64)
        if decoded_order is not None and pair.robot_euler_deg is not None:
            R_g2b = Rotation.from_euler(
                decoded_order,
                np.asarray(pair.robot_euler_deg, dtype=np.float64).reshape(3),
                degrees=True,
            ).as_matrix().astype(np.float64)

        robot_samples.append(
            RobotPoseSample(
                index=pair.sample_index,
                raw_row=[],  # No CSV row for real-time captures
                t_gripper2base=pair.t_gripper2base.astype(np.float64),
                R_gripper2base=R_g2b,
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


def _run_method_trials(
    robot_samples: list[RobotPoseSample],
    pairs,
    configured_offset: np.ndarray,
) -> tuple[dict, list[dict]]:
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
    return best_eval, method_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase7 Eye-to-Hand calibration")
    parser.add_argument(
        "--mode",
        choices=["capture", "analyze", "solve", "search_euler", "all"],
        default="all",
        help="capture: capture only | analyze: estimate Charuco poses | solve: solve hand-eye | search_euler: compare all Euler orders | all: full pipeline"
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
    euler_report_json = os.path.join(out_base, "reports", "euler_order_comparison.json")

    configured_euler_order = _normalize_euler_order(p7.get("euler_order", "xyz"))

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
        logger.info("Using Euler order for robot decode: %s", configured_euler_order)
        records = capture_samples_only(
            camera_index=camera_idx,
            out_dir=captures_dir,
            warmup_sec=float(p7.get("camera_warmup_sec", 1.0)),
            server_url=server_url,
            euler_order=configured_euler_order,
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

    # === SEARCH EULER PHASE ===
    if args.mode == "search_euler":
        logger.info("=" * 60)
        logger.info("PHASE: SEARCH_EULER (compare Euler orders)")
        logger.info("=" * 60)

        if not os.path.exists(sample_jsonl):
            raise ValueError(f"sample_pairs.jsonl not found: {sample_jsonl}\nRun with --mode analyze first")

        pairs = load_sample_pairs_jsonl(sample_jsonl)
        min_samples = int(p7.get("min_samples", 15))
        if len(pairs) < min_samples:
            raise ValueError(f"need at least {min_samples} pairs, got {len(pairs)}")

        configured_offset = np.asarray(p7.get("target_offset_gripper_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        comparisons = []
        best_entry = None

        for order in VALID_EULER_ORDERS:
            logger.info("Testing Euler order: %s", order)
            try:
                robot_samples = _sample_pairs_to_robot_samples(pairs, euler_order=order)
                best_eval, method_trials = _run_method_trials(robot_samples, pairs, configured_offset)
                pose_stats = validate_target_in_gripper_consistency(
                    T_cam2base=best_eval["res"].T_cam2base,
                    robot_samples=robot_samples,
                    pairs=pairs,
                )

                entry = {
                    "euler_order": order,
                    "success": True,
                    "selected_method": best_eval["res"].method,
                    "mean_mm_uncompensated": best_eval["val_stats_uncomp"].mean_mm,
                    "mean_mm_compensated": best_eval["val_stats_comp"].mean_mm,
                    "position_rmse_mm": pose_stats.position_rmse_mm,
                    "orientation_rmse_deg": pose_stats.orientation_rmse_deg,
                    "method_trials": method_trials,
                }
                comparisons.append(entry)

                logger.info(
                    "Order %s best=%s, position RMSE %.3f mm, orientation RMSE %.3f deg",
                    order,
                    entry["selected_method"],
                    entry["position_rmse_mm"],
                    entry["orientation_rmse_deg"],
                )

                if best_entry is None:
                    best_entry = entry
                else:
                    a = (entry["position_rmse_mm"], entry["orientation_rmse_deg"])
                    b = (best_entry["position_rmse_mm"], best_entry["orientation_rmse_deg"])
                    if a < b:
                        best_entry = entry
            except Exception as exc:
                comparisons.append({
                    "euler_order": order,
                    "success": False,
                    "error": str(exc),
                })
                logger.warning("Order %s failed: %s", order, exc)

        if best_entry is None:
            raise RuntimeError("all Euler-order trials failed")

        ranked = sorted(
            [c for c in comparisons if c.get("success")],
            key=lambda c: (c["position_rmse_mm"], c["orientation_rmse_deg"]),
        )

        save_result_json(
            euler_report_json,
            {
                "timestamp": now_iso(),
                "camera_name": camera_name,
                "num_pairs": len(pairs),
                "configured_euler_order": configured_euler_order,
                "best_euler_order": best_entry["euler_order"],
                "best_selected_method": best_entry["selected_method"],
                "ranking": ranked,
                "all_trials": comparisons,
            },
        )

        logger.info("✓ SEARCH_EULER complete")
        logger.info("  comparison report: %s", euler_report_json)
        logger.info(
            "  best order: %s (position RMSE %.3f mm, orientation RMSE %.3f deg)",
            best_entry["euler_order"],
            best_entry["position_rmse_mm"],
            best_entry["orientation_rmse_deg"],
        )
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
        robot_samples = _sample_pairs_to_robot_samples(pairs, euler_order=configured_euler_order)
        configured_offset = np.asarray(p7.get("target_offset_gripper_m", [0.0, 0.0, 0.0]), dtype=np.float64)
        best_eval, method_trials = _run_method_trials(robot_samples, pairs, configured_offset)
        candidate_methods = ["park", "daniilidis", "horaud"]

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
            "euler_order": configured_euler_order,
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
        config_written_path = _write_t_cam2arm_to_config(args.config, res.T_cam2base)

        logger.info("✓ SOLVE complete")
        logger.info(f"  T_cam2arm: {result_json}")
        logger.info(f"  matrix npy: {result_npy}")
        logger.info(f"  config updated: {config_written_path}")
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
