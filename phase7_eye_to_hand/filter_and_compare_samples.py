"""Filter Phase7 sample pairs and compare calibration error before/after.

Usage:
    conda run -n ro002 python phase7_eye_to_hand/filter_and_compare_samples.py

Default input/output:
    input : phase7_eye_to_hand/outputs/samples/sample_pairs.jsonl
    clean : phase7_eye_to_hand/outputs/samples/sample_pairs.clean.jsonl
    report: phase7_eye_to_hand/outputs/reports/filter_compare_report.json
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from phase7_eye_to_hand.src.handeye_solver import solve_eye_to_hand
from phase7_eye_to_hand.src.io_utils import (
    SamplePair,
    load_sample_pairs_jsonl,
    now_iso,
    save_result_json,
    save_sample_pairs_jsonl,
)
from phase7_eye_to_hand.src.robot_pose_parser import RobotPoseSample
from phase7_eye_to_hand.src.validation import validate_translation_error


@dataclass
class PairQuality:
    index: int
    keep: bool
    reasons: list[str]


def _sample_pairs_to_robot_samples(pairs: list[SamplePair]) -> list[RobotPoseSample]:
    robot_samples: list[RobotPoseSample] = []
    for pair in pairs:
        if pair.R_gripper2base is None or pair.t_gripper2base is None:
            raise ValueError(f"sample {pair.sample_index} missing robot state")
        robot_samples.append(
            RobotPoseSample(
                index=pair.sample_index,
                raw_row=[],
                t_gripper2base=pair.t_gripper2base.astype(np.float64),
                R_gripper2base=pair.R_gripper2base.astype(np.float64),
            )
        )
    return robot_samples


def _rot_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    dR = R_a.T @ R_b
    tr = float(np.trace(dR))
    cos_theta = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _estimate_target_offset_gripper_m(
    T_cam2base: np.ndarray,
    robot_samples: list[RobotPoseSample],
    pairs: list[SamplePair],
) -> np.ndarray:
    if len(robot_samples) != len(pairs):
        raise ValueError("robot_samples and pairs must have same length")
    if len(pairs) < 3:
        raise ValueError("need at least 3 samples to estimate offset")

    A_blocks: list[np.ndarray] = []
    b_blocks: list[np.ndarray] = []
    for rs, pair in zip(robot_samples, pairs):
        p_from_cam = T_cam2base[:3, :3] @ np.asarray(pair.t_target2cam, dtype=np.float64).reshape(3)
        p_from_cam = p_from_cam + T_cam2base[:3, 3]
        A_blocks.append(np.asarray(rs.R_gripper2base, dtype=np.float64))
        b_blocks.append((p_from_cam - np.asarray(rs.t_gripper2base, dtype=np.float64).reshape(3)).reshape(3, 1))

    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)
    off, *_ = np.linalg.lstsq(A, b, rcond=None)
    return off.reshape(3).astype(np.float64)


def _evaluate_best(
    pairs: list[SamplePair],
    methods: list[str],
) -> dict:
    robot_samples = _sample_pairs_to_robot_samples(pairs)

    best_eval = None
    method_trials: list[dict] = []
    for method in methods:
        try:
            res = solve_eye_to_hand(
                robot_samples=robot_samples,
                sample_pairs=pairs,
                method=method,
            )

            stats_uncomp, errs_uncomp = validate_translation_error(
                T_cam2base=res.T_cam2base,
                robot_samples=robot_samples,
                pairs=pairs,
                target_offset_gripper_m=np.zeros(3, dtype=np.float64),
            )

            try:
                target_offset = _estimate_target_offset_gripper_m(res.T_cam2base, robot_samples, pairs)
                offset_source = "estimated_from_samples"
            except Exception:
                target_offset = np.zeros(3, dtype=np.float64)
                offset_source = "fallback_zero"

            stats_comp, errs_comp = validate_translation_error(
                T_cam2base=res.T_cam2base,
                robot_samples=robot_samples,
                pairs=pairs,
                target_offset_gripper_m=target_offset,
            )

            method_trials.append(
                {
                    "method": method,
                    "success": True,
                    "mean_mm_uncompensated": stats_uncomp.mean_mm,
                    "mean_mm_compensated": stats_comp.mean_mm,
                }
            )

            trial = {
                "method": method,
                "result": res,
                "target_offset_gripper_m": target_offset,
                "offset_source": offset_source,
                "stats_uncomp": stats_uncomp,
                "errors_uncomp": errs_uncomp,
                "stats_comp": stats_comp,
                "errors_comp": errs_comp,
                "method_trials": method_trials,
            }
            if best_eval is None or stats_comp.mean_mm < best_eval["stats_comp"].mean_mm:
                best_eval = trial
        except Exception as exc:
            method_trials.append({"method": method, "success": False, "error": str(exc)})

    if best_eval is None:
        raise RuntimeError("all candidate methods failed")
    return best_eval


def _filter_pairs(
    pairs: list[SamplePair],
    min_corners: int,
    max_rot_jump_deg: float,
    min_rot_jump_trans_m: float,
    max_trans_jump_m: float,
    min_trans_jump_rot_deg: float,
) -> tuple[list[SamplePair], list[PairQuality]]:
    if not pairs:
        return [], []

    quality: list[PairQuality] = []
    keep_mask = [True] * len(pairs)

    for i, p in enumerate(pairs):
        reasons: list[str] = []
        if int(p.num_corners) < min_corners:
            reasons.append(f"low_corners<{min_corners} ({p.num_corners})")

        if i > 0:
            prev = pairs[i - 1]
            R_prev = np.asarray(prev.R_gripper2base, dtype=np.float64)
            R_cur = np.asarray(p.R_gripper2base, dtype=np.float64)
            t_prev = np.asarray(prev.t_gripper2base, dtype=np.float64).reshape(3)
            t_cur = np.asarray(p.t_gripper2base, dtype=np.float64).reshape(3)

            rot_jump = _rot_delta_deg(R_prev, R_cur)
            trans_jump = float(np.linalg.norm(t_cur - t_prev))

            if rot_jump > max_rot_jump_deg and trans_jump < min_rot_jump_trans_m:
                reasons.append(
                    f"rot_jump>{max_rot_jump_deg}deg_with_small_trans<{min_rot_jump_trans_m}m "
                    f"({rot_jump:.2f}deg/{trans_jump:.4f}m)"
                )
            if trans_jump > max_trans_jump_m and rot_jump < min_trans_jump_rot_deg:
                reasons.append(
                    f"trans_jump>{max_trans_jump_m}m_with_small_rot<{min_trans_jump_rot_deg}deg "
                    f"({trans_jump:.4f}m/{rot_jump:.2f}deg)"
                )

        keep = len(reasons) == 0
        keep_mask[i] = keep
        quality.append(PairQuality(index=int(p.sample_index), keep=keep, reasons=reasons))

    clean_pairs = [p for p, keep in zip(pairs, keep_mask) if keep]
    return clean_pairs, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter phase7 samples and compare before/after errors")
    parser.add_argument(
        "--input",
        default=os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "samples", "sample_pairs.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "samples", "sample_pairs.clean.jsonl"),
    )
    parser.add_argument(
        "--report",
        default=os.path.join(_ROOT, "phase7_eye_to_hand", "outputs", "reports", "filter_compare_report.json"),
    )
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--max-rot-jump-deg", type=float, default=55.0)
    parser.add_argument("--min-rot-jump-trans-m", type=float, default=0.01)
    parser.add_argument("--max-trans-jump-m", type=float, default=0.15)
    parser.add_argument("--min-trans-jump-rot-deg", type=float, default=2.0)
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--methods", nargs="+", default=["park", "daniilidis", "horaud"])
    return parser.parse_args()


def _summary_from_eval(e: dict) -> dict:
    stats_un = e["stats_uncomp"]
    stats_cp = e["stats_comp"]
    delta = stats_un.mean_mm - stats_cp.mean_mm
    delta_pct = (delta / stats_un.mean_mm * 100.0) if stats_un.mean_mm > 1e-9 else 0.0
    return {
        "selected_method": e["method"],
        "offset_source": e["offset_source"],
        "target_offset_gripper_m": e["target_offset_gripper_m"],
        "validation_uncompensated": {
            "mean_mm": stats_un.mean_mm,
            "median_mm": stats_un.median_mm,
            "p95_mm": stats_un.p95_mm,
            "max_mm": stats_un.max_mm,
            "count": stats_un.count,
        },
        "validation_compensated": {
            "mean_mm": stats_cp.mean_mm,
            "median_mm": stats_cp.median_mm,
            "p95_mm": stats_cp.p95_mm,
            "max_mm": stats_cp.max_mm,
            "count": stats_cp.count,
        },
        "improvement": {
            "mean_mm_delta": delta,
            "mean_mm_delta_percent": delta_pct,
        },
        "method_trials": e["method_trials"],
    }


def main() -> None:
    args = parse_args()

    pairs = load_sample_pairs_jsonl(args.input)
    clean_pairs, quality = _filter_pairs(
        pairs=pairs,
        min_corners=args.min_corners,
        max_rot_jump_deg=args.max_rot_jump_deg,
        min_rot_jump_trans_m=args.min_rot_jump_trans_m,
        max_trans_jump_m=args.max_trans_jump_m,
        min_trans_jump_rot_deg=args.min_trans_jump_rot_deg,
    )

    if len(clean_pairs) < args.min_samples:
        raise ValueError(
            f"filtered samples too few: {len(clean_pairs)} < {args.min_samples}; adjust thresholds"
        )

    eval_before = _evaluate_best(pairs, args.methods)
    eval_after = _evaluate_best(clean_pairs, args.methods)

    removed = [
        {
            "sample_index": q.index,
            "reasons": q.reasons,
        }
        for q in quality
        if not q.keep
    ]

    report = {
        "timestamp": now_iso(),
        "input_jsonl": args.input,
        "output_jsonl": args.output,
        "total_samples": len(pairs),
        "kept_samples": len(clean_pairs),
        "removed_samples": len(removed),
        "filter_rules": {
            "min_corners": args.min_corners,
            "max_rot_jump_deg": args.max_rot_jump_deg,
            "min_rot_jump_trans_m": args.min_rot_jump_trans_m,
            "max_trans_jump_m": args.max_trans_jump_m,
            "min_trans_jump_rot_deg": args.min_trans_jump_rot_deg,
        },
        "removed_detail": removed,
        "before": _summary_from_eval(eval_before),
        "after": _summary_from_eval(eval_after),
        "comparison": {
            "uncomp_mean_mm_delta_after_minus_before": (
                _summary_from_eval(eval_after)["validation_uncompensated"]["mean_mm"]
                - _summary_from_eval(eval_before)["validation_uncompensated"]["mean_mm"]
            ),
            "comp_mean_mm_delta_after_minus_before": (
                _summary_from_eval(eval_after)["validation_compensated"]["mean_mm"]
                - _summary_from_eval(eval_before)["validation_compensated"]["mean_mm"]
            ),
        },
    }

    save_sample_pairs_jsonl(clean_pairs, args.output)
    save_result_json(args.report, report)

    before_comp = report["before"]["validation_compensated"]["mean_mm"]
    after_comp = report["after"]["validation_compensated"]["mean_mm"]
    print(f"input samples={len(pairs)}, kept={len(clean_pairs)}, removed={len(removed)}")
    print(f"compensated mean error: before={before_comp:.3f} mm, after={after_comp:.3f} mm")
    print(f"clean jsonl: {args.output}")
    print(f"report json: {args.report}")


if __name__ == "__main__":
    main()
