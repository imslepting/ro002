"""Phase 0 — 硬體診斷 tkinter GUI

2x2 即時相機畫面 + 右側狀態面板 + 分辨率下拉選擇 + 按鈕操作
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from shared.types import CameraTestResult
from shared.camera_manager import CameraReader
from shared.tk_utils import (
    DARK_BG, PANEL_BG, ACCENT_GREEN, ACCENT_RED, ACCENT_YELLOW,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    FONT_TITLE, FONT_BODY, FONT_SMALL, FONT_MONO,
    BTN_STYLE, BTN_ACCENT, BTN_DANGER,
    cv_to_photoimage, CameraFeedWidget, probe_resolutions, run_in_thread,
)
from phase0_hw_diagnostics.src.camera_tester import test_camera


def _overlay_info(
    frame: np.ndarray,
    cam_name: str,
    role: str,
    result: CameraTestResult | None,
) -> np.ndarray:
    """在畫面左上角疊加相機資訊"""
    img = frame.copy()
    status = result.status if result else "ERROR"
    border_colors = {
        "OK": (0, 200, 0),
        "WARNING": (0, 200, 255),
        "ERROR": (0, 0, 220),
    }
    color = border_colors.get(status, (128, 128, 128))

    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 3)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    line1 = f"{cam_name} [{role}]"
    if result and result.resolution[0] > 0:
        line2 = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
    else:
        line2 = "---"

    cv2.putText(img, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def run_tk_display(
    camera_configs: dict,
    test_results: dict[str, CameraTestResult | None],
    tile_size: tuple[int, int] = (640, 480),
    on_save: callable = None,
    on_rescan: callable = None,
) -> None:
    """啟動 tkinter 版即時多相機顯示窗口"""

    tile_w, tile_h = tile_size
    cam_names = sorted(camera_configs.keys())

    # ── 啟動相機讀取線程 ──
    readers: dict[str, CameraReader] = {}
    for name in cam_names:
        idx = camera_configs[name]["index"]
        reader = CameraReader(idx)
        reader.start()
        readers[name] = reader

    # ── 建立 tkinter 窗口 ──
    root = tk.Tk()
    root.title("RO002 Hardware Diagnostics")
    root.configure(bg=DARK_BG)
    root.protocol("WM_DELETE_WINDOW", lambda: _quit(root, readers))

    # ── 主佈局：左=相機網格，右=面板 ──
    main_frame = tk.Frame(root, bg=DARK_BG)
    main_frame.pack(fill="both", expand=True)

    grid_frame = tk.Frame(main_frame, bg=DARK_BG)
    grid_frame.pack(side="left", padx=5, pady=5)

    panel_frame = tk.Frame(main_frame, bg=PANEL_BG, width=380)
    panel_frame.pack(side="right", fill="y", padx=(0, 5), pady=5)
    panel_frame.pack_propagate(False)

    # ── 探測分辨率（背景） ──
    resolutions: dict[str, list[tuple[int, int]]] = {}
    resolution_combos: dict[str, ttk.Combobox] = {}

    # ── 建立 2x2 相機網格 ──
    feeds: dict[str, CameraFeedWidget] = {}
    for idx_pos, name in enumerate(cam_names[:4]):
        row, col = divmod(idx_pos, 2)
        role = camera_configs[name].get("role", "")
        result = test_results.get(name)

        def make_overlay(n=name, r=role):
            def _overlay(frame, _n=n, _r=r):
                res = test_results.get(_n)
                return _overlay_info(frame, _n, _r, res)
            return _overlay

        feed = CameraFeedWidget(
            grid_frame,
            reader=readers.get(name),
            display_size=(tile_w, tile_h),
            overlay_fn=make_overlay(),
        )
        feed.grid(row=row, column=col, padx=2, pady=2)
        feed.start_feed()
        feeds[name] = feed

    # ── 右側面板 ──
    _build_status_panel(
        panel_frame, cam_names, camera_configs, test_results,
        readers, feeds, resolutions, resolution_combos,
        on_save, on_rescan, root, tile_w, tile_h,
    )

    # 分辨率下拉默認顯示當前值（不做探測）
    for name in cam_names:
        combo = resolution_combos.get(name)
        if combo:
            current = test_results.get(name)
            if current and current.resolution[0] > 0:
                cur_str = f"{current.resolution[0]}x{current.resolution[1]}"
                combo["values"] = [cur_str]
                combo.set(cur_str)
            else:
                combo.set("N/A")

    root.mainloop()


def _build_status_panel(
    parent, cam_names, camera_configs, test_results,
    readers, feeds, resolutions, resolution_combos,
    on_save, on_rescan, root, tile_w, tile_h,
):
    """建立右側狀態面板"""
    # Title
    tk.Label(
        parent, text="DEVICE STATUS", font=FONT_TITLE,
        bg=PANEL_BG, fg=TEXT_PRIMARY,
    ).pack(pady=(15, 5))
    ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=15, pady=5)

    # Camera status frames
    cam_status_frame = tk.Frame(parent, bg=PANEL_BG)
    cam_status_frame.pack(fill="x", padx=15, pady=5)

    status_labels: dict[str, dict] = {}

    for name in cam_names:
        cam_info = camera_configs[name]
        result = test_results.get(name)
        role = cam_info.get("role", "")

        frame = tk.Frame(cam_status_frame, bg=PANEL_BG)
        frame.pack(fill="x", pady=4)

        # Status dot
        dot_canvas = tk.Canvas(frame, width=18, height=18, bg=PANEL_BG,
                               highlightthickness=0)
        dot_canvas.pack(side="left", padx=(0, 6))

        if result is None:
            dot_color = ACCENT_RED
            status_text = "NOT FOUND"
        elif result.status == "OK":
            dot_color = ACCENT_GREEN
            status_text = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
        elif result.status == "WARNING":
            dot_color = ACCENT_YELLOW
            status_text = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
        else:
            dot_color = ACCENT_RED
            status_text = "ERROR"

        dot_canvas.create_oval(3, 3, 15, 15, fill=dot_color, outline="")

        info_frame = tk.Frame(frame, bg=PANEL_BG)
        info_frame.pack(side="left", fill="x", expand=True)

        tk.Label(
            info_frame, text=f"{name} [{role}]",
            font=FONT_BODY, bg=PANEL_BG, fg=TEXT_PRIMARY, anchor="w",
        ).pack(anchor="w")

        detail_label = tk.Label(
            info_frame, text=status_text,
            font=FONT_SMALL, bg=PANEL_BG, fg=TEXT_SECONDARY, anchor="w",
        )
        detail_label.pack(anchor="w")

        # Warnings
        warn_label = tk.Label(
            info_frame, text="", font=FONT_SMALL, bg=PANEL_BG,
            fg=ACCENT_YELLOW, anchor="w", wraplength=280, justify="left",
        )
        if result and result.warnings:
            warn_label.configure(text="\n".join(f"! {w}" for w in result.warnings))
        warn_label.pack(anchor="w")

        # Resolution combobox
        res_frame = tk.Frame(info_frame, bg=PANEL_BG)
        res_frame.pack(anchor="w", pady=(2, 0))
        tk.Label(res_frame, text="Resolution:", font=FONT_SMALL,
                 bg=PANEL_BG, fg=TEXT_DIM).pack(side="left")
        combo = ttk.Combobox(res_frame, width=12, state="readonly", font=FONT_SMALL)
        combo.set("probing...")
        combo.pack(side="left", padx=4)
        resolution_combos[name] = combo

        # Bind resolution change
        def _on_res_change(event, cam_name=name):
            _change_resolution(
                cam_name, camera_configs, resolution_combos, resolutions,
                readers, feeds, test_results, root, tile_w, tile_h,
                status_labels,
            )
        combo.bind("<<ComboboxSelected>>", _on_res_change)

        status_labels[name] = {
            "dot_canvas": dot_canvas,
            "detail_label": detail_label,
            "warn_label": warn_label,
        }

    # Separator
    ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=15, pady=10)

    # Overall status
    summary_frame = tk.Frame(parent, bg=PANEL_BG)
    summary_frame.pack(fill="x", padx=15)

    n_expected = len(cam_names)
    n_ok = sum(1 for r in test_results.values() if r and r.status == "OK")
    n_err = sum(1 for r in test_results.values() if r is None or r.status == "ERROR")

    if n_err == 0 and n_ok == n_expected:
        overall = "READY"
        overall_color = ACCENT_GREEN
    elif n_err == 0:
        overall = "WARNINGS"
        overall_color = ACCENT_YELLOW
    else:
        overall = "ISSUES DETECTED"
        overall_color = ACCENT_RED

    camera_count_label = tk.Label(
        summary_frame, text=f"Cameras: {n_ok}/{n_expected} OK",
        font=FONT_BODY, bg=PANEL_BG, fg=TEXT_SECONDARY,
    )
    camera_count_label.pack(anchor="w")

    overall_label = tk.Label(
        summary_frame, text=overall,
        font=FONT_TITLE, bg=PANEL_BG, fg=overall_color,
    )
    overall_label.pack(anchor="w", pady=(2, 0))

    # Store labels for updates
    status_labels["__summary__"] = {
        "camera_count": camera_count_label,
        "overall": overall_label,
    }

    # Separator
    ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=15, pady=10)

    # Buttons
    btn_frame = tk.Frame(parent, bg=PANEL_BG)
    btn_frame.pack(fill="x", padx=15, pady=5)

    def _do_save():
        if on_save:
            frames = {}
            for name in cam_names:
                r = readers.get(name)
                if r and r.frame is not None:
                    frames[name] = r.frame.copy()
            on_save(frames)

    def _do_rescan():
        if not on_rescan:
            return
        # Disable button, show scanning state
        rescan_btn.configure(text="Scanning...", state="disabled")

        def _scan():
            return on_rescan()

        def _on_done(new_results):
            test_results.clear()
            test_results.update(new_results)
            _refresh_status_labels(
                cam_names, camera_configs, test_results, status_labels,
            )
            rescan_btn.configure(text="Re-scan", state="normal")

        run_in_thread(_scan, _on_done, root)

    tk.Button(btn_frame, text="Save Report", command=_do_save, **BTN_ACCENT).pack(
        fill="x", pady=3,
    )
    rescan_btn = tk.Button(btn_frame, text="Re-scan", command=_do_rescan, **BTN_STYLE)
    rescan_btn.pack(fill="x", pady=3)
    tk.Button(
        btn_frame, text="Quit",
        command=lambda: _quit(root, readers),
        **BTN_DANGER,
    ).pack(fill="x", pady=3)


def _change_resolution(
    cam_name, camera_configs, resolution_combos, resolutions,
    readers, feeds, test_results, root, tile_w, tile_h,
    status_labels,
):
    """切換相機分辨率"""
    combo = resolution_combos[cam_name]
    val = combo.get()
    if not val or "x" not in val:
        return
    w, h = val.split("x")
    new_res = (int(w), int(h))
    cam_index = camera_configs[cam_name]["index"]

    # 立即更新 overlay 顯示的分辨率（先用選中值，等重測後會刷成實測值）
    old_result = test_results.get(cam_name)
    if old_result is not None:
        from dataclasses import replace
        test_results[cam_name] = replace(old_result, resolution=new_res)
        cam_names = sorted(camera_configs.keys())
        _refresh_status_labels(cam_names, camera_configs, test_results, status_labels)

    # Stop old reader
    old_reader = readers.get(cam_name)
    if old_reader:
        old_reader.stop()

    # Create new reader with resolution
    new_reader = CameraReader(cam_index, resolution=new_res)
    new_reader.start()
    readers[cam_name] = new_reader

    # Update feed widget
    feed = feeds.get(cam_name)
    if feed:
        feed.set_reader(new_reader)

    # Re-test camera in background (需要先停 reader 釋放設備)
    def _retest():
        readers[cam_name].stop()
        result = test_camera(cam_index, n_frames=30, resolution=new_res)
        # 重啟 reader with requested resolution
        restarted = CameraReader(cam_index, resolution=new_res)
        restarted.start()
        readers[cam_name] = restarted
        return result

    def _on_tested(result):
        test_results[cam_name] = result
        # 更新 feed 的 reader 引用
        feed = feeds.get(cam_name)
        if feed:
            feed.set_reader(readers[cam_name])
        cam_names = sorted(camera_configs.keys())
        _refresh_status_labels(cam_names, camera_configs, test_results, status_labels)

    run_in_thread(_retest, _on_tested, root)


def _refresh_status_labels(cam_names, camera_configs, test_results, status_labels):
    """刷新狀態面板中的所有標籤"""
    for name in cam_names:
        result = test_results.get(name)
        labels = status_labels.get(name)
        if not labels:
            continue

        dot_canvas = labels["dot_canvas"]
        detail_label = labels["detail_label"]
        warn_label = labels["warn_label"]

        if result is None:
            dot_color = ACCENT_RED
            status_text = "NOT FOUND"
        elif result.status == "OK":
            dot_color = ACCENT_GREEN
            status_text = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
        elif result.status == "WARNING":
            dot_color = ACCENT_YELLOW
            status_text = f"{result.resolution[0]}x{result.resolution[1]} {result.fps_measured:.0f}fps"
        else:
            dot_color = ACCENT_RED
            status_text = "ERROR"

        dot_canvas.delete("all")
        dot_canvas.create_oval(3, 3, 15, 15, fill=dot_color, outline="")
        detail_label.configure(text=status_text)
        if result and result.warnings:
            warn_label.configure(text="\n".join(f"! {w}" for w in result.warnings))
        else:
            warn_label.configure(text="")

    # Update summary
    summary = status_labels.get("__summary__")
    if summary:
        n_expected = len(cam_names)
        n_ok = sum(1 for r in test_results.values() if r and r.status == "OK")
        n_err = sum(1 for r in test_results.values() if r is None or r.status == "ERROR")

        if n_err == 0 and n_ok == n_expected:
            overall = "READY"
            overall_color = ACCENT_GREEN
        elif n_err == 0:
            overall = "WARNINGS"
            overall_color = ACCENT_YELLOW
        else:
            overall = "ISSUES DETECTED"
            overall_color = ACCENT_RED

        summary["camera_count"].configure(text=f"Cameras: {n_ok}/{n_expected} OK")
        summary["overall"].configure(text=overall, fg=overall_color)


def _quit(root, readers):
    """停止所有 reader 並關閉窗口"""
    for reader in readers.values():
        reader.stop()
    root.destroy()
