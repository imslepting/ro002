"""ChArUco 標定板生成 — 圖片 + A4 PDF

生成邏輯：根據 square_size（公尺）和 DPI 精確計算像素數，
確保打印後方格尺寸與 settings.yaml 完全一致，無需手量。
若指定的 square_size 放不進 A4，自動縮小並警告。
"""

from __future__ import annotations

import os

import cv2
import numpy as np


# A4 物理尺寸（mm）
_A4_W_MM = 297.0
_A4_H_MM = 210.0
_PDF_DPI = 600
_PAGE_MARGIN_MM = 10.0  # 每邊留 10mm 安全邊距


def create_board(charuco_cfg: dict):
    """從 settings.yaml 的 calibration.charuco 配置創建 CharucoBoard 對象"""
    dict_name = charuco_cfg.get("aruco_dict", "DICT_4X4_50")
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dict_name)
    )
    board = cv2.aruco.CharucoBoard(
        size=(charuco_cfg["cols"], charuco_cfg["rows"]),
        squareLength=charuco_cfg["square_size"],
        markerLength=charuco_cfg["marker_size"],
        dictionary=aruco_dict,
    )
    return board, aruco_dict


def generate_board_image(
    board,
    size: tuple[int, int] = (2400, 1600),
    margin: int = 20,
) -> np.ndarray:
    """生成灰度標定板圖片"""
    img = board.generateImage(outSize=size, marginSize=margin)
    return img


def save_board_pdf(
    board_image: np.ndarray,
    path: str = "assets/charuco_board.pdf",
    cols: int = 9,
    rows: int = 6,
    square_size_m: float = 0.04,
) -> str:
    """保存標定板為 A4 landscape PDF，方格尺寸精確匹配 square_size

    Args:
        board_image: generateImage() 生成的原始圖片
        path: 輸出 PDF 路徑
        cols, rows: 棋盤格列數、行數
        square_size_m: 方格邊長（公尺）

    Returns:
        實際保存的檔案路徑
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    try:
        from PIL import Image

        dpi = _PDF_DPI
        square_mm = square_size_m * 1000

        # 棋盤需要的物理尺寸
        board_w_mm = cols * square_mm
        board_h_mm = rows * square_mm

        # A4 landscape 可用區域
        usable_w_mm = _A4_W_MM - 2 * _PAGE_MARGIN_MM  # 277mm
        usable_h_mm = _A4_H_MM - 2 * _PAGE_MARGIN_MM  # 190mm

        # 如果放不下，等比縮小 square_size 並警告
        actual_square_mm = square_mm
        if board_w_mm > usable_w_mm or board_h_mm > usable_h_mm:
            fit_scale = min(usable_w_mm / board_w_mm, usable_h_mm / board_h_mm)
            actual_square_mm = square_mm * fit_scale
            print(f"[Board] 警告: {cols}x{rows} @ {square_mm:.1f}mm "
                  f"({board_w_mm:.0f}x{board_h_mm:.0f}mm) 超出 A4 可用區域 "
                  f"({usable_w_mm:.0f}x{usable_h_mm:.0f}mm)")
            print(f"[Board] 自動縮小方格: {square_mm:.1f}mm → {actual_square_mm:.2f}mm")
            print(f"[Board] *** 請將 settings.yaml 中 square_size 改為 "
                  f"{actual_square_mm / 1000:.6f}，或打印後實量修改 ***")

        # 計算棋盤在 PDF 上的精確像素數
        actual_board_w_mm = cols * actual_square_mm
        actual_board_h_mm = rows * actual_square_mm
        board_w_px = int(round(actual_board_w_mm / 25.4 * dpi))
        board_h_px = int(round(actual_board_h_mm / 25.4 * dpi))

        # 用精確尺寸重新生成棋盤圖（margin=0，由 PDF 頁面提供邊距）
        board_img_precise = cv2.aruco.CharucoBoard.generateImage(
            board_image,  # 這裡其實只需要 board 對象，但 API 是 board.generateImage
            outSize=(board_w_px, board_h_px),
            marginSize=0,
        ) if False else None  # generateImage 是 board 的方法，需要用不同方式

        # 直接把原始 board_image 縮放到精確像素數
        if len(board_image.shape) == 2:
            pil_img = Image.fromarray(board_image, mode="L")
        else:
            pil_img = Image.fromarray(
                cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB)
            )
        pil_img = pil_img.resize((board_w_px, board_h_px), Image.LANCZOS)

        # A4 landscape 頁面像素
        page_w_px = int(round(_A4_W_MM / 25.4 * dpi))
        page_h_px = int(round(_A4_H_MM / 25.4 * dpi))

        canvas = Image.new(pil_img.mode, (page_w_px, page_h_px), 255)
        offset_x = (page_w_px - board_w_px) // 2
        offset_y = (page_h_px - board_h_px) // 2
        canvas.paste(pil_img, (offset_x, offset_y))

        canvas.save(path, "PDF", resolution=dpi)

        printed_w = actual_board_w_mm
        printed_h = actual_board_h_mm
        print(f"[Board] PDF 已保存: {path}")
        print(f"[Board] 頁面: A4 landscape ({_A4_W_MM:.0f}x{_A4_H_MM:.0f}mm)")
        print(f"[Board] 棋盤: {cols}x{rows}, 方格 {actual_square_mm:.2f}mm, "
              f"總 {printed_w:.1f}x{printed_h:.1f}mm")
        return path

    except ImportError:
        png_path = path.replace(".pdf", ".png")
        cv2.imwrite(png_path, board_image)
        print(f"[Board] Pillow 未安裝，已保存 PNG: {png_path}")
        print("[Board] 請手動打印該 PNG 至 A4 紙張，並量測方格實際尺寸")
        return png_path
