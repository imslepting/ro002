# RO002 開發規範

## Python 環境

所有 Python 腳本必須使用 conda 環境 `ro002` 運行：

```bash
# 運行腳本
conda run -n ro002 python <script.py>

# 安裝依賴
conda run -n ro002 pip install <package>
```

環境資訊：
- 環境名稱：`ro002`
- Python 版本：3.11
- 已安裝：opencv-python, numpy, pyyaml

## 項目結構

- `config/settings.yaml` — 全局配置
- `doc` — 開發文檔
- `shared/` — 跨階段共用工具和類型定義
- `phase0_hw_diagnostics/` — 硬體診斷 GUI
- `phase1_intrinsics/` — 相機內參標定
- `phase2_extrinsics/` — 相機外參標定
- `phase3_stereo_depth/` — 雙目立體深度（Fast-FoundationStereo → Metric 深度圖）
- `phase4_multi_view_recon/` — 多視角重建（DA3Nested → 統一 Metric 點雲）
- `phase5_vlm_planning/` — VLM 規劃
- `phase6_arm_execution/` — 機械臂執行

## 運行示例

```bash
conda run -n ro002 python phase0_hw_diagnostics/main_diagnostics.py
```
