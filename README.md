# RO002 — VLM-Driven Robotic Arm Grasping System

VLM Agent 驅動的機械臂抓取規劃系統。VLM 作為工具使用者，通過 tool-use 調度 SAM3（分割）、GraspGen（抓取）、cuRobo（軌跡）等專業模塊。

## Quick Start

```bash
# 1. 建立 conda 環境
conda create -n ro002 python=3.11 -y
conda activate ro002

# 2. 安裝 PyTorch（匹配你的 CUDA 版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安裝依賴
pip install -r requirements.txt

# 4. 安裝本地外部套件（Phase 5 必需）
pip install -e external/sam3
pip install -e external/GraspGen --no-deps
pip install pycocotools

# 5. 下載模型權重
bash external/download_weights.sh

# 6. 啟動主 GUI
conda run -n ro002 python phase5_vlm_planning/test_vlm_agent.py
```

### Phase 5 常見環境錯誤

若啟動後看到以下訊息：

- `No module named 'sam3'`
- `No module named 'grasp_gen'`

請確認你已在 `ro002` 安裝本地套件：

```bash
conda run -n ro002 python -m pip install -e external/sam3
conda run -n ro002 python -m pip install -e external/GraspGen --no-deps
conda run -n ro002 python -m pip install pycocotools
```

快速驗證：

```bash
conda run -n ro002 python -c "import sam3, grasp_gen; print('ok')"
```

## Phases

| Phase | 説明 | 命令 |
|-------|------|------|
| 0 | 硬體診斷 | `python phase0_hw_diagnostics/main_diagnostics.py` |
| 1 | 內參標定 | `python phase1_intrinsics/main_intrinsics.py` |
| 2 | 外參標定 | `python phase2_extrinsics/main_extrinsics.py` |
| 3 | 雙目深度 | `python phase3_stereo_depth/main_stereo_depth.py` |
| 5 | **VLM Agent** | `python phase5_vlm_planning/test_vlm_agent.py` |

## Phase 5 — 三種模式

| 按鈕 | 速度 | 輸入 | 説明 |
|------|------|------|------|
| **Run (Direct)** | ~3-5s | 英文物件名 | 最快，不用 LLM |
| **Send (LLM)** | ~3-5s | 任意語言 | LLM 翻譯 + Direct |
| **Send (VLM)** | ~30-60s | 任意語言 | AI 全自主驗證 |

## Configuration

所有配置集中在 `config/settings.yaml`，詳見 `doc/project_development_status.md`。

## External Dependencies

模型權重不包含在 repo 中（共 ~6GB），用 `external/download_weights.sh` 下載：

- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) — 雙目匹配
- [GraspGen](https://github.com/NVlabs/GraspGen) — 6-DoF 抓取生成
- [SAM3](https://github.com/facebookresearch/sam3) — 物件分割
- [cuRobo](https://github.com/NVlabs/curobo) — GPU 軌跡規劃
