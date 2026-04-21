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
| 7 | **Eye-to-Hand** | `python phase7_eye_to_hand/main_eye_to_hand.py` |
| 7b | **Arm Mesh ICP 外參** | `python phase7_arm_icp/main_arm_icp.py` |
| 8 | **RealSense 即時點雲** | `python phase8_realsense_pointcloud/main_realsense_pointcloud.py` |

## Phase 5 — 三種模式

| 按鈕 | 速度 | 輸入 | 説明 |
|------|------|------|------|
| **Run (Direct)** | ~3-5s | 英文物件名 | 最快，不用 LLM |
| **Send (LLM)** | ~3-5s | 任意語言 | LLM 翻譯 + Direct |
| **Send (VLM)** | ~30-60s | 任意語言 | AI 全自主驗證 |

## Configuration

所有配置集中在 `config/settings.yaml`，詳見 `doc/project_development_status.md`。

## Phase 8 備註

- 需要 RealSense SDK Python 綁定：`pyrealsense2`
- 建議先確認相機可被系統偵測，再啟動：

```bash
conda run -n ro002 python phase8_realsense_pointcloud/main_realsense_pointcloud.py
```

- Phase 8 目前先獨立運行，不會改動 Phase 5 的 stereo 流程。

## Phase 7b 備註（Arm Mesh ICP）

- 目的：由相機點雲與機械臂 mesh 對齊，估計 `T_cam2arm`。
- 預設流程：直接在 `phase7_arm_icp` 內抓取 RealSense 一幀並執行 ICP（不需要先跑 phase8 存檔）。
- 常用初始值已整合到 `config/settings.yaml` 的 `phase7_arm_icp` 區塊（mesh、init pose、ICP 參數、是否自動回寫）。

```bash
conda run -n ro002 python phase7_arm_icp/main_arm_icp.py
```

- 若你要重跑舊資料，也可指定離線點雲：

```bash
conda run -n ro002 python phase7_arm_icp/main_arm_icp.py \
	--target-cloud phase8_realsense_pointcloud/outputs/cloud_xxx.ply
```

- 需要暫時覆蓋設定時，才用 CLI 參數（例如 `--init-txyz`、`--init-rxyz-deg`、`--mesh-dir`）。

## External Dependencies

模型權重不包含在 repo 中（共 ~6GB），用 `external/download_weights.sh` 下載：

- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) — 雙目匹配
- [GraspGen](https://github.com/NVlabs/GraspGen) — 6-DoF 抓取生成
- [SAM3](https://github.com/facebookresearch/sam3) — 物件分割
- [cuRobo](https://github.com/NVlabs/curobo) — GPU 軌跡規劃
