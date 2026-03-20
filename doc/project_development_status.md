# RO002 項目開發進度説明

> 更新日期：2026-03-19
> 環境：conda env `ro002` (Python 3.11)
> 運行方式：`conda run -n ro002 python <script.py>`

---

## 目錄

1. [項目概述與架構](#1-項目概述與架構)
2. [Phase 0 — 硬體診斷](#2-phase-0--硬體診斷)
3. [Phase 1 — 相機內參標定](#3-phase-1--相機內參標定)
4. [Phase 2 — 相機外參標定](#4-phase-2--相機外參標定)
5. [Phase 3 — 雙目立體深度](#5-phase-3--雙目立體深度)
6. [Phase 4 — 多視角重建](#6-phase-4--多視角重建)
7. [Phase 5 — VLM 規劃（核心）](#7-phase-5--vlm-規劃核心)
8. [Phase 6 — 機械臂執行](#8-phase-6--機械臂執行)
9. [全局配置參考](#9-全局配置參考)
10. [外部依賴](#10-外部依賴)

---

## 1. 項目概述與架構

### 1.1 設計理念

RO002 是一套 **VLM 驅動的機械臂抓取規劃系統**。核心思想：不把高維場景理解壓縮進端到端的 VLA（Vision-Language-Action）模型，而是讓 VLM 作為 **工具使用者**，通過 tool-use API 調度各專業模塊：

```
用戶指令（自然語言）
    ↓
VLM Agent（Claude / GPT-4o）
    ├── 調用 SAM3     → 物件分割
    ├── 調用 GraspGen  → 6-DoF 抓取位姿
    ├── 調用 cuRobo    → 無碰撞軌跡規劃
    └── 輸出 plan.json → 機械臂執行
```

### 1.2 數據流總覽

```
Phase 0: 硬體診斷 ──→ 確認相機可用
Phase 1: 內參標定 ──→ intrinsics.json (K, D)
Phase 2: 外參標定 ──→ extrinsics.json (R, T)
            ↓
Phase 3: 雙目深度 ──→ metric depth (公尺)
Phase 4: 多視角重建 → unified point cloud (.ply)
            ↓
Phase 5: VLM 規劃 ──→ plan.json (抓取位姿 + 軌跡)
            ↓
Phase 6: 機械臂執行 → 物理世界操作
```

### 1.3 項目結構

```
RO002/
├── config/settings.yaml          # 全局配置（所有階段共用）
├── shared/                       # 跨階段共用代碼
│   ├── types.py                  # 所有 dataclass 定義
│   ├── camera_manager.py         # 多線程相機讀取
│   └── tk_utils.py               # Tkinter UI 工具
├── phase0_hw_diagnostics/        # 硬體診斷 GUI
├── phase1_intrinsics/            # 相機內參標定 GUI
├── phase2_extrinsics/            # 相機外參標定 GUI
├── phase3_stereo_depth/          # 雙目立體深度推理
├── phase4_multi_view_recon/      # 多視角重建（骨架）
├── phase5_vlm_planning/          # VLM Agent 規劃（核心）
├── phase6_arm_execution/         # 機械臂執行（骨架）
├── external/                     # 外部模型倉庫
│   ├── Fast-FoundationStereo/    # 雙目匹配模型
│   ├── GraspGen/                 # 6-DoF 抓取生成
│   ├── sam3/                     # SAM3 分割模型
│   └── curobo/                   # GPU 軌跡規劃
└── doc/                          # 文檔
```

### 1.4 開發狀態總覽

| 階段 | 狀態 | 説明 |
|------|------|------|
| Phase 0 | ✅ 完成 | 硬體掃描 + 相機測試 GUI |
| Phase 1 | ✅ 完成 | ChArUco 內參標定 GUI |
| Phase 2 | ✅ 完成 | 雙目外參標定 GUI |
| Phase 3 | ✅ 完成 | Fast-FoundationStereo 即時推理 |
| Phase 4 | 🔧 骨架 | DA3Nested 待集成 |
| Phase 5 | ✅ 完成 | VLM Agent GUI + Direct/LLM 快速管線 |
| Phase 6 | 📋 規劃 | xArm SDK 待集成 |

---

## 2. Phase 0 — 硬體診斷

### 2.1 用途

在標定前掃描所有 USB 攝像頭，測試每台相機的分辨率、幀率、亮度、清晰度，確保硬體可用。

### 2.2 運行方式

```bash
conda run -n ro002 python phase0_hw_diagnostics/main_diagnostics.py
```

### 2.3 GUI 功能

- **自動掃描**：列舉 `/dev/video0` ~ `/dev/video15`，檢測可用設備
- **2×2 畫面**：同時顯示最多 4 台相機的即時畫面
- **每台相機測試指標**：
  - 分辨率（實際 vs 請求）
  - 幀率（API 回報 vs 實測）
  - 幀成功率（30 幀中成功讀取的百分比）
  - 平均亮度（< 30 或 > 230 警告）
  - 清晰度（Laplacian 方差，< 50 警告）
- **快捷鍵**：Q 退出 / S 保存報告 / R 重新掃描
- **分辨率切換**：每台相機獨立下拉選單

### 2.4 輸出

```
phase0_hw_diagnostics/outputs/reports/
├── diagnostic_YYYYMMDD_HHMMSS.json   # 結構化報告
└── screenshot_camN.png               # 各相機截圖
```

### 2.5 相關配置 (`settings.yaml → phase0`)

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `scan_max_index` | 16 | 掃描設備索引上限 |
| `test_frames` | 30 | 每台相機測試幀數 |
| `tile_size` | [640, 480] | GUI 中每個畫面尺寸 |
| `sharpness_threshold` | 50.0 | Laplacian 方差閾值 |
| `brightness_range` | [30, 230] | 正常亮度範圍 |

---

## 3. Phase 1 — 相機內參標定

### 3.1 用途

使用 ChArUco 標定板計算每台相機的內參矩陣 K（焦距 + 主點）和畸變係數 D。

### 3.2 運行方式

```bash
conda run -n ro002 python phase1_intrinsics/main_intrinsics.py
```

### 3.3 工作流程

1. **列印標定板**：程式自動生成 ChArUco PDF（9×6，方格 3cm，ArUco 2.23cm）
2. **採集圖像**：GUI 中選擇相機 → 按 SPACE 拍照（建議 ≥30 張，不同角度和距離）
3. **運行標定**：`cv2.aruco.calibrateCameraCharuco` 計算 K, D
4. **驗證**：查看每幀重投影誤差，RMS < 1.0 px 通過
5. **保存**：寫入 `intrinsics.json`

### 3.4 GUI 功能

- 相機選擇（cam0 ~ cam3）
- 即時畫面 + ChArUco 角點檢測疊加
- 手動拍照 / 定時拍照 / 連續拍照模式
- 每張圖的角點檢測結果即時回饋
- 標定完成後顯示重投影誤差分佈圖

### 3.5 輸出

```
phase1_intrinsics/outputs/
├── intrinsics.json                    # {cam_name: {K, D, image_size, rms}}
├── raw_images/{cam0,cam1,...}/        # 原始標定圖像
└── reports/                           # 重投影誤差可視化
```

### 3.6 相關配置 (`settings.yaml → calibration`)

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `charuco.cols` | 9 | ChArUco 列數 |
| `charuco.rows` | 6 | ChArUco 行數 |
| `charuco.square_size` | 0.03 | 方格邊長（公尺） |
| `charuco.marker_size` | 0.0223 | ArUco 標記邊長（公尺） |
| `charuco.aruco_dict` | DICT_4X4_50 | ArUco 字典 |
| `acceptance.max_rms_intrinsics` | 1.0 | RMS 驗收閾值（像素） |

---

## 4. Phase 2 — 相機外參標定

### 4.1 用途

計算相機對之間的相對位姿（R, T），用於雙目校正和多相機融合。

### 4.2 運行方式

```bash
conda run -n ro002 python phase2_extrinsics/main_extrinsics.py
```

### 4.3 工作流程

1. **選擇相機對**：例如 `cam0_cam1`（雙目對）
2. **同步採集**：兩台相機同時拍攝標定板（按 SPACE）
3. **立體標定**：`cv2.stereoCalibrate` 計算 R, T, E, F
4. **驗證**：極線幾何檢查（校正後左右圖像的對應點應在同一水平線上）
5. **Bundle Adjustment**（可選）：多相機星型拓撲全局優化

### 4.4 GUI 功能

- 雙畫面同步顯示
- 極線幾何可視化（校正後疊加水平線）
- RMS 誤差實時顯示

### 4.5 輸出

```
phase2_extrinsics/outputs/
├── extrinsics.json                    # {pair_name: {R, T, rms}}
├── raw_pairs/{pair_name}/             # 同步拍攝的圖像對
└── reports/                           # 極線誤差可視化
```

### 4.6 相關配置

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `acceptance.max_rms_extrinsics` | 1.0 | 立體標定 RMS 閾值（像素） |
| `acceptance.max_ba_residual` | 0.5 | Bundle Adjustment 殘差閾值 |

---

## 5. Phase 3 — 雙目立體深度

### 5.1 用途

使用 Fast-FoundationStereo 模型從校正後的雙目圖像推理視差圖，轉換為 **公尺制深度圖**。

### 5.2 運行方式

```bash
conda run -n ro002 python phase3_stereo_depth/main_stereo_depth.py
```

### 5.3 處理流程

```
左右原圖 → StereoRectifier 校正 → Fast-FoundationStereo 視差推理
    → disparity_to_depth (depth = focal × baseline / disp)
    → metric depth (float32, 公尺)
    → colorize_depth (偽彩色可視化)
```

### 5.4 核心模塊

| 模塊 | 功能 |
|------|------|
| `StereoRectifier` | 載入 Phase 1-2 結果，計算校正映射表，提供 `rectify()` |
| `StereoInference` | 封裝 Fast-FoundationStereo，`predict_disparity()` 返回 float32 視差 |
| `disparity_to_depth()` | 視差 → 深度（`depth = focal × baseline / disp`） |
| `colorize_depth()` | 深度 → 偽彩色 BGR 圖 |
| `depth_to_pointcloud()` | 深度 + Q 矩陣 → 3D 點雲 |

### 5.5 性能模式

| 模式 | `valid_iters` | 用途 |
|------|---------------|------|
| 快速 | 4 | 即時顯示（~15 FPS） |
| 精確 | 8 | 最終深度輸出 |

### 5.6 輸出

```
phase3_stereo_depth/outputs/
├── stereo_depth/depth_YYYYMMDD_HHMMSS.npy    # float32 深度（公尺）
└── disparity/disparity_YYYYMMDD_HHMMSS.png   # 偽彩色視差圖
```

### 5.7 相關配置 (`settings.yaml → stereo_depth`)

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `model_dir` | `external/Fast-Foundation.../model_best_bp2_serialize.pth` | 模型權重路徑 |
| `max_disparity` | 256 | 最大視差搜索範圍 |
| `valid_iters` | 8 | 推理迭代次數（4=快，8=精確） |
| `pad_multiple` | 32 | 張量填充倍數 |
| `min_depth` | 0.05 | 最小有效深度（公尺） |
| `max_depth` | 10.0 | 最大有效深度（公尺） |

---

## 6. Phase 4 — 多視角重建

### 6.1 狀態：🔧 骨架階段

目前僅有目錄結構和 `__init__.py`，核心模塊待實現。

### 6.2 規劃架構

```
多相機圖像 + Phase 1-2 標定
    → DA3Nested 推理（每相機獨立的相對深度）
    → RANSAC 對齊到 Phase 3 的雙目公尺深度（scale anchoring）
    → 多視角點雲融合（世界座標系）
    → 統一公尺制點雲 (.ply)
```

### 6.3 待實現模塊

- `da3_inference.py` — DA3Nested 模型推理封裝
- `scale_aligner.py` — 相對深度 → 公尺深度的尺度對齊
- `pointcloud_fusion.py` — 多視角點雲融合

### 6.4 相關配置

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `da3_model` | `depth-anything/DA3NESTED-GIANT-LARGE` | 模型名稱 |
| `export_format` | `ply` | 點雲輸出格式 |

---

## 7. Phase 5 — VLM 規劃（核心）

這是項目的核心階段，實現了 VLM Agent 驅動的物件抓取規劃系統。

### 7.1 運行方式

```bash
conda run -n ro002 python phase5_vlm_planning/test_vlm_agent.py
```

### 7.2 主 GUI — `test_vlm_agent.py`

#### 7.2.1 狀態機

```
LOADING ──→ IDLE ──→ LIVE ──→ AGENT_RUNNING ──→ PLAN_READY
                      ↑                              │
                      └──────────────────────────────┘
```

| 狀態 | 説明 |
|------|------|
| `LOADING` | 載入模型中（Phase 3 + SAM3 + GraspGen） |
| `IDLE` | 模型就緒，等待啟動相機 |
| `LIVE` | 雙目深度即時推理中，可發送任務 |
| `AGENT_RUNNING` | Agent/管線執行中 |
| `PLAN_READY` | 抓取計劃完成，可保存/查看 3D |

#### 7.2.2 GUI 佈局

```
┌─────────────────────────────────────────────────────────────┐
│ ARM-VLM Agent    │ Status: ...              │  FPS: 12.3    │
├──────────────────┬──────────────────────────────────────────┤
│ ┌──────┐┌──────┐ │  Agent Conversation                      │
│ │ RGB  ││Depth │ │  ─────────────────────                    │
│ │      ││(可點) │ │  [User] give me the bottle               │
│ └──────┘└──────┘ │  [Agent] 分割目標...                      │
│ ┌──────┐┌──────┐ │    -> [segment_object(desc="bottle")]     │
│ │Segmt ││Grasp │ │    <- 分割完成：1 mask, score=0.768       │
│ │      ││      │ │  [Agent] 計算抓取...                      │
│ └──────┘└──────┘ │    -> [compute_grasp()]                   │
│                  │    <- score=0.927, pos=[0.36,-0.22,0.79]  │
│ [Start][Stop]    │                                           │
│ T_cam2arm: cfg   │  Result Summary                           │
│ Cam: cam0        │  Score: 0.927  Width: 136mm               │
│ Provider: claude │  Pos(arm): [0.359, -0.220, 0.789]         │
│ [Load Agent]     │  [Save Plan] [View 3D]                    │
├──────────────────┴──────────────────────────────────────────┤
│ Task: [________________________] [Run(Direct)] [Send(VLM)] [Send(LLM)] │
└─────────────────────────────────────────────────────────────┘
```

#### 7.2.3 四面板顯示

| 面板 | 位置 | 顯示內容 |
|------|------|----------|
| **RGB** | 左上 | 即時校正左圖 / 場景快照 |
| **Depth** | 右上 | 偽彩色深度圖（**左鍵**點擊查詢深度值，多點自動連線顯示 ΔDepth，**右鍵**清除標記） |
| **Segment** | 左下 | SAM3 分割結果（高亮 mask + bounding box） |
| **Grasp** | 右下 | 抓取位姿可視化（綠十字=接觸點，紅箭頭=夾爪接近方向，綠線=夾爪張開方向） |

#### 7.2.4 三種執行模式

##### Run (Direct) — 最快，英文輸入

```
Task 輸入（英文物件描述，如 "silver bottle"）
    → SAM3 分割（2-5s）      ┐ 並行預計算場景點雲
    → GraspGen 抓取（0.5s）   │
    → 保存 plan.json（0.05s） │
    ──────────────────────────┘
    Total: ~3-5s
```

- **輸入要求**：英文物件視覺描述
- **不需要 VLM**：完全跳過 LLM 調用
- **最佳場景**：已知物件名稱，追求最快速度

##### Send (LLM) — 快速，任意語言

```
Task 輸入（任意語言，如 "幫我拿紅色杯子"）
    ┌── LLM 提取描述（→ ["red cup","red mug","cup"]）  ┐
    ├── SAM3 ViT 編碼（預快取）                          ├ 三路並行
    └── 預計算場景點雲                                    ┘
    → SAM3 decoder（ViT 已快取，~0.5s）
    → 多描述自動重試
    → GraspGen + save（~0.5s）
    ──────────────────────────────
    Total: ~3-5s (OpenAI) / ~6s (Claude CLI)
```

- **輸入要求**：任何語言的自然語言指令
- **LLM 僅調用一次**：提取英文物件描述（多候選）
- **並行策略**：LLM 提取 + SAM3 ViT 編碼 + 場景點雲預計算同時進行
- **自動重試**：如果第一個描述找不到物件，自動嘗試 LLM 給出的備選描述
- **LLM 優先級**：OpenAI gpt-4o-mini（~0.5-1s） → Claude CLI（~5-10s） → 原文 fallback

##### Send (VLM) — 最慢但最可靠

```
Task 輸入 + 場景圖片 → VLM Agent 自主循環
    VLM Turn 1: 觀察場景
    VLM Turn 2: 調用 segment_object → 驗證分割
    VLM Turn 3: 調用 compute_grasp → 驗證抓取
    VLM Turn 4: 調用 save_plan → 最終回覆
    ──────────────────────────────
    Total: ~30-60s（VLM 調用佔 90%+）
```

- **輸入要求**：任何語言的自然語言指令
- **完全自主**：VLM 自行決定調用哪個工具、驗證結果、重試
- **支持 VLM Provider**：`openai`（GPT-4o）或 `claude_code`（Claude CLI）
- **最佳場景**：複雜任務、需要 AI 驗證分割/抓取正確性

#### 7.2.5 控制按鈕

| 按鈕 | 功能 | 可用狀態 |
|------|------|----------|
| Start Live | 啟動雙目相機 + 即時深度推理 | IDLE |
| Stop | 停止相機 | LIVE, PLAN_READY |
| Load T... | 載入 T_cam2arm 變換矩陣（JSON/NPY） | 非 LOADING/RUNNING |
| Load Agent | 預載 VLM client（加速首次調用） | 非 LOADING/RUNNING |
| Cancel | 取消正在運行的 Agent | AGENT_RUNNING |
| Skip Verify | 跳過 VLM 驗證步驟 | AGENT_RUNNING |
| Clear | 清空對話和所有面板 | 非 LOADING/RUNNING |
| Save Plan | 手動保存 plan.json | PLAN_READY |
| View 3D | 開啟 Open3D 3D 查看器 | PLAN_READY |

#### 7.2.6 3D 查看器（View 3D）

點擊 View 3D 後開啟 Open3D 視窗，顯示：

- **灰色**：場景點雲（非物件部分）
- **彩色**：物體點雲（SAM3 mask 區域）
- **綠色線框**：夾爪模型（按 grasp pose 放置）
- **紅色線段**：Approach 路徑（pre-grasp → TCP）
- **黃色球**：Pre-grasp 位置
- **紅色球**：TCP 位置
- **綠色球**：接觸點
- **橙色線框**：Workspace 邊界
- **RGB 座標軸**：原點參考

#### 7.2.7 Depth 互動功能

在右上角 Depth 面板上：

- **左鍵點擊**：顯示該像素的深度值（公尺），畫綠色十字標記
- **連續點擊**：多個點之間自動畫黃色連線，標注 ΔDepth（深度差）
- **右鍵**：清除所有標記
- 底部顯示最近一次查詢的座標和深度值

### 7.3 技能模塊詳解

#### 7.3.1 SAM3 分割技能 (`skills/skill_sam3/`)

**原理**：使用 Meta SAM3 模型，輸入圖片 + 英文文字描述，輸出物件的分割 mask。

**處理流程**：
```
BGR 圖片 → cvtColor → PIL.Image
    → set_image() [ViT 編碼, ~2-4s, 可快取]
    → set_text_prompt() [Decoder, ~0.5s]
    → masks (N,H,W) + scores (N,) + boxes (N,4)
    → 按 score 排序 → 過濾 < threshold → 取 top-N
    → 疊加可視化 → SAM3Result
```

**快取機制**：同一場景圖片的 ViT 編碼會被快取。同場景多次查詢（換描述重試）只需跑 decoder（~0.5s），省掉 ViT 的 2-4s。

**配置** (`settings.yaml → skill_sam3`)：

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `checkpoint` | `external/sam3/weights/sam3.pt` | 本地權重路徑 |
| `score_threshold` | 0.3 | mask 置信度閾值 |
| `max_masks` | 5 | 最多保留的 mask 數 |
| `overlay_alpha` | 0.4 | 可視化疊加透明度 |

#### 7.3.2 GraspGen 抓取技能 (`skills/skill_capture_point/`)

**原理**：使用 NVIDIA GraspGen（SE(3) Diffusion）在物件點雲上生成 6-DoF 夾爪抓取位姿。

**處理流程**：
```
SAM3 mask + depth → mask_to_pointcloud (cam 座標)
    → T_cam2arm 變換 → arm 座標系
    → workspace 過濾
    → GraspGen.run_inference()
        → Diffusion 生成 N 候選 (4×4 SE(3))
        → Discriminator 打分 → top-K
    → 碰撞過濾（可選）
    → 桌面高度過濾
    → Approach 方向過濾（漸進式放寬）
    → 選擇最佳 → 可視化 → CapturePointResult
```

**Grasp 座標約定**：
```
4×4 齊次變換矩陣 (arm 座標系):
  ┌               ┐
  │ Rx Ry Rz  Tx  │    X 軸 = 夾爪張開方向（contact direction）
  │  ...      Ty  │    Y 軸 = 垂直方向（cross product）
  │  ...      Tz  │    Z 軸 = 接近方向（approach direction）
  │ 0  0  0   1   │    T   = TCP 位置
  └               ┘
```

**Approach 方向過濾**（限制從上往下抓取）：

```
目標方向: [0, 1, 0]（arm 座標 +Y = 綠軸 = 向下）
過濾邏輯: dot(grasp_z, target) ≥ threshold
漸進放寬: 0.9 (25°) → 0.7 (45°) → 0.5 (60°) → 選最接近的
保證: 至少返回 1 個結果（最接近目標方向的候選）
```

**Fallback 安全鏈**：每一步過濾失敗只退回上一步結果，不會退回未過濾的原始 grasps。

**配置** (`settings.yaml → skill_capture_point`)：

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `gripper_config` | `.../graspgen_robotiq_2f_140.yml` | 夾爪模型配置 |
| `num_grasps` | 50 | Diffusion 採樣數（50=快速，200=精確） |
| `num_candidates` | 10 | 保留前 N 個候選 |
| `filter_collisions` | true | 是否啟用碰撞過濾 |
| `gripper_width` | 0.136 | 夾爪最大張開寬度（公尺） |
| `pre_grasp_distance` | 0.10 | Pre-grasp 後退距離（公尺） |
| `table_height` | "auto" | 桌面高度（"auto" 或固定數值） |
| `approach_direction` | [0, 1, 0] | 目標接近方向（arm 座標系） |
| `approach_threshold` | 0.9 | 方向過濾閾值（0.9≈25°偏差） |
| `workspace_limits` | x:±0.5 y:±0.5 z:0~2 | 工作空間限制（公尺） |

#### 7.3.3 軌跡規劃技能 (`skills/skill_trajectory_planning/`)

**原理**：使用 NVIDIA cuRobo 在 GPU 上進行無碰撞軌跡規劃。

**配置** (`settings.yaml → skill_trajectory_planning`)：

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `algorithm` | "curobo" | 規劃算法 |
| `robot_config` | `config/robot_curobo.yml` | 機器人 URDF 配置 |
| `collision_from_pointcloud` | true | 從點雲生成碰撞體素 |
| `voxel_size` | 0.02 | 碰撞體素大小（公尺） |
| `interpolation_steps` | 100 | 軌跡插值步數 |

### 7.4 VLM Client 架構

```
VLMClient (abstract)
├── OpenAIVLM      — OpenAI API (gpt-4o), 原生 tool-use
└── ClaudeCodeVLM  — Claude CLI subprocess, 結構化 prompt 模擬 tool-use
```

**OpenAIVLM**：直接調用 `openai.OpenAI().chat.completions.create()`，支持原生 function calling。圖片以 base64 inline 傳送。

**ClaudeCodeVLM**：啟動 `claude` CLI 子進程，將對話歷史格式化為結構化 prompt（含 `[SYSTEM]`、`[USER]`、`[ASSISTANT]`、`[TOOL RESULT]` 標記）。圖片保存為臨時文件，Claude 用 Read 工具讀取。

**配置** (`settings.yaml → vlm`)：

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `provider` | "claude_code" | VLM Provider（"openai" 或 "claude_code"） |
| `openai_model` | "gpt-4o" | OpenAI 模型 |
| `claude_model` | "claude-sonnet-4-20250514" | Claude 模型 |
| `max_tokens` | 4096 | 最大輸出 token |

### 7.5 Agent Loop 架構

```python
while turn_count < MAX_TURNS (20):
    response = vlm.create(messages, system, tools)
    if response.stop_reason == "end_turn":
        break
    if response.stop_reason == "tool_use":
        for tool_call in response:
            result = executor.execute(tool_name, tool_input)
            messages.append(tool_result)
```

**工具 Schema**：

| 工具 | 輸入 | 輸出 |
|------|------|------|
| `capture_scene` | （無） | 深度統計文字 |
| `segment_object` | `object_description`（英文） | 標注圖 + mask 統計 |
| `compute_grasp` | （無，用上一步 mask） | 抓取參數（score, pos, width） |
| `save_plan` | `task_description` | plan.json 路徑 |

### 7.6 Plan 輸出格式

```json
{
  "task": "grasp silver bottle on the table",
  "timestamp": "2026-03-19T11:01:58",
  "grasp": {
    "position": [0.359, -0.220, 0.789],
    "rotation": [[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]],
    "score": 0.927,
    "width_m": 0.136,
    "pixel": [320, 240],
    "num_candidates": 1
  },
  "session_dir": "phase5_vlm_planning/outputs/sessions/20260319_110158/"
}
```

Session 目錄包含完整快照：
```
sessions/20260319_110158/
├── task_input.txt          # 用戶原始指令
├── rgb_snapshot.jpg        # 場景 RGB
├── depth.npy               # 場景深度
├── sam3_annotated.jpg      # 分割結果
├── best_mask.npy           # 分割 mask
├── grasp_viz.jpg           # 抓取可視化
└── plan.json               # 執行計劃
```

### 7.7 性能基準

| 模式 | 首次查詢 | 同場景重試 | 瓶頸 |
|------|----------|------------|------|
| Run (Direct) | ~3-5s | ~0.5-1s (ViT 快取) | SAM3 ViT (~80%) |
| Send (LLM) + OpenAI | ~3-5s | ~2s | SAM3 ViT |
| Send (LLM) + Claude | ~6-8s | ~6s | Claude CLI subprocess |
| Send (VLM) | ~30-60s | ~30-60s | VLM 多輪調用 |

### 7.8 其他測試腳本

| 腳本 | 用途 |
|------|------|
| `test_skill_sam3.py` | SAM3 獨立測試（圖片 + 文字 → 分割） |
| `test_skill_capture_point.py` | GraspGen 獨立測試（RGB + depth + mask → grasp） |
| `test_skill_trajectory_planning.py` | cuRobo 獨立測試（joints + goal → trajectory） |

---

## 8. Phase 6 — 機械臂執行

### 8.1 狀態：📋 規劃階段

目前僅有目錄結構（`src/__init__.py`、`outputs/`），暫不實現。

### 8.2 規劃架構

```
plan.json (Phase 5 輸出)
    → arm_controller.py   (xArm SDK 封裝)
    → trajectory_executor  (逐 waypoint 追蹤)
    → gripper_controller   (夾爪開合控制)
    → safety_monitor       (力矩/碰撞監控)
    → execution logs       (JSONL 格式)
```

### 8.3 相關配置 (`settings.yaml → arm`)

| 參數 | 預設值 | 説明 |
|------|--------|------|
| `sdk` | "xarm" | 機械臂 SDK |
| `ip` | "192.168.1.xxx" | 機械臂 IP 地址 |
| `base_frame` | "world" | 基座標系 |
| `T_cam2arm` | 單位旋轉 + [0,-0.1,0] 平移 | 相機→機械臂變換（暫定值） |

**座標系約定**：
```
相機座標系 (cam frame):  x=右, y=下, z=前（深度方向）
機械臂座標系 (arm frame): 同相機座標系（暫定，正式部署用 hand-eye calibration 替換）
T_cam2arm: P_arm = R_cam2arm × P_cam + t_cam2arm
目前: R = I, t = [0, -0.1, 0]（arm 原點在 cam 下方 10cm）
```

---

## 9. 全局配置參考

完整配置文件：`config/settings.yaml`

### 9.1 快速調參指南

**想加快推理速度？**
```yaml
skill_capture_point:
  num_grasps: 50          # 降低 → 更快（最低建議 30）
  filter_collisions: false # 關閉碰撞過濾省 200-500ms
stereo_depth:
  valid_iters: 4          # 快速模式
```

**想提高抓取精度？**
```yaml
skill_capture_point:
  num_grasps: 200         # 增加候選數
  num_candidates: 20      # 保留更多候選
  approach_threshold: 0.95 # 更嚴格的方向約束
```

**想放寬抓取方向？**
```yaml
skill_capture_point:
  approach_threshold: 0.5  # 允許 60° 偏差
  # 或完全關閉方向約束：
  # approach_direction: null
```

**想切換 VLM Provider？**
```yaml
vlm:
  provider: "openai"       # 切換到 OpenAI（需 OPENAI_API_KEY 環境變量）
```

---

## 10. 外部依賴

| 倉庫 | 用途 | 位置 |
|------|------|------|
| **Fast-FoundationStereo** | 雙目匹配 foundation model | `external/Fast-FoundationStereo/` |
| **GraspGen** | SE(3) Diffusion 6-DoF 抓取生成 | `external/GraspGen/` |
| **SAM3** | Meta Segment Anything 3 | `external/sam3/` |
| **cuRobo** | NVIDIA GPU 無碰撞軌跡規劃 | `external/curobo/` |

**權重下載**：`external/download_weights.sh`

**Python 依賴**（conda env `ro002`）：
- opencv-python, numpy, pyyaml（基礎）
- torch, torchvision（深度學習）
- open3d（3D 可視化，可選）
- openai（OpenAI API client，Send LLM 模式用）
- Pillow（圖像處理）
