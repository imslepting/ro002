# RO002 項目結構規劃

## 總覽

```
/RO002
├── README.md
├── config/                        # 全局配置（相機索引、路徑、超參數）
│   └── settings.yaml
├── assets/                        # 靜態資源（標定板圖片、說明文件）
│   └── charuco_board.png
├── docs/                          # ── 開發文檔 ──
│
├── phase0_hw_diagnostics/         # ── 硬體診斷 GUI ──
├── phase1_intrinsics/             # ── 相機內參標定 ──
├── phase2_extrinsics/             # ── 相機外參標定 ──
├── phase3_stereo_depth/           # ── 雙目立體深度（Fast-FoundationStereo → Metric 深度圖）──
├── phase4_multi_view_recon/       # ── 多視角重建（DA3Nested → 統一 Metric 點雲）──
├── phase5_vlm_planning/           # ── VLM 規劃（Agentic Loop）──
├── phase6_arm_execution/          # ── 機械臂執行 ──
└── shared/                        # 跨階段共用工具
```

---

## 詳細結構

```
/RO002
│
│
├── config/
│   └── settings.yaml              # 相機索引、路徑前綴、模型路徑等
│
├── assets/
│   └── charuco_board.png          # 打印用標定板（由 phase1 生成）
│
│
│ ════════════════════════════════════════════════
│  DOCS — 開發文檔
│ ════════════════════════════════════════════════
├── docs/
│   ├── 00_project_overview.md         # 項目總覽、動機、架構設計理念
│   ├── 01_hardware_setup.md           # 硬體清單、相機型號、安裝方式、接線
│   ├── 02_environment_setup.md        # 依賴安裝、Python 環境、CUDA 版本
│   │
│   ├── phase1_intrinsics/
│   │   └── intrinsics_guide.md        # 內參標定原理、ChArUco 使用說明、驗收標準
│   │
│   ├── phase2_extrinsics/
│   │   ├── stereo_guide.md            # 雙目外參標定流程說明
│   │   ├── multicam_guide.md          # 多相機星形拓撲說明、BA 原理
│   │   └── topology_diagram.png       # 相機拓撲示意圖
│   │
│   ├── phase3_stereo_depth/
│   │   └── stereo_depth_guide.md      # Fast-FoundationStereo 使用說明、Metric 深度圖生成
│   │
│   ├── phase4_multi_view_recon/
│   │   ├── da3_guide.md               # DA3Nested 模型系列說明、多視角推理流程
│   │   ├── scale_alignment_guide.md   # 雙目 metric 錨定 → DA3 尺度對齊
│   │   └── pointcloud_pipeline.md     # 多視角點雲融合流程說明
│   │
│   ├── phase5_vlm_planning/
│   │   ├── agentic_loop_guide.md      # Agentic loop 整體流程說明
│   │   ├── skill_sam3_guide.md        # skill-SAM3 使用說明、prompt 設計
│   │   ├── skill_capture_point_guide.md  # skill-CapturePoint 座標轉換說明
│   │   └── skill_trajectory_guide.md  # skill-TrajectoryPlanning 軌跡規劃說明
│   │
│   ├── phase6_arm_execution/
│   │   ├── arm_sdk_guide.md           # 機械臂 SDK 說明、座標系定義
│   │   └── safety_protocol.md         # 安全監控規則、緊急停止流程
│   │
│   ├── design_decisions/              # 設計決策記錄（ADR）
│   │   ├── ADR001_why_not_VLA.md      # 為何不採用 VLA 路線
│   │   ├── ADR002_da3_vs_vggt.md      # DA3 vs VGGT 選型依據
│   │   └── ADR003_stereo_anchor.md    # 雙目作為 metric 錨點的設計決策
│   │
│   └── api_reference/                 # 各模塊的函數接口說明（自動生成或手寫）
│       ├── phase3_api.md
│       ├── phase4_api.md
│       ├── phase5_api.md
│       └── phase6_api.md
│
│
│ ════════════════════════════════════════════════
│  PHASE 0 — 硬體診斷 GUI
│ ════════════════════════════════════════════════
├── phase0_hw_diagnostics/
│   ├── main_diagnostics.py          # 主入口：掃描設備 + 即時畫面 + 狀態面板
│   │
│   ├── src/
│   │   ├── device_scanner.py        # 掃描枚舉所有可用相機設備
│   │   ├── camera_tester.py         # 單台相機測試（分辨率、FPS、亮度、清晰度）
│   │   ├── feed_display.py          # 多相機即時畫面拼接顯示
│   │   ├── status_panel.py          # 設備狀態面板渲染（純 OpenCV 繪圖）
│   │   └── hw_report.py             # 生成診斷報告 JSON + 截圖
│   │
│   └── outputs/
│       └── reports/                 # YYYYMMDD_HHMMSS_hw_report.json + 截圖
│
│
│ ════════════════════════════════════════════════
│  PHASE 1 — 相機內參標定
│ ════════════════════════════════════════════════
├── phase1_intrinsics/
│   ├── main_intrinsics.py         # 主入口：互動採集 + 標定 + 驗收
│   │
│   ├── src/                       # 功能模塊
│   │   ├── board_generator.py     # 生成 ChArUco 標定板
│   │   ├── image_collector.py     # 互動式採集（SPACE 拍照）
│   │   ├── charuco_detector.py    # ChArUco 角點偵測 + cornerSubPix 精化
│   │   ├── calibrator.py          # calibrateCameraCharuco 封裝
│   │   ├── intrinsics_io.py       # intrinsics.json 合併式讀寫
│   │   └── validator.py           # 重投影誤差可視化與驗收報告
│   │
│   └── outputs/                   # 結果存放
│       ├── raw_images/            # 採集的原始標定圖像
│       │   ├── cam0/
│       │   ├── cam1/
│       │   ├── cam2/
│       │   └── cam3/
│       ├── intrinsics.json        # 最終內參（K, D, image_size, rms）
│       └── reports/               # 每台相機的重投影誤差圖
│           ├── cam0_reprojection_errors.png
│           └── ...
│
│
│ ════════════════════════════════════════════════
│  PHASE 2 — 相機外參標定
│ ════════════════════════════════════════════════
├── phase2_extrinsics/
│   ├── main_extrinsics.py           # 主入口：雙目外參標定 GUI
│   │
│   ├── src/
│   │   ├── stereo_calibrator.py     # stereoCalibrate 封裝 + 校正可視化
│   │   ├── bundle_adjustment.py     # scipy least_squares Bundle Adjustment（多相機用）
│   │   ├── extrinsics_io.py         # extrinsics.json 合併式讀寫
│   │   └── tk_gui.py               # tkinter GUI（配對選擇、採集、驗收）
│   │
│   └── outputs/
│       ├── raw_pairs/               # 同步採集的圖像對
│       │   ├── cam0_cam1/left, right
│       │   └── ...
│       ├── extrinsics.json          # 外參標定結果（R, T, rms）
│       └── reports/
│           └── cam0_cam1_epipolar_errors.png
│
│
│ ════════════════════════════════════════════════
│  PHASE 3 — 雙目立體深度
│  Fast-FoundationStereo → Metric 深度圖
│ ════════════════════════════════════════════════
├── phase3_stereo_depth/
│   ├── main_stereo_depth.py         # 主入口：校正立體對 → 推理 → metric 深度圖
│   │
│   ├── src/
│   │   ├── stereo_rectifier.py      # stereoRectify + initUndistortRectifyMap
│   │   ├── stereo_inference.py      # Fast-FoundationStereo 推理封裝
│   │   │                            # 輸入：校正後左右圖 → 輸出：disparity map
│   │   ├── depth_converter.py       # disparity → metric depth（利用 baseline + focal length）
│   │   └── depth_utils.py           # depth_to_pointcloud、深度圖可視化工具
│   │
│   └── outputs/
│       ├── stereo_depth/            # metric 深度圖（.npy）
│       │   └── YYYYMMDD_HHMMSS_stereo.npy
│       └── disparity/               # 視差圖（可視化用）
│           └── YYYYMMDD_HHMMSS_disparity.png
│
│
│ ════════════════════════════════════════════════
│  PHASE 4 — 多視角重建
│  DA3Nested 多視角推理 → 統一 Metric 點雲
│ ════════════════════════════════════════════════
├── phase4_multi_view_recon/
│   ├── main_multi_view.py           # 主入口：多相機幀 → DA3 推理 → 尺度對齊 → 融合點雲
│   │
│   ├── src/
│   │   ├── camera_capture.py        # 多相機同步讀幀（VideoCapture 封裝）
│   │   ├── da3_inference.py         # DA3Nested 多視角推理封裝
│   │   │                            # 輸入：各相機圖像 → 輸出：相對深度圖
│   │   ├── scale_aligner.py         # RANSAC 尺度對齊（DA3 相對深度 → metric，
│   │   │                            #   以 phase3 雙目 metric 深度為錨點）
│   │   ├── pointcloud_fusion.py     # 多視角 metric 點雲融合、濾波、下採樣
│   │   └── depth_utils.py           # depth_to_pointcloud 工具（共用或引用 phase3）
│   │
│   └── outputs/
│       ├── frames/                  # 每次推理的原始幀存檔（可選）
│       │   └── YYYYMMDD_HHMMSS/
│       │       ├── cam0.jpg
│       │       ├── cam1.jpg
│       │       ├── cam2.jpg
│       │       └── cam3.jpg
│       ├── depth_maps/              # 各相機 metric 深度圖（.npy）
│       │   └── YYYYMMDD_HHMMSS/
│       └── pointclouds/             # 融合後 metric 點雲（.ply）
│           └── YYYYMMDD_HHMMSS_scene.ply
│
│
│ ════════════════════════════════════════════════
│  PHASE 5 — VLM 規劃（Agentic Loop）
│
│  完整執行流程：
│
│  ① 任務輸入
│     文字輸入 / 語音輸入（STT）→ 任務文字
│              ↓
│  ② VLM 解析目標
│     VLM 理解任務，提取：
│       - 目標物件描述（object_description）
│       - 操作意圖（pick / place / move ...）
│              ↓
│  ③ skill-SAM3（目標定位）
│     輸入：RGB 圖像 + object_description
│     輸出：標注圖像（annotated_image）+ masks + scores
│              ↓
│  ④ VLM 驗證標注（最多重試 N 次）
│     VLM 觀看標注圖像，判斷是否有明顯錯誤
│     ├─ 正確 → 進入步驟 ⑤
│     └─ 錯誤 → 修正 object_description → 回到步驟 ③
│              ↓
│  ⑤ skill-CapturePoint（抓取點計算）
│     核心算法：GraspGen — 職責僅限「在哪裡抓、朝哪個方向抓、夾多寬」
│     輸入：SAM3 best_mask → 裁剪 phase4 metric 點雲（XYZRGB）
│           + T_cam2arm（phase2 hand-eye 標定）
│           + workspace_limits（工作台範圍）
│     輸出：
│       - grasp_pose_arm : (4×4) arm 座標系下的完整抓取位姿
│                          ← GraspGen 直接輸出，包含位置 + 方向
│       - grasp_width    : (float) 夾爪張開寬度（公尺）
│       - grasp_score    : (float) GraspGen 置信度
│       - grasp_pixel    : (u, v) 像素座標（用於 VLM 可視化）
│     ⚠️ GraspGen 不規劃軌跡，只輸出目標位姿
│              ↓
│  ⑥ VLM 最終確認
│     VLM 同時接收：
│       - 標注圖像（SAM3 標注 + 抓取點疊加可視化）
│       - grasp_pose_arm 座標（文字描述：位置 xyz + 方向）
│       - grasp_width（夾爪寬度，判斷是否合理）
│       - 場景點雲截圖（空間合理性參考）
│     VLM 判斷座標與方向是否合理
│     ├─ 合理 → 進入步驟 ⑦
│     └─ 不合理 → 回到步驟 ③ 重新定位（最多重試 N 次）
│              ↓
│  ⑦ skill-TrajectoryPlanning（軌跡規劃）
│     核心算法：cuRobo — 職責僅限「如何到達 GraspGen 給出的目標位姿」
│     輸入：arm 當前關節狀態（start_state）
│           + grasp_pose_arm（來自 GraspGen 的 4×4 目標位姿）
│           + world_config（從 phase4 點雲生成的碰撞體素）
│     輸出：
│       - joint_trajectory  : 關節軌跡 waypoints（無碰撞）
│       - trajectory_preview: 軌跡預覽圖（供確認）
│       - estimated_duration: 預估執行時長（秒）
│     ⚠️ cuRobo 不估計抓取位姿，只規劃到達路徑
│              ↓
│  ⑧ 輸出執行計劃 → phase6
│     plan.json 包含：
│       - grasp_pose_arm（4×4，GraspGen 輸出）
│       - grasp_width（夾爪寬度，phase6 用於控制夾爪）
│       - joint_trajectory（cuRobo 輸出）
│       - 任務描述（用於日誌）
│ ════════════════════════════════════════════════
├── phase5_vlm_planning/
│   ├── main_planning.py             # 主入口：任務輸入 → agentic loop → 輸出 plan.json
│   │
│   ├── skills/                      # VLM 可調用的技能模塊（統一接口）
│   │   │
│   │   ├── skill_sam3.py            # skill-SAM3
│   │   │   # input : rgb_image (np.ndarray)
│   │   │   #         object_description (str)
│   │   │   # output: annotated_image (np.ndarray)
│   │   │   #         masks (list[np.ndarray])
│   │   │   #         scores (list[float])
│   │   │   #         best_mask (np.ndarray)  ← 最高信心 mask
│   │   │
│   │   ├── skill_capture_point/    # skill-CapturePoint
│   │   │   │   # 核心算法：GraspGen（NVlabs/GraspGen, ICRA 2026）
│   │   │   │   #   SE(3) Diffusion-based 6-DoF 抓取位姿估計，直接吃 metric 點雲
│   │   │   │   #   職責：「在哪裡抓 + 朝哪個方向抓 + 夾多寬」，不規劃軌跡
│   │   │   │   #   GitHub: https://github.com/NVlabs/GraspGen
│   │   │   │   #
│   │   │   ├── __init__.py
│   │   │   ├── capture_point_skill.py   # 核心類別：載入 GraspGen + 推理
│   │   │   ├── pointcloud_cropper.py    # mask → 裁剪點雲 + 座標變換
│   │   │   └── grasp_visualizer.py      # 抓取位姿可視化
│   │   │   #
│   │   │   # input : rgb_image (np.ndarray BGR)
│   │   │   #         depth (np.ndarray float32 metric)
│   │   │   #         best_mask (np.ndarray bool)     ← 來自 skill-SAM3
│   │   │   #         K (3×3 內參)
│   │   │   #         T_cam2arm (4×4)                 ← 來自 phase2 hand-eye 標定
│   │   │   # output: grasp_pose_arm (4×4 float)      ← GraspGen 輸出的完整位姿
│   │   │   #           包含位置（translation）+ 方向（rotation_matrix）
│   │   │   #           ⚡ 此矩陣直接作為 skill-TrajectoryPlanning 的 target_pose
│   │   │   #         grasp_width   (float, 公尺)     ← 夾爪張開寬度，發給 gripper
│   │   │   #         grasp_score   (float)           ← GraspGen 置信度
│   │   │   #         grasp_pixel   (u: int, v: int)  ← 用於 VLM 可視化疊加
│   │   │
│   │   └── skill_trajectory_planning.py  # skill-TrajectoryPlanning
│   │       # 核心算法：cuRobo（NVlabs/curobo）
│   │       #   GPU 加速軌跡規劃，無 ROS 依賴，支援碰撞避免
│   │       #   職責：「如何從當前狀態到達 GraspGen 給出的目標位姿」，不估計抓取點
│   │       #   GitHub: https://github.com/NVlabs/curobo
│   │       #   文檔:   https://curobo.org
│   │       #
│   │       # input : start_state   (arm 當前關節角度向量)
│   │       #         target_pose   (grasp_pose_arm 4×4) ← 直接來自 skill-CapturePoint
│   │       #         world_config  (dict: 從 phase4 點雲生成的碰撞體素)
│   │       # output: joint_trajectory     (list of waypoints, 每個 waypoint 為關節角度向量)
│   │       #         trajectory_preview   (np.ndarray, 可視化圖)
│   │       #         estimated_duration   (float, 秒)
│   │
│   ├── src/
│   │   ├── task_receiver.py         # 任務輸入介面
│   │   │                            #   - 文字輸入（stdin / socket）
│   │   │                            #   - 語音輸入（Whisper STT → 文字）
│   │   │
│   │   ├── vlm_client.py            # VLM API 統一封裝
│   │   │                            #   - call_vlm(prompt, images) → str
│   │   │                            #   - 支援 Anthropic / OpenAI / 本地模型
│   │   │
│   │   ├── vlm_verifier.py          # VLM 驗證邏輯（獨立模塊，方便單獨測試）
│   │   │                            #   - verify_sam3_result(annotated_image, description)
│   │   │                            #       → (is_correct: bool, feedback: str)
│   │   │                            #   - verify_grasp_point(annotated_image, grasp_arm, scene_img)
│   │   │                            #       → (is_correct: bool, feedback: str)
│   │   │
│   │   ├── object_parser.py         # VLM 輸出解析
│   │   │                            #   - parse_object_description(vlm_output)
│   │   │                            #       → object_description (str)
│   │   │                            #   - parse_plan(vlm_output) → plan dict
│   │   │
│   │   ├── loop_controller.py       # Agentic loop 控制器
│   │   │                            #   - 管理重試次數、狀態機、超時
│   │   │                            #   - MAX_RETRY_SAM3 = 3
│   │   │                            #   - MAX_RETRY_GRASP = 2
│   │   │
│   │   └── plan_serializer.py       # 執行計劃序列化 → plan.json
│   │
│   └── outputs/
│       ├── session/                 # 每次任務的完整過程存檔
│       │   └── YYYYMMDD_HHMMSS/
│       │       ├── task_input.txt           # 原始任務文字
│       │       ├── rgb_snapshot.jpg         # 任務觸發時的場景截圖
│       │       ├── sam3_annotated.jpg        # SAM3 標注結果
│       │       ├── grasp_point_viz.jpg       # 抓取點可視化（疊加在 RGB 上）
│       │       ├── trajectory_preview.jpg    # 軌跡預覽圖
│       │       └── vlm_dialogue.jsonl        # 完整 VLM 對話記錄
│       │
│       └── plans/                   # 最終執行計劃（供 phase6 讀取）
│           └── YYYYMMDD_HHMMSS_plan.json
│
│
│ ════════════════════════════════════════════════
│  PHASE 6 — 機械臂執行
│ ════════════════════════════════════════════════
├── phase6_arm_execution/
│   ├── main_execution.py          # 主入口：接收 plan.json → 驅動機械臂
│   │
│   ├── src/
│   │   ├── arm_controller.py      # 機械臂 SDK 封裝（關節控制、末端位姿控制）
│   │   ├── trajectory_executor.py # 執行 plan.json 中的 joint_trajectory waypoints
│   │   │                          # （軌跡由 phase5 cuRobo 規劃，此處只負責逐點發送）
│   │   ├── gripper_controller.py  # 夾爪控制（張開寬度來自 plan.json grasp.width）
│   │   ├── safety_monitor.py      # 執行安全監控（碰撞偵測、限位保護、力矩超限）
│   │   └── feedback_listener.py   # 執行反饋（到位確認、力矩監控、夾取成功判斷）
│   │
│   └── outputs/
│       ├── execution_logs/        # 每次執行的詳細日誌（.jsonl）
│       │   └── YYYYMMDD_HHMMSS_exec.jsonl
│       └── trajectories/          # 記錄的關節軌跡（用於回放/分析）
│
│
│ ════════════════════════════════════════════════
│  SHARED — 跨階段共用
│ ════════════════════════════════════════════════
└── shared/
    ├── calib_loader.py            # 統一載入 intrinsics.json + extrinsics.json
    ├── camera_manager.py          # 全局相機管理（開啟/關閉/同步讀幀）
    ├── tk_utils.py                # tkinter 共用工具（樣式、CameraFeedWidget 等）
    ├── logger.py                  # 統一日誌格式
    ├── timer.py                   # 推理計時工具
    └── types.py                   # 共用 dataclass
                                   #   CalibResult、PointCloud、
                                   #   GraspPoint、Trajectory、Plan 等
```

---

## 各 Phase 主文件職責說明

| 主文件 | 輸入 | 輸出 | 觸發方式 |
|---|---|---|---|
| `phase0_hw_diagnostics/main_diagnostics.py` | `settings.yaml` | 診斷報告 `.json` + 截圖 | 部署前一次性執行（最先） |
| `phase1_intrinsics/main_intrinsics.py` | 相機索引 | `intrinsics.json` | 部署前一次性執行 |
| `phase2_extrinsics/main_extrinsics.py` | `intrinsics.json` + 相機對 | `extrinsics.json` | 部署前一次性執行 |
| `phase3_stereo_depth/main_stereo_depth.py` | 標定結果 + 雙目幀 | metric 深度圖 `.npy` | 每次任務執行時調用 |
| `phase4_multi_view_recon/main_multi_view.py` | 標定結果 + 多相機幀 + phase3 深度 | metric 點雲 `.ply` | 每次任務執行時調用 |
| `phase5_vlm_planning/main_planning.py` | 任務文字/語音 + 場景 RGB + 深度圖 | `plan.json`（軌跡 + 抓取點） | 每次任務執行時調用 |
| `phase6_arm_execution/main_execution.py` | `plan.json` | 機械臂動作 + 執行日誌 | 每次任務執行時調用 |

---

## 運行時調用順序

```
[部署前，一次性]
python phase0_hw_diagnostics/main_diagnostics.py  # 先確認硬體正常
python phase1_intrinsics/main_intrinsics.py
python phase2_extrinsics/main_extrinsics.py

[每次執行任務]
python phase3_stereo_depth/main_stereo_depth.py          # 雙目 → metric 深度圖
    → python phase4_multi_view_recon/main_multi_view.py  # DA3 多視角 → metric 點雲
        → python phase5_vlm_planning/main_planning.py    # Agentic loop → plan.json
            → python phase6_arm_execution/main_execution.py  # 機械臂執行

[或統一由頂層 orchestrator 串接，待後續設計]
```

---

## Phase 5 Agentic Loop 狀態機

```
                    ┌─────────────────┐
                    │   任務輸入       │ ← 文字 / 語音
                    └────────┬────────┘
                             ↓
                    ┌─────────────────┐
                    │  VLM 解析目標    │ → object_description
                    └────────┬────────┘
                             ↓
              ┌──────────────────────────┐
              │       skill-SAM3         │
              │  rgb + description → mask │
              └──────────────┬───────────┘
                             ↓
              ┌──────────────────────────┐
              │    VLM 驗證標注結果       │
              └───┬──────────────────────┘
         正確 ←──┤                   錯誤（修正 description，重試 ≤3）
                 ↓
              ┌──────────────────────────────────────┐
              │         skill-CapturePoint            │
              │  [GraspGen]                           │
              │  mask → 裁剪點雲 → 6-DoF grasp poses  │
              │  → 選最高分 → grasp_arm（arm 座標系）  │
              └──────────────┬───────────────────────┘
                             ↓
              ┌──────────────────────────┐
              │  VLM 確認座標合理性       │
              └───┬──────────────────────┘
         合理 ←──┤                   不合理（重試 ≤2，回到 SAM3）
                 ↓
              ┌──────────────────────────────────────┐
              │      skill-TrajectoryPlanning         │
              │  [cuRobo]                             │
              │  start_state + grasp_arm → 軌跡規劃   │
              │  GPU 加速，含碰撞避免                  │
              └──────────────┬───────────────────────┘
                             ↓
                    ┌─────────────────┐
                    │  輸出 plan.json  │ → phase6
                    └─────────────────┘
```

## skill 核心算法依賴

| skill | 算法庫 | GitHub |
|---|---|---|
| skill-SAM3 | Meta SAM 3 | github.com/facebookresearch/sam3 |
| skill-CapturePoint | GraspGen (NVIDIA, ICRA 2026) | github.com/NVlabs/GraspGen |
| skill-TrajectoryPlanning | cuRobo | github.com/NVlabs/curobo |

GraspGen 補充說明：開源無 license 限制，SE(3) Diffusion-based（DDPM + PointTransformerV3），20Hz 推理，FetchBench 高 17%（vs AnyGrasp），記憶體少 21×，支援 Robotiq-2F-140 / Franka Panda / Suction 三種夾爪 checkpoint，直接輸入 metric 點雲，輸出所有候選抓取位姿並按置信度排序。

cuRobo 補充說明：無 ROS 依賴，GPU 加速（毫秒級規劃），可直接消費 GraspGen 輸出的 6-DoF 位姿，支援從點雲生成碰撞體素作為場景約束。

---

## plan.json 格式規範

```json
{
  "task_id": "YYYYMMDD_HHMMSS",
  "task_description": "把紅色瓶子移到托盤上",
  "object_description": "紅色瓶子",
  "grasp": {
    "pixel": { "u": 342, "v": 287 },
    "pose_arm": [
      [0.0,  0.0,  1.0,  0.341],
      [0.0, -1.0,  0.0,  0.127],
      [1.0,  0.0,  0.0,  0.198],
      [0.0,  0.0,  0.0,  1.0  ]
    ],
    "width": 0.052,
    "score": 0.87
  },
  "trajectory": {
    "waypoints": [
      { "joints": [0.1, -0.5, 0.3, 1.2, 0.0, 0.8, 0.0] },
      "..."
    ],
    "estimated_duration_sec": 3.2
  },
  "session_dir": "phase5_vlm_planning/outputs/session/YYYYMMDD_HHMMSS"
}
```

欄位說明：

| 欄位 | 來源 | phase6 用途 |
|---|---|---|
| `grasp.pose_arm` | GraspGen 輸出（4×4） | 末端位姿目標 |
| `grasp.width` | GraspGen 輸出 | 控制夾爪張開寬度 |
| `grasp.score` | GraspGen 置信度 | 日誌記錄 |
| `trajectory.waypoints` | cuRobo 輸出 | 關節軌跡執行 |

---

## Phase 3 → Phase 4 數據流

```
Phase 3 (雙目立體深度)             Phase 4 (多視角重建)
┌─────────────────────┐           ┌──────────────────────────┐
│ 校正立體對           │           │ 各相機圖像（cam0~cam3）    │
│      ↓              │           │      ↓                   │
│ Fast-FoundationStereo│          │ DA3Nested 推理            │
│      ↓              │           │      ↓                   │
│ disparity map       │           │ 相對深度圖（各相機）        │
│      ↓              │           │      ↓                   │
│ metric 深度圖 ──────────────────→ 尺度對齊（RANSAC）         │
│ （baseline + K 已知）│           │      ↓                   │
└─────────────────────┘           │ metric 點雲（各相機）      │
                                  │      ↓                   │
                                  │ 多視角融合 + 濾波          │
                                  │      ↓                   │
                                  │ 統一 metric 點雲 (.ply)   │
                                  └──────────────────────────┘
```

---

## outputs/ 命名規範

- 標定結果：固定文件名（`intrinsics.json`、`extrinsics.json`），覆蓋更新
- 推理結果：帶時間戳（`YYYYMMDD_HHMMSS`），永久保留，便於回溯
- 日誌：`.jsonl` 格式，每行一條記錄，便於流式寫入

---

## config/settings.yaml 示例

```yaml
cameras:
  cam0: { index: 0, role: stereo_left }
  cam1: { index: 1, role: stereo_right }
  cam2: { index: 2, role: aux }
  cam3: { index: 3, role: aux }

phase0:
  scan_max_index: 16
  test_frames: 30
  tile_size: [640, 480]
  sharpness_threshold: 50.0
  brightness_range: [30, 230]

calibration:
  charuco:
    cols: 9
    rows: 6
    square_size: 0.03
    marker_size: 0.0223
    aruco_dict: DICT_4X4_50
  acceptance:
    max_rms_intrinsics: 1.0
    max_rms_extrinsics: 1.0
    max_ba_residual: 0.5
    max_pointcloud_error: 0.005

stereo_depth:
  model: "Fast-FoundationStereo"
  max_disparity: 256

multi_view_recon:
  da3_model: "depth-anything/DA3NESTED-GIANT-LARGE"
  export_format: "ply"

vlm_planning:
  provider: "anthropic"
  model: "claude-opus-4-6"
  max_tokens: 4096
  max_retry_sam3: 3
  max_retry_grasp: 2
  stt_model: "whisper-large-v3"

skill_capture_point:
  algorithm: "graspgen"
  gripper_config: "external/GraspGen/models/checkpoints/graspgen_robotiq_2f_140.yml"
  num_grasp_candidates: 10       # 取前 N 個候選抓取位姿
  workspace_limits:              # 工作台範圍裁剪（公尺，arm 座標系）
    x: [-0.5, 0.5]
    y: [-0.5, 0.5]
    z: [0.0, 0.6]

skill_trajectory_planning:
  algorithm: "curobo"
  robot_config: "config/robot_curobo.yml"  # 機械臂 URDF + 運動學參數
  collision_from_pointcloud: true           # 從 phase4 點雲生成碰撞體素
  voxel_size: 0.02                          # 碰撞體素尺寸（公尺）
  interpolation_steps: 100                  # 軌跡插值點數

arm:
  sdk: "xarm"
  ip: "192.168.1.xxx"
  base_frame: "world"
```