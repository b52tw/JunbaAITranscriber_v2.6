# Junba AI Transcriber v2.6

Windows 10/11 x64｜繁體中文 GUI｜離線 Whisper + Intel NPU/GPU + NVIDIA CUDA + Google Gemini。

## v2.6 主要更新

- 新增「硬體加速」選單：
  - 自動（效能優先）
  - NVIDIA GPU / CUDA
  - Intel GPU / OpenVINO
  - Intel NPU / OpenVINO（省電）
  - CPU / faster-whisper（Intel/AMD）
- 啟動/重新偵測時會列出 CPU、CUDA、OpenVINO GPU/NPU 與實際裝置名稱。
- Intel NPU/GPU 使用 **OpenVINO GenAI WhisperPipeline**。
- OpenVINO 模型採官方 `OpenVINO/whisper-*-int8-ov`：large-v3 / medium / small / base。
- OpenVINO 模型第一次使用自動下載至使用者資料夾，之後可離線重用。
- CPU 路徑繼續使用 faster-whisper/CTranslate2 INT8；Intel 與 AMD x86-64 CPU 都可用。
- NVIDIA GPU 繼續使用 faster-whisper/CTranslate2 CUDA + FP16。
- 保留 v2.5 的「立即停止並輸出目前結果」、切割/合併、Gemini、多講者、繁體中文、Word/TXT/SRT/VTT。

## 你這台 Intel AI Boost / Intel Arc 電腦

如果 OpenVINO 與 Intel 驅動正常，程式應同時看到：

- `Intel NPU / OpenVINO`（工作管理員 NPU 0 / Intel(R) AI Boost）
- `Intel GPU / OpenVINO`（Intel Arc）
- `CPU / faster-whisper（Intel）`

若要明確讓工作管理員的 NPU 使用率上升，請不要選「自動」，而是直接選：

**Intel NPU / OpenVINO（省電）**

「自動（效能優先）」的順序為：NVIDIA CUDA → Intel GPU → Intel NPU → CPU。

> 16 GB RAM 的電腦如果當下記憶體已高占用，large-v3 第一次 NPU/GPU 編譯可能較吃記憶體。可先關閉大型程式，或改用 medium / small。

## AMD / Intel CPU 相容性

CPU 模式使用 CTranslate2 預編譯 x86-64 路徑。CTranslate2 會依 CPU 自動選擇適合的指令集；Intel 與 AMD CPU 都可使用。AMD Radeon GPU **本版尚未做 GPU 加速**，AMD 電腦會使用 CPU 模式，除非另有 NVIDIA CUDA GPU。

## Intel NPU / GPU 模型

OpenVINO 模式會自動下載對應模型：

- large-v3 → `OpenVINO/whisper-large-v3-int8-ov`
- medium → `OpenVINO/whisper-medium-int8-ov`
- small → `OpenVINO/whisper-small-int8-ov`
- base → `OpenVINO/whisper-base-int8-ov`

第一次下載需要網路；完成後模型留在本機，之後可以離線辨識。

## 音檔流程

1. 加入或拖曳音檔。
2. 選擇是否先切割（2/5/10/15/30/60 分鐘或自訂）。
3. 選辨識引擎：離線 Whisper / Google Gemini / 混合模式。
4. 若使用離線 Whisper，選硬體加速。
5. 開始處理；切割會先完整完成，再開始辨識。
6. 可暫停、繼續、或「立即停止並輸出目前結果」。
7. 匯出 Word/TXT/SRT/VTT。

## Google Gemini

- 支援音訊上傳轉錄。
- 多人講者 + 字詞時間戳可同時使用；Smart 智慧逐字稿與這兩項互斥，GUI 會自動切換。
- 中文/華台混合輸出可自動轉為台灣繁體中文。
- API Key 儲存在 Windows 認證儲存區，不寫死在 EXE。

## GitHub Actions 建置

`.github/workflows/build-windows-v2.6.yml` 會在 `windows-latest`：

- 安裝 PySide6 / faster-whisper / OpenVINO / OpenVINO GenAI 等依賴
- compileall
- pytest
- OpenVINO import 與裝置列舉 smoke test
- Source GUI offscreen self-test
- 建置 Single EXE
- Single EXE self-test + UI self-test
- 建置 Portable 版
- Portable EXE self-test + UI self-test

成功後會有兩個 Artifact：

- `Junba-AI-Transcriber-v2.6-Single-EXE-Windows-x64`
- `Junba-AI-Transcriber-v2.6-Portable-Windows-x64`

先用 Single EXE；若遇到 OpenVINO/CUDA/DLL 或防毒解包問題，再改用 Portable 版。

## 注意

- NPU 需要 Windows 11 與正確的 Intel NPU 驅動；程式會以 OpenVINO 實際列出的裝置為準。
- Intel GPU 需要正確的 Intel Graphics Driver。
- 模型沒有內嵌進 EXE，所以第一次使用新的 Whisper/OpenVINO 模型需要下載。
