from __future__ import annotations

from pathlib import Path
import os
from app.core.models import Segment, TranscriptResult


REPO_MAP = {
    'large-v3': 'OpenVINO/whisper-large-v3-int8-ov',
    'medium': 'OpenVINO/whisper-medium-int8-ov',
    'small': 'OpenVINO/whisper-small-int8-ov',
    'base': 'OpenVINO/whisper-base-int8-ov',
}


def _model_root(model_name: str) -> Path:
    try:
        from platformdirs import user_data_dir
        base = Path(user_data_dir('JunbaAITranscriber', 'Junba'))
    except Exception:
        base = Path.home() / '.junba_ai_transcriber'
    return base / 'models' / 'openvino' / model_name


def ensure_openvino_model(model_name: str, local_model_dir: str = '', progress_cb=None) -> str:
    # If user explicitly selected an OpenVINO IR model directory, reuse it.
    if local_model_dir:
        p = Path(local_model_dir)
        if p.is_dir() and any(p.glob('openvino_*model.xml')):
            return str(p)
    if model_name not in REPO_MAP:
        raise ValueError(f'OpenVINO 尚未設定此 Whisper 模型：{model_name}')
    target = _model_root(model_name)
    # Whisper OpenVINO repos contain encoder/decoder XMLs. If present, no network is needed.
    if target.is_dir() and any(target.glob('openvino_*model.xml')):
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(2, f'第一次使用 OpenVINO {model_name}，下載 INT8 模型中…')
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=REPO_MAP[model_name],
        local_dir=str(target),
    )
    if not any(target.glob('openvino_*model.xml')):
        raise RuntimeError(f'OpenVINO 模型下載完成但找不到 IR XML：{target}')
    if progress_cb:
        progress_cb(10, f'OpenVINO 模型已準備：{target}')
    return str(target)


def _decode_16k_mono(path: str, progress_cb=None):
    """Decode MP3/M4A/etc. to normalized mono float32 @16 kHz via bundled FFmpeg.

    Returns a NumPy float32 array. This is much more memory-efficient than a
    Python list for 10–30 minute chunks, which matters on 16 GB systems.
    """
    import subprocess
    import numpy as np
    from app.core.audio_tools import ffmpeg_exe
    if progress_cb:
        progress_cb(12, '解碼音訊為 16 kHz 單聲道')
    cmd = [
        ffmpeg_exe(), '-v', 'error', '-i', str(path),
        '-vn', '-ac', '1', '-ar', '16000',
        '-f', 's16le', '-acodec', 'pcm_s16le', 'pipe:1',
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=None)
    if r.returncode != 0:
        err = r.stderr.decode('utf-8', errors='replace')[-3000:]
        raise RuntimeError(f'FFmpeg 音訊解碼失敗：{err}')
    if not r.stdout:
        return np.empty((0,), dtype=np.float32)
    pcm = np.frombuffer(r.stdout, dtype='<i2')
    return (pcm.astype(np.float32) / 32768.0)



class OpenVINOWhisperProvider:
    """Whisper on Intel GPU/NPU using OpenVINO GenAI.

    Model weights are downloaded once from the official OpenVINO Hugging Face
    repositories and then remain locally reusable/offline.
    """

    def __init__(self, model_name='large-v3', device='NPU', local_model_dir='', progress_cb=None):
        import openvino as ov
        import openvino_genai as ov_genai

        self.device = device.upper()
        self.model_name = model_name
        core = ov.Core()
        available = [str(d).upper() for d in core.available_devices]
        if not any(d.startswith(self.device) for d in available):
            raise RuntimeError(f'OpenVINO 找不到 {self.device}。目前裝置：{core.available_devices}')

        self.model_dir = ensure_openvino_model(model_name, local_model_dir, progress_cb)
        cache_dir = str(_model_root(model_name) / f'compiled_cache_{self.device.lower()}')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        kwargs = {'CACHE_DIR': cache_dir}
        if progress_cb:
            progress_cb(15, f'正在編譯 Whisper 至 {self.device}；第一次可能較久…')
        # Segment timestamps are enough for SRT/VTT and have lower overhead than word timestamps.
        self.pipe = ov_genai.WhisperPipeline(self.model_dir, self.device, **kwargs)
        if progress_cb:
            progress_cb(20, f'OpenVINO {self.device} 已就緒')

    def transcribe(self, path: str, language: str | None = None, stop_flag=None,
                   progress_cb=None, wait_cb=None) -> TranscriptResult:
        if wait_cb:
            wait_cb()
        if stop_flag and stop_flag():
            return TranscriptResult('', [], engine=f'OpenVINO Whisper ({self.device})')
        audio = _decode_16k_mono(path, progress_cb)
        if len(audio) == 0:
            raise RuntimeError('音訊解碼後沒有資料')
        if progress_cb:
            progress_cb(25, f'OpenVINO {self.device} 推論中')
        kwargs = {'task': 'transcribe', 'return_timestamps': True}
        if language and language != 'auto':
            kwargs['language'] = language
        result = self.pipe.generate(audio, **kwargs)
        text = ''
        try:
            texts = list(result.texts)
            text = texts[0] if texts else ''
        except Exception:
            text = str(result)
        segs: list[Segment] = []
        try:
            chunks = result.chunks or []
        except Exception:
            chunks = []
        for c in chunks:
            txt = str(getattr(c, 'text', '') or '').strip()
            if txt:
                segs.append(Segment(float(getattr(c, 'start_ts', 0.0)),
                                    float(getattr(c, 'end_ts', 0.0)), txt))
        if not segs and text.strip():
            segs = [Segment(0.0, 0.0, text.strip())]
        if progress_cb:
            progress_cb(100, f'OpenVINO {self.device} 轉錄完成')
        lang = getattr(result, 'language', None)
        return TranscriptResult(text.strip(), segs, lang, f'OpenVINO GenAI Whisper ({self.device})')
