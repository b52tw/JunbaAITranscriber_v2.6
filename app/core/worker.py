from __future__ import annotations
import json
import hashlib
import threading
import time
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from app.core.audio_tools import ensure_split_audio, audio_duration_seconds, split_output_dir
from app.core.models import TranscriptResult, Segment
from app.providers.local_whisper import LocalWhisperProvider
from app.providers.openvino_whisper import OpenVINOWhisperProvider
from app.core.hardware import resolve_acceleration
from app.providers.gemini import GeminiProvider
from app.exporters.exporters import export_all
from app.core.text_normalize import normalize_result_traditional, to_traditional_taiwan


class TranscribeWorker(QThread):
    status = Signal(str)
    progress = Signal(int)
    stage_progress = Signal(int)
    stage_text = Signal(str)
    log = Signal(str)
    checkpoint = Signal(str)
    file_done = Signal(str)
    failed = Signal(str)
    finished_ok = Signal()
    cancelled = Signal()

    def __init__(self, files, output_dir, mode, split_minutes, model_name, local_model_dir, language,
                 api_key, diarization, timestamps, smart, traditional_output, formats, acceleration='auto'):
        super().__init__()
        self.files = files
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.split_minutes = split_minutes
        self.model_name = model_name
        self.local_model_dir = local_model_dir or ''
        self.acceleration = acceleration or 'auto'
        self.language = language
        self.api_key = api_key
        self.diarization = diarization
        self.timestamps = timestamps
        self.smart = smart
        self.traditional_output = traditional_output
        self.formats = formats
        self._pause = threading.Event()
        self._stop = threading.Event()
        self._pause.set()

    def pause(self):
        self._pause.clear()

    def resume(self):
        self._pause.set()

    def stop(self):
        # Safe-stop semantics: stop scheduling new work, keep everything already
        # recognized, and export a valid partial document before the thread exits.
        self._stop.set()
        self._pause.set()

    def _wait(self):
        while not self._pause.is_set() and not self._stop.is_set():
            time.sleep(0.1)

    def _set_stage(self, text: str, pct: int = -1):
        self.stage_text.emit(text)
        self.stage_progress.emit(pct)
        self.status.emit(text)
        self.log.emit(text)

    def _chunk_progress(self, fi: int, ci: int, chunk_count: int, total_files: int, pct: int, detail: str = ''):
        pct = max(0, min(100, int(pct)))
        self.stage_progress.emit(pct)
        if detail:
            self.stage_text.emit(detail)
        # Reserve first 15% for preprocessing, 80% for recognition, 5% for export.
        within_file = (ci + pct / 100.0) / max(1, chunk_count)
        recognize_fraction = (fi + within_file) / max(1, total_files)
        overall = 15 + int(recognize_fraction * 80)
        self.progress.emit(max(15, min(95, overall)))

    def _effective_split(self, duration: float) -> int:
        effective_split = int(self.split_minutes)
        if self.mode == 'Google Gemini':
            max_minutes = 30 if (self.diarization or self.timestamps) else 60
            if duration > max_minutes * 60 and (effective_split == 0 or effective_split > max_minutes):
                effective_split = max_minutes
                self.log.emit(f'依 Gemini 音訊上限自動切為每 {max_minutes} 分鐘。')
        return effective_split

    def _prepare_plans(self):
        """Split all source files before model/API initialization."""
        plans = []
        total_files = max(1, len(self.files))
        for fi, source in enumerate(self.files):
            if self._stop.is_set():
                break
            self._wait()
            srcp = Path(source)
            if not srcp.exists():
                raise FileNotFoundError(f'找不到音檔：{source}')
            duration = audio_duration_seconds(source)
            effective_split = self._effective_split(duration)
            if effective_split > 0 and (duration <= 0 or duration > effective_split * 60):
                split_dir = split_output_dir(str(self.output_dir), source, effective_split)
                self._set_stage(f'先切割音檔：{srcp.name}', 0)

                def split_progress(p, _fi=fi):
                    self.stage_progress.emit(p)
                    self.progress.emit(min(14, int(((_fi + p / 100.0) / total_files) * 15)))

                chunks = ensure_split_audio(
                    source, split_dir, effective_split,
                    progress_cb=split_progress,
                    cancel_cb=self._stop.is_set,
                )
                self.log.emit(f'切割完成：{len(chunks)} 段｜{split_dir}')
            else:
                chunks = [source]
                self.progress.emit(min(14, int(((fi + 1) / total_files) * 15)))
                if effective_split > 0:
                    self.log.emit(f'{srcp.name} 未超過 {effective_split} 分鐘，不需實際切割。')
            plans.append({
                'source': source,
                'duration': duration,
                'effective_split': effective_split,
                'chunks': chunks,
            })
        self.progress.emit(15)
        return plans

    def _compose_result(self, all_text, all_segments, engine_suffix='') -> TranscriptResult:
        final_text = '\n'.join(x for x in all_text if x)
        if self.traditional_output and self.language in ('zh', 'auto'):
            final_text = to_traditional_taiwan(final_text)
            for seg in all_segments:
                seg.text = to_traditional_taiwan(seg.text)
        engine = self.mode + engine_suffix
        return TranscriptResult(final_text, list(all_segments), engine=engine)

    def _export_current(self, srcp: Path, all_text, all_segments, partial=False) -> list[str]:
        if not all_text and not all_segments:
            return []
        suffix = '_中止版_逐字稿' if partial else '_逐字稿'
        note = None
        if partial:
            note = (
                '【中止版】使用者已按「立即停止並輸出目前結果」。'
                '本檔只包含停止前已完成或已取得的辨識內容；尚未完成的區段不會出現在本檔。'
            )
        result = self._compose_result(
            all_text,
            all_segments,
            engine_suffix='（中止版／部分結果）' if partial else '',
        )
        base = self.output_dir / f'{srcp.stem}{suffix}'
        paths = export_all(result, str(base), self.formats, note=note)
        if paths:
            self.file_done.emit('\n'.join(paths))
        return paths

    def run(self):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._set_stage('準備音檔；需要切割的檔案會先完成切割', 0)
            plans = self._prepare_plans()
            if self._stop.is_set():
                self._set_stage('已停止；尚未開始辨識，因此沒有可輸出的逐字稿。', 0)
                self.cancelled.emit()
                return

            local = None
            gemini = None
            resolved_accel = None
            if self.mode in ('離線 Whisper', '混合模式'):
                resolved_accel = resolve_acceleration(self.acceleration)
                self.log.emit(f'硬體加速：要求={self.acceleration}｜實際={resolved_accel}')
                if resolved_accel in ('openvino_npu', 'openvino_gpu'):
                    ov_device = 'NPU' if resolved_accel == 'openvino_npu' else 'GPU'
                    self._set_stage(f'載入 OpenVINO Whisper 至 {ov_device}；第一次會下載/編譯模型…', -1)
                    local = OpenVINOWhisperProvider(
                        self.model_name,
                        device=ov_device,
                        local_model_dir=self.local_model_dir,
                        progress_cb=lambda p, d='': (
                            self.stage_progress.emit(p),
                            self.stage_text.emit(d),
                            self.log.emit(d) if d else None,
                        ),
                    )
                    self._set_stage(f'Whisper 已就緒：OpenVINO {ov_device}', 100)
                else:
                    model_source = self.local_model_dir if self.local_model_dir else self.model_name
                    device = 'cuda' if resolved_accel == 'cuda' else 'cpu'
                    compute = 'float16' if device == 'cuda' else 'int8'
                    self._set_stage(f'載入 faster-whisper：{device}/{compute}；第一次下載模型可能需要數分鐘…', -1)
                    local = LocalWhisperProvider(model_source, device=device, compute_type=compute)
                    self._set_stage(f'Whisper 已就緒：{local.device} / {local.compute_type}', 100)
            if self.mode in ('Google Gemini', '混合模式'):
                self._set_stage('初始化 Google Gemini…', -1)
                gemini = GeminiProvider(self.api_key)
                self._set_stage('Google Gemini 已就緒', 100)

            total_files = max(1, len(plans))
            stopped = False
            for fi, plan in enumerate(plans):
                if self._stop.is_set():
                    stopped = True
                    break
                self._wait()
                source = plan['source']
                srcp = Path(source)
                effective_split = plan['effective_split']
                chunks = plan['chunks']

                try:
                    sig = f'{srcp.resolve()}|{srcp.stat().st_size}|{srcp.stat().st_mtime_ns}|{self.mode}|{self.model_name}|{self.local_model_dir}|{self.acceleration}|{effective_split}|{self.language}|{self.diarization}|{self.timestamps}|{self.smart}|{self.traditional_output}'
                except Exception:
                    sig = f'{source}|{self.mode}|{self.model_name}|{self.local_model_dir}|{self.acceleration}|{effective_split}|{self.language}|{self.diarization}|{self.timestamps}|{self.smart}|{self.traditional_output}'
                profile = hashlib.sha1(sig.encode('utf-8')).hexdigest()[:12]
                work = self.output_dir / '.junba_cache' / f'{srcp.stem}_{profile}'
                work.mkdir(parents=True, exist_ok=True)

                all_segments = []
                all_text = []
                running_offset = 0.0
                completed_chunks = 0

                for ci, chunk in enumerate(chunks):
                    if self._stop.is_set():
                        stopped = True
                        break
                    self._wait()
                    cache_file = work / f'result_{ci:03d}.json'
                    r = None
                    result_is_complete_chunk = True

                    if cache_file.exists():
                        self._set_stage(f'讀取快取：{ci+1}/{len(chunks)}', 100)
                        r = self._load_cached_result(cache_file)
                        if self.traditional_output and self.language in ('zh', 'auto'):
                            r = normalize_result_traditional(r)
                        self._chunk_progress(fi, ci, len(chunks), total_files, 100, '已載入快取')
                    else:
                        self._set_stage(f'辨識 {srcp.name}｜區段 {ci+1}/{len(chunks)}', 0)
                        cb = lambda p, d='', _fi=fi, _ci=ci, _n=len(chunks): self._chunk_progress(_fi, _ci, _n, total_files, p, d)
                        if self.mode == 'Google Gemini':
                            lang_codes = self._gemini_language_codes(self.language)
                            # Gemini returns a chunk atomically. If stop is pressed while the
                            # network call is in flight, we can only stop after this call returns.
                            r = gemini.transcribe(
                                chunk, self.diarization, self.timestamps, self.smart,
                                language_codes=lang_codes, progress_cb=cb,
                            )
                            result_is_complete_chunk = True
                        else:
                            lang = None if self.language == 'auto' else self.language
                            r = local.transcribe(
                                chunk, lang, stop_flag=self._stop.is_set,
                                progress_cb=cb, wait_cb=self._wait,
                            )
                            # LocalWhisperProvider returns what it has accumulated even when
                            # stop_flag becomes true; treat that result as partial and do not
                            # overwrite the full-chunk cache.
                            result_is_complete_chunk = not self._stop.is_set()

                        if r and self.traditional_output and self.language in ('zh', 'auto'):
                            r = normalize_result_traditional(r)
                            self.log.emit('已轉為繁體中文（台灣用字）。')
                        if r and result_is_complete_chunk:
                            self._save_cached_result(cache_file, r)

                    if r:
                        for s in r.segments:
                            all_segments.append(Segment(
                                s.start + running_offset,
                                s.end + running_offset,
                                s.text,
                                s.speaker,
                            ))
                        if r.text:
                            all_text.append(r.text)

                    chunk_duration = max(0.0, audio_duration_seconds(chunk))
                    # Only advance to the next full chunk offset when this chunk completed.
                    # For a partial local Whisper chunk, its segments already carry the correct
                    # local timestamps and we are about to stop anyway.
                    if result_is_complete_chunk:
                        running_offset += chunk_duration
                        completed_chunks = ci + 1
                        self._write_checkpoint(source, completed_chunks, len(chunks), fi, total_files)

                    if self._stop.is_set():
                        stopped = True
                        if self.mode == 'Google Gemini' and r:
                            # The API call already returned a full chunk, so include it in the
                            # resumable checkpoint as completed.
                            if completed_chunks < ci + 1:
                                completed_chunks = ci + 1
                                running_offset += chunk_duration
                                self._write_checkpoint(source, completed_chunks, len(chunks), fi, total_files)
                        break

                if stopped or self._stop.is_set():
                    self._set_stage('正在整理已完成內容並輸出中止版檔案…', -1)
                    paths = self._export_current(srcp, all_text, all_segments, partial=True)
                    if paths:
                        self.log.emit('立即停止完成：已輸出可開啟的中止版檔案。')
                    else:
                        self.log.emit('立即停止完成：目前尚無已辨識文字，因此沒有建立空白逐字稿。')
                    break

                final_text = '\n'.join(x for x in all_text if x)
                if self.mode == '混合模式' and gemini:
                    self._set_stage('Gemini 整理逐字稿…', 0)
                    final_text = gemini.postprocess_text(
                        final_text,
                        target='繁體中文（台灣）',
                        progress_cb=lambda p, d='': (
                            self.stage_progress.emit(p),
                            self.stage_text.emit(d),
                        ),
                    )
                    all_text = [final_text]

                self._set_stage('匯出 Word / 字幕檔…', -1)
                self._export_current(srcp, all_text, all_segments, partial=False)
                self.stage_progress.emit(100)
                self.progress.emit(95 + int(((fi + 1) / total_files) * 5))

            if stopped or self._stop.is_set():
                self._set_stage('工作已停止；目前結果已整理並輸出。', 100)
                self.cancelled.emit()
            else:
                self.progress.emit(100)
                self._set_stage('全部工作完成', 100)
                self.finished_ok.emit()
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}')

    @staticmethod
    def _gemini_language_codes(language: str) -> list[str]:
        return {'en': ['en-US'], 'ja': ['ja-JP']}.get(language, [])

    @staticmethod
    def _save_cached_result(path, result):
        payload = {
            'text': result.text,
            'language': result.language,
            'engine': result.engine,
            'segments': [
                {'start': s.start, 'end': s.end, 'text': s.text, 'speaker': s.speaker}
                for s in result.segments
            ],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def _load_cached_result(path):
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
        segs = [Segment(**x) for x in payload.get('segments', [])]
        return TranscriptResult(payload.get('text', ''), segs, payload.get('language'), payload.get('engine', ''))

    def _write_checkpoint(self, source, completed, total, file_index, total_files):
        p = self.output_dir / '.junba_checkpoint.json'
        data = {
            'source': source,
            'completed_chunks': completed,
            'total_chunks': total,
            'file_index': file_index + 1,
            'total_files': total_files,
            'mode': self.mode,
            'model': self.model_name,
            'acceleration': self.acceleration,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        self.checkpoint.emit(f'進度已儲存：{completed}/{total} 區段｜{p}')
