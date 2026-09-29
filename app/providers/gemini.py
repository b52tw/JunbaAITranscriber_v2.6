from __future__ import annotations
import mimetypes
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from app.core.models import Segment, TranscriptResult


def _seconds(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip()
    if s.endswith('s'):
        s = s[:-1]
    try:
        return float(s)
    except Exception:
        return 0.0


@contextmanager
def ascii_upload_alias(path: str):
    """Create an ASCII-basename alias for SDK multipart upload.

    Some Windows HTTP multipart stacks attempt to encode the uploaded filename as
    ASCII. A Chinese/Japanese filename can therefore fail before the request is
    sent. Keep the user's real file unchanged and upload through a temporary
    ASCII-only basename instead.
    """
    src = Path(path)
    suffix = src.suffix.lower()
    if not suffix or any(ord(ch) > 127 for ch in suffix):
        suffix = '.m4a'
    alias_name = f'junba_audio_{uuid.uuid4().hex}{suffix}'
    alias = src.with_name(alias_name)
    created = False
    try:
        try:
            os.link(src, alias)
            created = True
        except Exception:
            try:
                shutil.copy2(src, alias)
                created = True
            except Exception:
                # Last resort: system temp. The basename remains ASCII.
                td = tempfile.mkdtemp(prefix='junba_gemini_')
                alias = Path(td) / alias_name
                shutil.copy2(src, alias)
                created = True
        yield str(alias)
    finally:
        if created:
            try:
                alias.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            parent = alias.parent
            if parent.name.startswith('junba_gemini_'):
                shutil.rmtree(parent, ignore_errors=True)
        except Exception:
            pass


class GeminiProvider:
    TRANSCRIBE_MODEL = 'gemini-3.5-transcribe'
    POSTPROCESS_MODEL = 'gemini-3.8-flash'

    def __init__(self, api_key: str, model: str | None = None):
        if not api_key:
            raise ValueError('尚未設定 Gemini API Key')
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model or self.TRANSCRIBE_MODEL

    def transcribe(self, path: str, diarization=True, timestamps=True, smart=False,
                   language_codes=None, progress_cb=None) -> TranscriptResult:
        original_mime = mimetypes.guess_type(path)[0] or 'audio/mp4'
        uploaded = None
        try:
            if progress_cb:
                progress_cb(5, '建立安全上傳檔名')
            with ascii_upload_alias(path) as upload_path:
                if progress_cb:
                    progress_cb(10, '上傳音檔至 Google')
                uploaded = self.client.files.upload(file=upload_path)
            if progress_cb:
                progress_cb(35, 'Google 已收到音檔，開始轉錄')
            if smart:
                mode = 'smart'
            else:
                mode = {'type': 'verbatim'}
                if diarization:
                    mode['diarization_mode'] = 'speaker'
                if timestamps:
                    mode['timestamp_granularities'] = ['word']
            cfg = {'transcription_config': {'language_codes': language_codes or [], 'mode': mode}}
            interaction = self.client.interactions.create(
                model=self.model,
                input=[{
                    'type': 'audio',
                    'uri': uploaded.uri,
                    'mime_type': uploaded.mime_type or original_mime,
                }],
                generation_config=cfg,
            )
            if progress_cb:
                progress_cb(90, '解析 Gemini 回傳內容')
            text = getattr(interaction, 'output_text', '') or ''
            words = []
            for step in getattr(interaction, 'steps', []) or []:
                for content in getattr(step, 'content', []) or []:
                    for ann in getattr(content, 'annotations', []) or []:
                        if getattr(ann, 'type', None) == 'word_info':
                            words.append(ann)
            segs = self._group_words(words)
            if not segs and text:
                segs = [Segment(0.0, 0.0, text)]
            if progress_cb:
                progress_cb(100, 'Gemini 轉錄完成')
            return TranscriptResult(text=text, segments=segs, language=None, engine=self.model)
        except UnicodeEncodeError as e:
            raise RuntimeError(
                'Google 音檔上傳遇到檔名字元編碼錯誤。v2.6 已使用 ASCII 暫存別名；'
                f'若仍出現此訊息，請回報完整錯誤：{e}'
            ) from e
        finally:
            if uploaded is not None:
                try:
                    name = getattr(uploaded, 'name', None)
                    if name:
                        self.client.files.delete(name=name)
                except Exception:
                    pass

    @staticmethod
    def _group_words(words) -> list[Segment]:
        result: list[Segment] = []
        current = None
        for w in words:
            speaker = getattr(w, 'speaker', None) or 'spk_1'
            start = _seconds(getattr(w, 'start_offset', None))
            end = _seconds(getattr(w, 'end_offset', None))
            txt = str(getattr(w, 'text', '') or '').strip()
            if not txt:
                continue
            if current and current.speaker == speaker and start - current.end <= 1.2:
                current.end = end
                if txt[:1] in '，。！？,.!?;；:：':
                    current.text += txt
                else:
                    current.text += (' ' if current.text and current.text[-1:].isascii() and txt[:1].isascii() else '') + txt
            else:
                current = Segment(start, end, txt, speaker)
                result.append(current)
        return result

    def postprocess_text(self, text: str, target='繁體中文', progress_cb=None) -> str:
        if progress_cb:
            progress_cb(10, '送出文字整理要求')
        prompt = (
            '請整理以下語音逐字稿。保留原意，不自行補造事實；修正明顯標點與斷句，'
            f'輸出使用{target}。若有講者標記請保留。\n\n{text}'
        )
        r = self.client.models.generate_content(model=self.POSTPROCESS_MODEL, contents=prompt)
        if progress_cb:
            progress_cb(100, 'Gemini 文字整理完成')
        return getattr(r, 'text', '') or text
