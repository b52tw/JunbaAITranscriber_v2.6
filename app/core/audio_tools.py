from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import imageio_ffmpeg


def ffmpeg_exe() -> str:
    bundled = Path(__file__).resolve().parents[2] / 'tools' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
    if bundled.exists():
        return str(bundled)
    return imageio_ffmpeg.get_ffmpeg_exe()


def audio_duration_seconds(path: str) -> float:
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return float(c.duration) / 1_000_000.0
            for s in c.streams.audio:
                if s.duration is not None and s.time_base is not None:
                    return float(s.duration * s.time_base)
    except Exception:
        pass
    try:
        import re
        r = subprocess.run([ffmpeg_exe(), '-i', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, encoding='utf-8', errors='replace', timeout=15)
        m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', r.stderr or '')
        if m:
            h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
            return h * 3600 + mi * 60 + sec
    except Exception:
        pass
    return 0.0


def _run_ffmpeg_progress(cmd: list[str], duration: float, progress_cb=None, cancel_cb=None):
    full = [cmd[0], '-hide_banner', '-nostats', '-progress', 'pipe:1'] + cmd[1:]
    p = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding='utf-8', errors='replace')
    try:
        if p.stdout:
            for line in p.stdout:
                if cancel_cb and cancel_cb():
                    p.terminate()
                    raise RuntimeError('工作已取消')
                line = line.strip()
                if line.startswith('out_time_us=') and progress_cb and duration > 0:
                    try:
                        sec = float(line.split('=', 1)[1]) / 1_000_000.0
                        progress_cb(max(0, min(99, int(sec / duration * 100))))
                    except Exception:
                        pass
                elif line == 'progress=end' and progress_cb:
                    progress_cb(100)
        stderr = p.stderr.read() if p.stderr else ''
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(stderr[-3000:] or f'ffmpeg 失敗，代碼 {rc}')
    finally:
        if p.poll() is None:
            p.kill()


def _source_signature(input_path: str, minutes: int) -> dict:
    p = Path(input_path)
    st = p.stat()
    return {
        'source': str(p.resolve()),
        'size': int(st.st_size),
        'mtime_ns': int(st.st_mtime_ns),
        'minutes': int(minutes),
    }


def split_output_dir(output_root: str, input_path: str, minutes: int) -> str:
    """Visible split folder usable by this app and other transcription tools."""
    src = Path(input_path)
    digest = hashlib.sha1(str(src.resolve()).encode('utf-8')).hexdigest()[:8]
    # Keep path manageable on Windows while retaining a recognizable original name.
    short_stem = src.stem[:70] or 'audio'
    return str(Path(output_root) / '切割音檔' / f'{short_stem}_{minutes}min_{digest}')


def split_audio(input_path: str, out_dir: str, minutes: int, progress_cb=None, cancel_cb=None) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(input_path).stem
    for old in Path(out_dir).glob('*_chunk_*.m4a'):
        try:
            old.unlink()
        except Exception:
            pass
    pattern = str(Path(out_dir) / f'{stem}_chunk_%03d.m4a')
    cmd = [
        ffmpeg_exe(), '-y', '-i', input_path,
        '-map', '0:a:0', '-c:a', 'aac', '-b:a', '128k',
        '-f', 'segment', '-segment_time', str(max(1, minutes) * 60),
        '-reset_timestamps', '1', pattern,
    ]
    try:
        _run_ffmpeg_progress(cmd, audio_duration_seconds(input_path), progress_cb, cancel_cb)
    except RuntimeError as e:
        if str(e) == '工作已取消':
            raise
        raise RuntimeError('音檔切割失敗：\n' + str(e))
    chunks = [str(p) for p in sorted(Path(out_dir).glob('*_chunk_*.m4a'))]
    if not chunks:
        raise RuntimeError('音檔切割完成但找不到任何區段檔。')
    return chunks


def ensure_split_audio(input_path: str, out_dir: str, minutes: int, progress_cb=None, cancel_cb=None) -> list[str]:
    """Reuse a previous split when source metadata and split duration are unchanged."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'split_manifest.json'
    sig = _source_signature(input_path, minutes)
    if manifest.exists():
        try:
            old = json.loads(manifest.read_text(encoding='utf-8'))
            chunks = [str(out / x) for x in old.get('chunks', [])]
            if old.get('signature') == sig and chunks and all(Path(x).exists() for x in chunks):
                if progress_cb:
                    progress_cb(100)
                return chunks
        except Exception:
            pass
    chunks = split_audio(input_path, out_dir, minutes, progress_cb, cancel_cb)
    manifest.write_text(json.dumps({
        'signature': sig,
        'chunks': [Path(x).name for x in chunks],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return chunks


def merge_audio(inputs: list[str], output_path: str, progress_cb=None, cancel_cb=None) -> str:
    if not inputs:
        raise ValueError('沒有可合併的音檔')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = sum(audio_duration_seconds(x) for x in inputs)
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / 'concat.txt'
        with lst.open('w', encoding='utf-8') as f:
            for item in inputs:
                escaped = str(Path(item).resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        cmd = [
            ffmpeg_exe(), '-y', '-f', 'concat', '-safe', '0', '-i', str(lst),
            '-vn', '-c:a', 'aac', '-b:a', '128k', output_path,
        ]
        try:
            _run_ffmpeg_progress(cmd, duration, progress_cb, cancel_cb)
        except RuntimeError as e:
            if str(e) == '工作已取消':
                raise
            raise RuntimeError('音檔合併失敗：\n' + str(e))
    return output_path


def copy_or_prepare(input_path: str, work_dir: str) -> str:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    dst = Path(work_dir) / Path(input_path).name
    if Path(input_path).resolve() != dst.resolve():
        shutil.copy2(input_path, dst)
    return str(dst)
