from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import subprocess
from typing import Iterable


@dataclass
class AcceleratorInfo:
    key: str
    label: str
    available: bool
    detail: str = ''
    recommended_rank: int = 999


def _run_powershell(script: str) -> str:
    if os.name != 'nt':
        return ''
    try:
        p = subprocess.run(
            ['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding='utf-8', errors='replace', timeout=6,
        )
        return p.stdout.strip() if p.returncode == 0 else ''
    except Exception:
        return ''


def cpu_identity() -> tuple[str, str]:
    name = platform.processor() or platform.machine() or 'Unknown CPU'
    vendor = ''
    if os.name == 'nt':
        raw = _run_powershell("Get-CimInstance Win32_Processor | Select-Object -First 1 Name,Manufacturer | ConvertTo-Json -Compress")
        if raw:
            try:
                import json
                d = json.loads(raw)
                name = (d.get('Name') or name).strip()
                vendor = (d.get('Manufacturer') or '').strip()
            except Exception:
                pass
    low = f'{vendor} {name}'.lower()
    if 'amd' in low:
        vendor = 'AMD'
    elif 'intel' in low:
        vendor = 'Intel'
    return vendor or 'x86-64', name


def _cuda_count() -> int:
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def openvino_devices() -> dict[str, str]:
    """Return {device_id: full device name}; empty dict if OpenVINO is unavailable."""
    try:
        import openvino as ov
        core = ov.Core()
        result: dict[str, str] = {}
        for dev in core.available_devices:
            full = dev
            try:
                full = str(core.get_property(dev, 'FULL_DEVICE_NAME'))
            except Exception:
                pass
            result[str(dev)] = full
        return result
    except Exception:
        return {}


def accelerator_options() -> list[AcceleratorInfo]:
    vendor, cpu = cpu_identity()
    ov_devices = openvino_devices()
    cuda = _cuda_count()

    npu_keys = [k for k in ov_devices if k.upper().startswith('NPU')]
    gpu_keys = [k for k in ov_devices if k.upper().startswith('GPU')]
    npu_detail = '; '.join(f'{k}: {ov_devices[k]}' for k in npu_keys)
    gpu_detail = '; '.join(f'{k}: {ov_devices[k]}' for k in gpu_keys)

    return [
        AcceleratorInfo('auto', '自動（效能優先）', True,
                        '優先：NVIDIA CUDA → Intel GPU/OpenVINO → Intel NPU/OpenVINO → CPU', 0),
        AcceleratorInfo('cuda', 'NVIDIA GPU / CUDA', cuda > 0,
                        f'偵測到 {cuda} 個 CUDA 裝置' if cuda else '未偵測到 NVIDIA CUDA', 1),
        AcceleratorInfo('openvino_gpu', 'Intel GPU / OpenVINO', bool(gpu_keys),
                        gpu_detail or '未偵測到 OpenVINO GPU', 2),
        AcceleratorInfo('openvino_npu', 'Intel NPU / OpenVINO（省電）', bool(npu_keys),
                        npu_detail or '未偵測到 OpenVINO NPU', 3),
        AcceleratorInfo('cpu', f'CPU / faster-whisper（{vendor}）', True,
                        f'{cpu}｜x86-64 CPU 路徑，Intel/AMD 皆可使用', 4),
    ]


def recommended_acceleration() -> str:
    opts = {o.key: o for o in accelerator_options()}
    for key in ('cuda', 'openvino_gpu', 'openvino_npu', 'cpu'):
        if opts.get(key) and opts[key].available:
            return key
    return 'cpu'


def resolve_acceleration(requested: str) -> str:
    if requested == 'auto':
        return recommended_acceleration()
    opts = {o.key: o for o in accelerator_options()}
    info = opts.get(requested)
    if info is None:
        raise ValueError(f'未知硬體加速模式：{requested}')
    if not info.available:
        raise RuntimeError(f'目前電腦無法使用「{info.label}」：{info.detail}')
    return requested


def hardware_summary() -> str:
    vendor, cpu = cpu_identity()
    lines = [f'CPU：{cpu} ({vendor})']
    for o in accelerator_options():
        if o.key == 'auto':
            continue
        lines.append(f"{'✓' if o.available else '○'} {o.label}：{o.detail}")
    lines.append(f'自動模式目前會選擇：{recommended_acceleration()}')
    return '\n'.join(lines)
