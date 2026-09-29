import os
import sys
from pathlib import Path


def self_test(argv):
    from app.core.diagnostics import environment_report
    out = argv[0] if argv else str(Path.cwd() / 'JunbaAITranscriber_SELFTEST.txt')
    ok, report = environment_report(str(Path(out).parent), '')
    Path(out).write_text(('SELFTEST_OK\n' if ok else 'SELFTEST_FAILED\n') + report, encoding='utf-8')
    return 0 if ok else 2


def ui_self_test(argv):
    # Does not open a visible window. Used by Windows GitHub Actions after packaging.
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    out = argv[0] if argv else str(Path.cwd() / 'JunbaAITranscriber_UI_SELFTEST.txt')
    try:
        from PySide6.QtWidgets import QApplication
        from app.ui.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        win = MainWindow()
        checks = []
        checks.append(('version', 'v2.6' in win.windowTitle()))
        checks.append(('traditional-label', '繁體中文' in win.language.itemText(1)))
        checks.append(('traditional-default', win.traditional.isChecked()))
        checks.append(('stop-exports-partial-label', '停止並輸出目前結果' in win.stop_btn.text()))
        checks.append(('hardware-auto-option', win.accel.findData('auto') >= 0))
        checks.append(('hardware-npu-option', win.accel.findData('openvino_npu') >= 0))
        checks.append(('hardware-intel-gpu-option', win.accel.findData('openvino_gpu') >= 0))
        checks.append(('hardware-cuda-option', win.accel.findData('cuda') >= 0))
        checks.append(('hardware-cpu-option', win.accel.findData('cpu') >= 0))
        win.mode.setCurrentText('Google Gemini')
        app.processEvents()
        checks.append(('google-diar-enabled', win.diar.isEnabled()))
        checks.append(('google-timestamps-enabled', win.timestamps.isEnabled()))
        win.smart.setChecked(True)
        app.processEvents()
        checks.append(('smart-auto-disables-diar', not win.diar.isChecked()))
        checks.append(('smart-auto-disables-timestamps', not win.timestamps.isChecked()))
        win.smart.setChecked(False)
        win.diar.setChecked(True)
        win.timestamps.setChecked(True)
        app.processEvents()
        checks.append(('diar-and-timestamps-together', win.diar.isChecked() and win.timestamps.isChecked() and not win.smart.isChecked()))
        ok = all(v for _, v in checks)
        report = '\n'.join(f"{'PASS' if v else 'FAIL'} {k}" for k, v in checks)
        Path(out).write_text(('UI_SELFTEST_OK\n' if ok else 'UI_SELFTEST_FAILED\n') + report, encoding='utf-8')
        win.close()
        return 0 if ok else 3
    except Exception as e:
        Path(out).write_text(f'UI_SELFTEST_FAILED\n{type(e).__name__}: {e}\n', encoding='utf-8')
        return 3


def main():
    if '--self-test' in sys.argv:
        i = sys.argv.index('--self-test')
        raise SystemExit(self_test(sys.argv[i+1:i+2]))
    if '--ui-self-test' in sys.argv:
        i = sys.argv.index('--ui-self-test')
        raise SystemExit(ui_self_test(sys.argv[i+1:i+2]))
    from PySide6.QtWidgets import QApplication
    from app.ui.main_window import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName('Junba AI Transcriber')
    app.setOrganizationName('Junba')
    win = MainWindow(); win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
