from __future__ import annotations
import math
import os
import struct
import tempfile
import wave
from pathlib import Path
from PySide6.QtCore import Qt, Signal, QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QFileDialog, QComboBox, QSpinBox, QCheckBox, QLineEdit, QProgressBar,
    QMessageBox, QGroupBox, QFormLayout, QTabWidget, QAbstractItemView, QDialog,
    QDialogButtonBox, QPlainTextEdit
)
from app.core.settings import settings, load_api_key, save_api_key
from app.core.worker import TranscribeWorker
from app.core.audio_tools import (
    merge_audio, audio_duration_seconds, ensure_split_audio, split_output_dir
)
from app.core.diagnostics import environment_report
from app.core.hardware import accelerator_options, hardware_summary

AUDIO_EXTS = {'.m4a','.mp3','.wav','.aac','.flac','.ogg','.mp4','.webm','.aiff','.opus'}
API_KEY_URL = 'https://aistudio.google.com/app/apikey'


class AudioListWidget(QListWidget):
    filesDropped = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setToolTip('可一次多選，也可從檔案總管拖曳多個音檔到這裡。')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
        else: super().dragMoveEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            paths=[]
            for u in event.mimeData().urls():
                p=u.toLocalFile()
                if p and Path(p).suffix.lower() in AUDIO_EXTS: paths.append(p)
            if paths:
                self.filesDropped.emit(paths); event.acceptProposedAction(); return
        super().dropEvent(event)


class ApiKeyTestWorker(QThread):
    done = Signal(bool, str)
    def __init__(self, key: str):
        super().__init__(); self.key = key

    def run(self):
        uploaded = None
        try:
            from google import genai
            client = genai.Client(api_key=self.key)
            # Real end-to-end smoke test: upload a tiny ASCII-named WAV and invoke
            # gemini-3.5-transcribe. This catches failures that a /models check misses.
            with tempfile.TemporaryDirectory(prefix='junba_api_test_') as td:
                wav_path = Path(td) / 'junba_api_test.wav'
                rate = 16000
                frames = bytearray()
                # Quiet 440 Hz tone; no personal/user audio is used for the test.
                for i in range(rate // 2):
                    sample = int(600 * math.sin(2 * math.pi * 440 * i / rate))
                    frames.extend(struct.pack('<h', sample))
                with wave.open(str(wav_path), 'wb') as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate); wf.writeframes(bytes(frames))
                uploaded = client.files.upload(file=str(wav_path))
                client.interactions.create(
                    model='gemini-3.5-transcribe',
                    input=[{'type':'audio','uri':uploaded.uri,'mime_type':uploaded.mime_type or 'audio/wav'}],
                )
            self.done.emit(True, 'API Key、音訊上傳與 gemini-3.5-transcribe 測試成功。')
        except Exception as e:
            self.done.emit(False, f'Gemini 完整測試失敗：{type(e).__name__}: {e}')
        finally:
            try:
                if uploaded is not None and getattr(uploaded, 'name', None):
                    client.files.delete(name=uploaded.name)
            except Exception:
                pass


class DiagnosticsWorker(QThread):
    done = Signal(bool, str)
    def __init__(self, output_dir: str, model_dir: str):
        super().__init__(); self.output_dir=output_dir; self.model_dir=model_dir
    def run(self):
        self.done.emit(*environment_report(self.output_dir, self.model_dir))


class SplitOnlyWorker(QThread):
    progress = Signal(int)
    stage = Signal(str)
    log = Signal(str)
    done = Signal(str)
    failed = Signal(str)

    def __init__(self, files: list[str], output_dir: str, minutes: int):
        super().__init__(); self.files=files; self.output_dir=output_dir; self.minutes=minutes

    def run(self):
        try:
            if self.minutes <= 0:
                raise ValueError('「只切割音檔」需要先設定大於 0 的切割分鐘數。')
            roots=[]; n=max(1,len(self.files))
            for i, src in enumerate(self.files):
                dur=audio_duration_seconds(src)
                if dur > 0 and dur <= self.minutes*60:
                    self.log.emit(f'{Path(src).name} 未超過 {self.minutes} 分鐘，不需切割。')
                    self.progress.emit(int((i+1)/n*100)); continue
                out=split_output_dir(self.output_dir, src, self.minutes)
                self.stage.emit(f'切割 {Path(src).name}')
                chunks=ensure_split_audio(
                    src, out, self.minutes,
                    progress_cb=lambda p,_i=i: self.progress.emit(int(((_i+p/100)/n)*100)),
                )
                roots.append(out)
                self.log.emit(f'切割完成：{len(chunks)} 段｜{out}')
            self.progress.emit(100)
            self.done.emit('\n'.join(roots) if roots else str(Path(self.output_dir)/'切割音檔'))
        except Exception as e:
            self.failed.emit(f'{type(e).__name__}: {e}')


class SplitChoiceDialog(QDialog):
    def __init__(self, count: int, current: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle('確認音檔切割')
        root=QVBoxLayout(self)
        root.addWidget(QLabel(f'已加入 {count} 個音檔。辨識前是否先切割？'))
        self.choice=QComboBox(); self.choice.addItem('不切割',0)
        for m in (2,5,10,15,30,60): self.choice.addItem(f'每 {m} 分鐘切一段',m)
        self.choice.addItem('自訂分鐘數',-1)
        self.custom=QSpinBox(); self.custom.setRange(1,180); self.custom.setValue(current if current>0 else 10); self.custom.setSuffix(' 分鐘')
        row=QHBoxLayout(); row.addWidget(self.choice,1); row.addWidget(self.custom); root.addLayout(row)
        note=QLabel('v2.6 會「先完成切割，再開始載入 Whisper 或上傳 Gemini」。切割檔會保存在輸出位置\\切割音檔，可供其他辨識軟體使用。')
        note.setWordWrap(True); root.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); root.addWidget(buttons)
        idx=self.choice.findData(current)
        if idx>=0:self.choice.setCurrentIndex(idx)
        self.choice.currentIndexChanged.connect(self._sync); self._sync()
    def _sync(self): self.custom.setEnabled(self.choice.currentData()==-1)
    def minutes(self): return self.custom.value() if self.choice.currentData()==-1 else int(self.choice.currentData())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Junba AI Transcriber v2.6')
        self.resize(1120,860)
        self.worker=None; self.split_worker=None; self.api_test_worker=None; self.diag_worker=None
        self.qs=settings()
        tabs=QTabWidget(); tabs.addTab(self._build_workspace(),'工作區'); tabs.addTab(self._build_settings(),'設定')
        self.setCentralWidget(tabs); self.statusBar().showMessage('就緒'); self._update_mode_ui()

    def _build_workspace(self):
        w=QWidget(); root=QVBoxLayout(w)
        title=QLabel('Junba AI Transcriber v2.6｜離線 Whisper × Intel NPU/GPU × Google Gemini')
        title.setStyleSheet('font-size:20px;font-weight:700;padding:6px;'); root.addWidget(title)
        hint=QLabel('可一次多選或拖曳音檔。設定切割後，程式會先把音檔完整切好，再開始辨識。')
        hint.setStyleSheet('color:#b8c7d9;'); root.addWidget(hint)
        self.files=AudioListWidget(); self.files.filesDropped.connect(self._add_paths); root.addWidget(self.files,1)
        row=QHBoxLayout()
        for text,fn in [('加入音檔',self.add_files),('移除選取',self.remove_selected),('清空',self.files.clear),('合併音檔',self.merge_selected)]:
            b=QPushButton(text); b.clicked.connect(fn); row.addWidget(b)
        root.addLayout(row)

        box=QGroupBox('工作流程'); form=QFormLayout(box)
        self.mode=QComboBox(); self.mode.addItems(['離線 Whisper','Google Gemini','混合模式']); self.mode.currentTextChanged.connect(self._update_mode_ui)
        self.model=QComboBox(); self.model.addItems(['large-v3','medium','small','base'])
        self.local_model=QLineEdit(self.qs.value('local_model_dir',''))
        modelrow=QHBoxLayout(); modelrow.addWidget(self.local_model,1); mb=QPushButton('本機模型…'); mb.clicked.connect(self.choose_model_dir); modelrow.addWidget(mb)
        self.accel=QComboBox()
        self.accel.setToolTip('自動模式會選最快可用路徑；也可明確指定 Intel NPU、Intel GPU、NVIDIA CUDA 或 CPU。')
        self.hw_refresh=QPushButton('重新偵測硬體'); self.hw_refresh.clicked.connect(self.refresh_hardware)
        hwrow=QHBoxLayout(); hwrow.addWidget(self.accel,1); hwrow.addWidget(self.hw_refresh)
        self.hw_status=QLabel('正在偵測硬體…'); self.hw_status.setWordWrap(True); self.hw_status.setStyleSheet('color:#b8c7d9;')
        self.hw_note=QLabel('Intel NPU/GPU 第一次使用會下載 OpenVINO INT8 Whisper 模型並編譯；完成後模型可留在本機離線重用。16GB 記憶體若正在高占用，large-v3 建議先關閉其他大型程式，或改用 medium/small。')
        self.hw_note.setWordWrap(True); self.hw_note.setStyleSheet('color:#b8c7d9;')
        self.refresh_hardware(initial=True)
        self.language=QComboBox(); self.language.addItem('自動偵測（輸出仍可轉台灣繁體）','auto'); self.language.addItem('繁體中文／華語・台語混合（台灣）','zh'); self.language.addItem('英文','en'); self.language.addItem('日文','ja')
        self.split=QSpinBox(); self.split.setRange(0,180); self.split.setValue(10); self.split.setSuffix(' 分鐘（0=不切割）')
        splitrow=QHBoxLayout(); splitrow.addWidget(self.split,1)
        self.split_only_btn=QPushButton('只切割音檔'); self.split_only_btn.clicked.connect(self.split_only); splitrow.addWidget(self.split_only_btn)
        open_split=QPushButton('開啟切割資料夾'); open_split.clicked.connect(self.open_split_folder); splitrow.addWidget(open_split)
        self.diar=QCheckBox('多人講者（Gemini 音訊模式）'); self.diar.setChecked(True)
        self.timestamps=QCheckBox('字詞時間戳'); self.timestamps.setChecked(True)
        self.smart=QCheckBox('Gemini 智慧逐字稿（與多人講者／時間戳擇一）')
        self.traditional=QCheckBox('繁體中文（台灣用字）輸出'); self.traditional.setChecked(True)
        self.diar.toggled.connect(lambda checked:self._sync_gemini_features('diar', checked))
        self.timestamps.toggled.connect(lambda checked:self._sync_gemini_features('timestamps', checked))
        self.smart.toggled.connect(lambda checked:self._sync_gemini_features('smart', checked))
        flags=QHBoxLayout(); flags.addWidget(self.diar); flags.addWidget(self.timestamps); flags.addWidget(self.smart); flags.addWidget(self.traditional)
        self.privacy=QLabel(''); self.privacy.setWordWrap(True)
        splitnote=QLabel('切割目的：降低單檔大小、避開服務限制、失敗只重跑單一區段。Gemini 開多人講者或字詞時間戳時每段最多 30 分鐘；建議長錄音先切 10～15 分鐘。')
        splitnote.setWordWrap(True); splitnote.setStyleSheet('color:#b8c7d9;')
        form.addRow('辨識引擎',self.mode); form.addRow('Whisper 模型',self.model); form.addRow('本機模型資料夾',modelrow); form.addRow('硬體加速',hwrow); form.addRow('',self.hw_status); form.addRow('',self.hw_note); form.addRow('語言',self.language); form.addRow('切割',splitrow); form.addRow('',splitnote); form.addRow('功能',flags); form.addRow('',self.privacy)
        root.addWidget(box)

        outrow=QHBoxLayout(); self.output=QLineEdit(str(Path.home()/'Documents'/'JunbaTranscripts')); ob=QPushButton('選擇輸出位置'); ob.clicked.connect(self.choose_output); outrow.addWidget(self.output,1); outrow.addWidget(ob); root.addLayout(outrow)
        fmts=QHBoxLayout(); self.f_docx=QCheckBox('Word'); self.f_docx.setChecked(True); self.f_txt=QCheckBox('TXT'); self.f_txt.setChecked(True); self.f_srt=QCheckBox('SRT'); self.f_srt.setChecked(True); self.f_vtt=QCheckBox('VTT')
        for x in (self.f_docx,self.f_txt,self.f_srt,self.f_vtt): fmts.addWidget(x)
        fmts.addStretch(); root.addLayout(fmts)

        self.stage_label=QLabel('目前階段：待命'); root.addWidget(self.stage_label)
        self.stage_progress=QProgressBar(); self.stage_progress.setRange(0,100); self.stage_progress.setValue(0); self.stage_progress.setFormat('目前階段 %p%')
        self.overall_progress=QProgressBar(); self.overall_progress.setRange(0,100); self.overall_progress.setValue(0); self.overall_progress.setFormat('整體進度 %p%')
        root.addWidget(self.stage_progress); root.addWidget(self.overall_progress)
        self.checkpoint_label=QLabel('進度快取：尚未建立'); self.checkpoint_label.setWordWrap(True); root.addWidget(self.checkpoint_label)
        controls=QHBoxLayout(); self.start_btn=QPushButton('▶ 開始'); self.pause_btn=QPushButton('⏸ 暫停'); self.resume_btn=QPushButton('▶ 繼續'); self.stop_btn=QPushButton('■ 立即停止並輸出目前結果')
        self.start_btn.clicked.connect(self.start); self.pause_btn.clicked.connect(self.pause); self.resume_btn.clicked.connect(self.resume); self.stop_btn.clicked.connect(self.stop)
        for b in (self.start_btn,self.pause_btn,self.resume_btn,self.stop_btn): controls.addWidget(b)
        root.addLayout(controls)
        self.logbox=QPlainTextEdit(); self.logbox.setReadOnly(True); self.logbox.setMaximumBlockCount(800); self.logbox.setMaximumHeight(155); self.logbox.setPlaceholderText('切割、上傳、轉錄及錯誤都會顯示在這裡。')
        root.addWidget(self.logbox)
        return w

    def _build_settings(self):
        w=QWidget(); root=QVBoxLayout(w)
        g=QGroupBox('Google AI Studio / Gemini API'); f=QFormLayout(g)
        self.api_key=QLineEdit(load_api_key()); self.api_key.setEchoMode(QLineEdit.Password)
        show=QCheckBox('顯示 API Key'); show.toggled.connect(lambda on:self.api_key.setEchoMode(QLineEdit.Normal if on else QLineEdit.Password))
        save=QPushButton('儲存 API Key'); save.clicked.connect(self.save_key)
        getkey=QPushButton('前往取得 API Key'); getkey.clicked.connect(lambda:QDesktopServices.openUrl(QUrl(API_KEY_URL)))
        self.testkey=QPushButton('完整測試 API Key（含音訊上傳）'); self.testkey.clicked.connect(self.test_api_key)
        keybuttons=QHBoxLayout(); keybuttons.addWidget(save); keybuttons.addWidget(getkey); keybuttons.addWidget(self.testkey)
        self.api_status=QLabel('尚未測試'); self.api_status.setWordWrap(True)
        f.addRow('Gemini API Key',self.api_key); f.addRow('',show); f.addRow('',keybuttons); f.addRow('連線狀態',self.api_status); root.addWidget(g)
        dg=QGroupBox('程式環境檢查'); dl=QVBoxLayout(dg)
        self.diag_btn=QPushButton('執行環境檢查'); self.diag_btn.clicked.connect(self.run_diagnostics)
        self.diag_text=QPlainTextEdit(); self.diag_text.setReadOnly(True); self.diag_text.setMaximumHeight(240)
        dl.addWidget(self.diag_btn); dl.addWidget(self.diag_text); root.addWidget(dg)
        info=QLabel('v2.6：離線 Whisper 新增 Intel NPU / Intel GPU OpenVINO 加速，以及 NVIDIA CUDA、Intel/AMD CPU 選項。\nGemini 多人講者與字詞時間戳可同時使用；智慧逐字稿與這兩項互斥，程式會自動切換。\n中文／華台混合輸出會自動轉為台灣繁體中文，避免簡體字。\nAPI Key 儲存在 Windows 認證儲存區；Google Gemini 模式會上傳音訊，混合模式只上傳文字，離線 Whisper 不上傳音訊。')
        info.setWordWrap(True); root.addWidget(info); root.addStretch(); return w

    def refresh_hardware(self, initial=False):
        try:
            previous = self.qs.value('acceleration','auto') if initial else (self.accel.currentData() or 'auto')
            self.accel.blockSignals(True)
            self.accel.clear()
            opts = accelerator_options()
            for o in opts:
                label = o.label if o.available else o.label + '（目前不可用）'
                self.accel.addItem(label, o.key)
                idx = self.accel.count()-1
                self.accel.setItemData(idx, bool(o.available), Qt.UserRole+1)
                self.accel.setItemData(idx, o.detail, Qt.ToolTipRole)
            idx = self.accel.findData(previous)
            if idx < 0 or self.accel.itemData(idx, Qt.UserRole+1) is False:
                idx = self.accel.findData('auto')
            self.accel.setCurrentIndex(max(0, idx))
            self.accel.blockSignals(False)
            self.hw_status.setText(hardware_summary().replace('\n','｜'))
            if not initial:
                self.statusBar().showMessage('硬體偵測已更新。', 4000)
        except Exception as e:
            if hasattr(self, 'hw_status'):
                self.hw_status.setText(f'硬體偵測失敗：{type(e).__name__}: {e}')

    def _all_files(self): return [self.files.item(i).data(Qt.UserRole) or self.files.item(i).text() for i in range(self.files.count())]
    def add_files(self):
        paths,_=QFileDialog.getOpenFileNames(self,'選擇音檔','','音訊/影片 (*.m4a *.mp3 *.wav *.aac *.flac *.ogg *.aiff *.opus *.mp4 *.webm);;所有檔案 (*.*)'); self._add_paths(paths)
    def _add_paths(self,paths):
        existing=set(self._all_files()); added=[]
        for p in paths:
            p=str(Path(p))
            if p not in existing and Path(p).suffix.lower() in AUDIO_EXTS:
                dur=audio_duration_seconds(p); label=f'{p}   [{dur/60:.1f} 分]' if dur>0 else p
                item=QListWidgetItem(label); item.setData(Qt.UserRole,p); self.files.addItem(item); existing.add(p); added.append(p)
        if added:
            self._append_log(f'已加入 {len(added)} 個音檔。')
            dlg=SplitChoiceDialog(len(added),self.split.value(),self)
            if dlg.exec()==QDialog.Accepted:
                self.split.setValue(dlg.minutes()); self._append_log('切割設定：'+('不切割' if dlg.minutes()==0 else f'每 {dlg.minutes()} 分鐘；辨識前先切割'))
    def remove_selected(self):
        for item in self.files.selectedItems(): self.files.takeItem(self.files.row(item))
    def merge_selected(self):
        selected=[x.data(Qt.UserRole) or x.text() for x in self.files.selectedItems()]; paths=selected if len(selected)>=2 else self._all_files()
        if len(paths)<2: QMessageBox.information(self,'提示','至少加入兩個音檔才能合併。'); return
        out,_=QFileDialog.getSaveFileName(self,'合併輸出',str(Path(self.output.text())/'合併音檔.m4a'),'M4A (*.m4a)')
        if not out:return
        try:
            self.stage_label.setText('目前階段：合併音檔'); merge_audio(paths,out,progress_cb=self._set_stage_progress); QMessageBox.information(self,'完成',f'已合併：\n{out}')
        except Exception as e: QMessageBox.critical(self,'合併失敗',str(e))

    def split_only(self):
        if self.worker and self.worker.isRunning(): QMessageBox.warning(self,'忙碌中','目前正在辨識，請先完成或停止。'); return
        files=self._all_files()
        if not files: QMessageBox.warning(self,'缺少音檔','請先加入音檔。'); return
        if self.split.value()<=0: QMessageBox.warning(self,'未設定切割','請將切割分鐘數設為 1 以上。'); return
        Path(self.output.text()).mkdir(parents=True,exist_ok=True)
        self.split_only_btn.setEnabled(False); self.stage_label.setText('目前階段：只切割音檔'); self.overall_progress.setValue(0)
        self.split_worker=SplitOnlyWorker(files,self.output.text(),self.split.value())
        self.split_worker.progress.connect(self.overall_progress.setValue); self.split_worker.stage.connect(lambda t:self.stage_label.setText('目前階段：'+t)); self.split_worker.log.connect(self._append_log); self.split_worker.done.connect(self._split_done); self.split_worker.failed.connect(self._split_failed); self.split_worker.finished.connect(lambda:self.split_only_btn.setEnabled(True)); self.split_worker.start()
    def _split_done(self,path):
        self._append_log('音檔切割工作完成。'); QMessageBox.information(self,'切割完成',f'切割檔已保存於：\n{path}\n\n之後按「開始」會直接重用相同切割結果，不會再重切。')
    def _split_failed(self,text): self._append_log('切割錯誤：'+text); QMessageBox.critical(self,'切割失敗',text)
    def open_split_folder(self):
        p=Path(self.output.text())/'切割音檔'; p.mkdir(parents=True,exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def choose_model_dir(self):
        p=QFileDialog.getExistingDirectory(self,'選擇 faster-whisper 本機模型資料夾',self.local_model.text() or str(Path.cwd()/'models'))
        if p:self.local_model.setText(p); self.qs.setValue('local_model_dir',p)
    def choose_output(self):
        p=QFileDialog.getExistingDirectory(self,'選擇輸出資料夾',self.output.text())
        if p:self.output.setText(p)
    def save_key(self):
        try: save_api_key(self.api_key.text()); self.api_status.setText('API Key 已儲存到 Windows 認證儲存區。')
        except Exception as e: QMessageBox.critical(self,'儲存失敗',str(e))
    def test_api_key(self):
        key=self.api_key.text().strip()
        if not key: QMessageBox.warning(self,'缺少 API Key','請先貼上 Gemini API Key。'); return
        self.testkey.setEnabled(False); self.api_status.setText('完整測試中：建立測試音訊 → 上傳 → Gemini 3.5 Transcribe…')
        self.api_test_worker=ApiKeyTestWorker(key); self.api_test_worker.done.connect(self._api_test_done); self.api_test_worker.start()
    def _api_test_done(self,ok,text): self.testkey.setEnabled(True); self.api_status.setText(('✓ ' if ok else '✗ ')+text)
    def run_diagnostics(self):
        self.diag_btn.setEnabled(False); self.diag_text.setPlainText('檢查中…'); self.diag_worker=DiagnosticsWorker(self.output.text(),self.local_model.text().strip()); self.diag_worker.done.connect(self._diag_done); self.diag_worker.start()
    def _diag_done(self,ok,text): self.diag_btn.setEnabled(True); self.diag_text.setPlainText(text+('\n\n整體：可開始測試。' if ok else '\n\n整體：有必要元件缺失，請先修正紅叉項目。'))
    def _update_mode_ui(self):
        if not hasattr(self, 'mode'):
            return
        mode = self.mode.currentText()
        is_google = mode == 'Google Gemini'
        self.smart.setEnabled(is_google)
        self.diar.setEnabled(is_google and not self.smart.isChecked())
        self.timestamps.setEnabled(is_google and not self.smart.isChecked())
        self.traditional.setEnabled(True)
        self.model.setEnabled(mode != 'Google Gemini')
        self.local_model.setEnabled(mode != 'Google Gemini')
        self.accel.setEnabled(mode != 'Google Gemini')
        self.hw_refresh.setEnabled(mode != 'Google Gemini')
        if mode == '離線 Whisper':
            self.privacy.setText('🔒 完全離線：音訊不會上傳。繁體中文輸出可套用台灣用字轉換。')
        elif mode == '混合模式':
            self.privacy.setText('🔀 Whisper 本機辨識；只把文字送至 Gemini 整理，音訊不會上傳。最終可轉為台灣繁體中文。')
        else:
            self.privacy.setText('☁ 線上模式：音訊會上傳至 Google Gemini。多人講者與時間戳可同時使用；Smart 智慧逐字稿需單獨使用。')

    def _sync_gemini_features(self, source: str, checked: bool):
        """Keep Gemini feature combinations valid without blocking the user with a dialog."""
        if not checked:
            self._update_mode_ui()
            return
        if source == 'smart':
            # Smart transcription cannot be combined with diarization/timestamps.
            for w in (self.diar, self.timestamps):
                w.blockSignals(True); w.setChecked(False); w.blockSignals(False)
            self.statusBar().showMessage('Gemini 智慧逐字稿已啟用；多人講者與字詞時間戳已自動關閉。', 6000)
        elif source in ('diar', 'timestamps') and self.smart.isChecked():
            self.smart.blockSignals(True); self.smart.setChecked(False); self.smart.blockSignals(False)
            self.statusBar().showMessage('多人講者／時間戳需 verbatim 模式；已自動關閉智慧逐字稿。', 6000)
        self._update_mode_ui()

    def start(self):
        if self.split_worker and self.split_worker.isRunning(): QMessageBox.warning(self,'切割中','請先等待「只切割音檔」完成。'); return
        files=self._all_files()
        if not files: QMessageBox.warning(self,'缺少音檔','請先加入音檔。'); return
        mode=self.mode.currentText(); key=self.api_key.text().strip()
        if mode in ('Google Gemini','混合模式') and not key: QMessageBox.warning(self,'缺少 API Key','請到「設定」輸入 Gemini API Key。'); return
        if mode=='Google Gemini' and self.smart.isChecked() and (self.diar.isChecked() or self.timestamps.isChecked()):
            self.smart.setChecked(False)
            self._append_log('已自動關閉 Gemini 智慧逐字稿：多人講者／字詞時間戳需使用 verbatim 模式。')
        formats=[]
        if self.f_docx.isChecked():formats.append('docx')
        if self.f_txt.isChecked():formats.append('txt')
        if self.f_srt.isChecked():formats.append('srt')
        if self.f_vtt.isChecked():formats.append('vtt')
        if not formats: QMessageBox.warning(self,'缺少輸出格式','請至少選一種輸出格式。'); return
        try:Path(self.output.text()).mkdir(parents=True,exist_ok=True)
        except Exception as e: QMessageBox.critical(self,'輸出位置無法使用',str(e)); return
        if self.local_model.text().strip() and not Path(self.local_model.text().strip()).is_dir(): QMessageBox.warning(self,'本機模型路徑錯誤','指定的本機模型資料夾不存在。'); return
        self.qs.setValue('local_model_dir',self.local_model.text().strip())
        accel_key=self.accel.currentData() or 'auto'
        accel_available=self.accel.currentData(Qt.UserRole+1)
        if mode != 'Google Gemini' and accel_available is False:
            QMessageBox.warning(self,'硬體不可用',f'目前電腦未偵測到「{self.accel.currentText()}」。請改用自動或其他可用裝置。')
            return
        self.qs.setValue('acceleration',accel_key)
        self.worker=TranscribeWorker(files,self.output.text(),mode,self.split.value(),self.model.currentText(),self.local_model.text().strip(),self.language.currentData(),key,self.diar.isChecked(),self.timestamps.isChecked(),self.smart.isChecked(),self.traditional.isChecked(),formats,acceleration=accel_key)
        self.worker.status.connect(self.statusBar().showMessage); self.worker.progress.connect(self.overall_progress.setValue); self.worker.stage_progress.connect(self._set_stage_progress); self.worker.stage_text.connect(lambda t:self.stage_label.setText('目前階段：'+t)); self.worker.log.connect(self._append_log); self.worker.checkpoint.connect(self.checkpoint_label.setText); self.worker.file_done.connect(self._file_done); self.worker.failed.connect(self._failed); self.worker.finished_ok.connect(self._ok); self.worker.cancelled.connect(self._cancelled); self.worker.finished.connect(self._thread_finished)
        self._set_running(True); self.overall_progress.setValue(0); self.stage_progress.setValue(0); self._append_log(f'v2.6：開始工作；硬體={self.accel.currentText()}；若設定切割，會先完成切割再進入辨識。'); self.worker.start()
    def pause(self):
        if self.worker and self.worker.isRunning():self.worker.pause(); self._append_log('已要求暫停；會在下一個安全點停住。')
    def resume(self):
        if self.worker and self.worker.isRunning():self.worker.resume(); self._append_log('繼續處理。')
    def stop(self):
        if self.worker and self.worker.isRunning():self.worker.stop(); self._append_log('已要求立即停止；不再開始下一段，正在整理已取得內容並輸出中止版檔案。')
    def _set_stage_progress(self,pct):
        if pct<0:self.stage_progress.setRange(0,0); self.stage_progress.setFormat('處理中…')
        else:
            if self.stage_progress.minimum()==0 and self.stage_progress.maximum()==0:self.stage_progress.setRange(0,100)
            self.stage_progress.setValue(max(0,min(100,pct))); self.stage_progress.setFormat('目前階段 %p%')
    def _append_log(self,text):self.logbox.appendPlainText(text)
    def _set_running(self,running):self.start_btn.setEnabled(not running); self.pause_btn.setEnabled(running); self.resume_btn.setEnabled(running); self.stop_btn.setEnabled(running); self.split_only_btn.setEnabled(not running)
    def _file_done(self,text):self._append_log('已輸出： '+text.replace('\n',' | ')); self.statusBar().showMessage('檔案輸出完成')
    def _failed(self,text):self._append_log('錯誤：'+text); QMessageBox.critical(self,'處理失敗',text)
    def _ok(self):self._append_log('所有工作已完成。'); QMessageBox.information(self,'完成','所有工作已完成。')
    def _cancelled(self):self._append_log('工作已停止。若已有辨識內容，已輸出「中止版」可開啟檔案；完整完成的區段也保留於快取，可再次開始續跑。')
    def _thread_finished(self):
        self._set_running(False)
        if self.worker:self.worker.deleteLater(); self.worker=None
