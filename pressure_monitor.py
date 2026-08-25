"""
1x16 압력센서 실시간 모니터 (STM32F103 + AD7175 보드용)

레이아웃:
- 왼쪽: 포트 설정, Start/Stop 버튼, CSV 저장, 상태 표시
- 오른쪽: 16채널 값을 jet 컬러맵 컨투어(등고선 채움)로 시각화

사용법:
1. 왼쪽 포트 입력란에 COM 포트 번호 입력 (예: COM3)
2. Start 버튼 클릭 -> STREAM_ON 명령 전송, 실시간 데이터 수신 시작
3. Stop 버튼 클릭 -> STREAM_OFF 명령 전송, 정지
4. 저장 버튼 -> Start~Stop 구간 전체를 CSV로 저장 (elapsed_ms는 0부터 10ms 간격)

컨투어 관련 참고:
- 센서가 물리적으로 1행 16열이라, 16개 값을 가로축으로 부드럽게 보간해서
  색이 연속적으로 이어지는 "띠" 형태로 그린다 (세로 방향은 시각적 두께일 뿐,
  실제로 세로 방향 센서 데이터가 있는 게 아님).
- 색상 스케일은 지금까지 관측된 최소~최대 값 기준으로 자동 조정된다.
- 컨투어는 데이터 수신 주기와 동일하게 10ms마다 다시 그린다.
  PC 사양에 따라 렌더링 부하로 UI가 버벅일 수 있다.

필요 라이브러리: pip install pyserial numpy matplotlib

주의: 테라텀 등 다른 프로그램이 같은 COM 포트를 열고 있으면 연결에 실패한다.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import serial
import threading
import queue
import time
import csv
import datetime
import os
import glob as _glob
import json
import hashlib
import subprocess
import sys
import importlib
import traceback
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

torch = None
nn = None
_TORCH_OK = None
_TORCH_ERROR = ""
_TORCH_DISABLED = os.environ.get("PRESSURE_UI_DISABLE_TORCH", "").strip() == "1"
_TORCH_DLL_DIRS = []


def _runtime_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _write_torch_error_log(detail):
    try:
        path = os.path.join(_runtime_app_dir(), "pytorch_error.log")
        with open(path, "w", encoding="utf-8") as f:
            f.write("PyTorch load failed\n")
            f.write(f"executable: {sys.executable}\n")
            f.write(f"frozen: {getattr(sys, 'frozen', False)}\n")
            f.write(f"_MEIPASS: {getattr(sys, '_MEIPASS', '')}\n")
            f.write("dll_dirs:\n")
            for d in _TORCH_DLL_DIRS:
                f.write(f"  {d}\n")
            f.write("\nerror:\n")
            f.write(detail)
            f.write("\n")
    except OSError:
        pass


def _prepare_torch_dll_dirs():
    if not hasattr(os, "add_dll_directory"):
        return
    bases = [_runtime_app_dir()]
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        bases.append(meipass)

    candidates = []
    for base in bases:
        candidates.extend([
            os.path.join(base, "torch", "lib"),
            os.path.join(base, "_internal", "torch", "lib"),
        ])

    for path in candidates:
        if path in _TORCH_DLL_DIRS or not os.path.isdir(path):
            continue
        try:
            os.add_dll_directory(path)
            _TORCH_DLL_DIRS.append(path)
        except OSError:
            pass


def _ensure_torch():
    """PyTorch is large, so load it only when ML features are actually used."""
    global torch, nn, _TORCH_OK, _TORCH_ERROR
    if _TORCH_DISABLED:
        _TORCH_OK = False
        _TORCH_ERROR = "PRESSURE_UI_DISABLE_TORCH=1"
        return False
    if _TORCH_OK and torch is not None and nn is not None:
        return True
    try:
        _prepare_torch_dll_dirs()
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        _TORCH_OK = True
        _TORCH_ERROR = ""
        return True
    except Exception as e:
        torch = None
        nn = None
        _TORCH_OK = False
        _TORCH_ERROR = traceback.format_exc()
        _write_torch_error_log(_TORCH_ERROR)
        return False

APP_DIR = _runtime_app_dir()


def _app_path(*parts):
    return os.path.join(APP_DIR, *parts)


ML_ACTIVE_CH  = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]
ML_SEQ_LEN    = 30
ML_HIDDEN     = 32
ML_MODEL_PATH   = _app_path('anomaly_model.pt')
ML_STATS_PATH   = _app_path('anomaly_stats.npz')
ML_LOG_PATH     = _app_path('ml_anomaly_log.csv')
ML_PREVIEW_REPORT_PATH = _app_path('ml_training_preview.json')
ML_PREVIEW_HTML_PATH = _app_path('ml_training_preview.html')
RULE_THRESH     = 0.5   # 규칙 기반 임계: 정규화 압력 ≥0.5 = 감지
ML_CLIPS_DIR    = _app_path('clips')       # 이상 구간 자동 클립 저장 폴더
SETTINGS_PATH   = _app_path('settings.json')  # 사용자 설정 저장 파일

def _create_lstm_autoencoder():
    if not _ensure_torch():
        return None

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            n = len(ML_ACTIVE_CH)
            self.encoder = nn.LSTM(n, ML_HIDDEN, batch_first=True)
            self.decoder = nn.LSTM(ML_HIDDEN, n, batch_first=True)
        def forward(self, x):
            _, (h, _) = self.encoder(x)
            rep = h[-1].unsqueeze(1).expand(-1, x.size(1), -1)
            out, _ = self.decoder(rep)
            return out

    return LSTMAutoencoder()

NUM_CHANNELS = 16
DEFAULT_PORT = 'COM3'
DEFAULT_BAUD = 115200
SCALE_MIN = 0
SCALE_MAX = 4095  # 히트맵 색상 범위 고정. 음수는 나오지 않게 클리핑한다.

ADMIN_CONFIG_PATH = _app_path('admin_config.json')
RECORDS_DIR = _app_path('records')
DEFAULT_ADMIN_ID = 'admin'
DEFAULT_ADMIN_PW = '1234'
ANOMALY_LOG_PATH = _app_path('anomaly_log.csv')

DARK = dict(
    BG="#0d1117", PANEL="#161b22", CARD="#21262d", BORD="#30363d",
    TEXT="#e6edf3", DIM="#8b949e", GREEN="#238636", GRNH="#2ea043",
    BLUE="#388bfd", ORNG="#d29922", RED="#da3633", FIG="#0d1117",
    HDR="#0a0f16", WARN_BG="#332200", WARN_FG="#ffcc00", CAL_FG_PRESS="#ff6b6b",
)
LIGHT = dict(
    BG="#f3f6fb", PANEL="#ffffff", CARD="#eef3f8", BORD="#c8d2df",
    TEXT="#172033", DIM="#5f6f82", GREEN="#1f8f4d", GRNH="#2fad62",
    BLUE="#1f6feb", ORNG="#b66a00", RED="#d1242f", FIG="#f8fafc",
    HDR="#e8eef6", WARN_BG="#fff2cc", WARN_FG="#8a5700", CAL_FG_PRESS="#d1242f",
)

# ── UI 테마 ──────────────────────────────────────────────────────────
T_BG    = LIGHT["BG"]     # 창 배경
T_PANEL = LIGHT["PANEL"]  # 좌측 패널
T_CARD  = LIGHT["CARD"]   # 카드/입력 배경
T_BORD  = LIGHT["BORD"]   # 구분선
T_TEXT  = LIGHT["TEXT"]   # 기본 텍스트
T_DIM   = LIGHT["DIM"]    # 보조 텍스트
T_GREEN = LIGHT["GREEN"]  # 시작/정상
T_GRNH  = LIGHT["GRNH"]   # 시작 hover
T_BLUE  = LIGHT["BLUE"]   # 강조
T_ORNG  = LIGHT["ORNG"]   # 경고
T_RED   = LIGHT["RED"]    # 정지/오류
T_FIG   = LIGHT["FIG"]    # matplotlib figure 배경
T_HDR   = LIGHT["HDR"]    # 헤더 스트립


def _hash_password(pw):
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()


def load_or_create_admin_config():
    """관리자 아이디/비밀번호(해시)를 별도 설정파일에서 읽어온다.
    파일이 없으면 기본 계정(admin/1234)으로 새로 만든다."""
    if not os.path.exists(ADMIN_CONFIG_PATH):
        config = {
            "admin_id": DEFAULT_ADMIN_ID,
            "admin_pw_hash": _hash_password(DEFAULT_ADMIN_PW),
        }
        with open(ADMIN_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        return config, True  # True = 방금 새로 생성됨
    with open(ADMIN_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f), False


def save_admin_config(config):
    with open(ADMIN_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


class SerialReader(threading.Thread):
    """시리얼 포트를 별도 스레드에서 읽고, 파싱 결과를 큐에 넣는다."""

    def __init__(self, port, baud, out_queue):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.out_queue = out_queue
        self.ser = None
        self.running = False

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            self.out_queue.put(('error', str(e)))
            return

        time.sleep(2)  # 보드 리셋 안정화 대기
        self.running = True
        self.send_command('STREAM_ON')

        while self.running:
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                self.out_queue.put(('error', str(e)))
                break

            if not raw:
                continue

            line = raw.decode('utf-8', errors='ignore').strip()
            if not line.startswith('@'):
                continue

            parts = line.split(',')
            if len(parts) != NUM_CHANNELS + 2:
                continue

            try:
                index = int(parts[0][1:])
                timestamp = int(parts[1])
                adc_values = [int(v) for v in parts[2:]]
            except ValueError:
                continue

            self.out_queue.put(('data', index, timestamp, adc_values))

    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write((cmd + '\r\n').encode('utf-8'))

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            try:
                self.send_command('STREAM_OFF')
                time.sleep(0.1)
            except Exception:
                pass
            self.ser.close()


class App:
    STRIP_ROWS = 40
    CONTOUR_REDRAW_INTERVAL = 0.033

    # ── 채널 이상 감지 기준 ─────────────────────────────────────────────
    _FROZEN_SAMPLES = 30   # 무변화 판정에 쓸 최근 샘플 수
    _FROZEN_STD    = 3.0   # 표준편차 이하 → 무변화(frozen) 판정
    _SAT_LOW       = 50    # ADC 이하 → 단선 의심
    _SAT_HIGH      = 4090  # ADC 이상 → 포화/단선 의심
    _CAL_UNDERFLOW = -500  # 보정Δ 이 이하 → 보정 이탈 의심

    def __init__(self, root):
        self.root = root
        self.root.title("Pressure Monitor  ·  1×16ch")
        self.root.geometry("1120x700")
        self.root.configure(bg=T_BG)

        self.data_queue = queue.Queue()
        self.reader = None
        self._has_live_data = False

        # 색상 범위는 SCALE_MIN~SCALE_MAX 고정. 아래는 참고 표시용 실시간 min/max.
        self.live_min = None
        self.live_max = None
        self.current_values = [SCALE_MAX] * NUM_CHANNELS

        # Start~Stop 구간 동안 쌓이는 기록 (index, elapsed_ms, raw_timestamp, ch0..ch15)
        self.recorded_rows = []
        self.record_start_time = None

        self._last_contour_draw = 0.0
        self.smoothness = 50  # 0=계단식, 100=경계 넓게 부드러움 (칸 중심은 항상 원본값)

        self.cal_offsets = {}
        self.cal_apply = False
        self._ch_history   = [[] for _ in range(NUM_CHANNELS)]
        self._prev_anomaly = [None] * NUM_CHANNELS  # 직전 이상 상태 (변화 시에만 로그)

        self._is_dark = False
        self._warn_bg = LIGHT["WARN_BG"]; self._warn_fg = LIGHT["WARN_FG"]
        self._cal_fg_press = LIGHT["CAL_FG_PRESS"]
        self.cmap_name = "jet"

        # ── 채널 ON/OFF 필터 ─────────────────────────────────────────────
        _DEAD_CH = {1, 5, 14, 15}
        self._ch_enabled = [ch not in _DEAD_CH for ch in range(NUM_CHANNELS)]

        # ── ML 이상 감지 상태 ────────────────────────────────────────────
        self._ml_model       = None
        self._ml_threshold   = None
        self._ml_score       = 0.0
        self._ml_buffer      = deque(maxlen=ML_SEQ_LEN)
        self._ml_was_anomaly     = False   # 이전 프레임 이상 여부 (상태 변화 시에만 로그)
        self._ml_mean_err        = 0.0
        self._ml_std_err         = 0.0
        self._ml_sigma_k         = 3.0    # 임계값 = mean + k*std
        self._ml_score_history   = deque(maxlen=300)  # rolling 점수 이력
        self._ml_anomaly_history = deque(maxlen=300)  # rolling 이상 여부 이력
        self._rule_max_history   = deque(maxlen=300)  # rolling 규칙 기반 최대 압력
        self._rule_thresh        = RULE_THRESH        # 슬라이더로 실시간 조절 가능
        # 알람 상태
        self._alarm_enabled    = False
        self._alarm_mode       = "소리"   # "소리" | "토스트" | "소리+토스트"
        self._alarm_cooldown   = 30.0    # 초
        self._alarm_last_t     = 0.0
        # 클립 자동 저장 상태
        self._clip_pre_buf    = deque(maxlen=30)  # 이상 직전 문맥 프레임
        self._clip_rec_buf    = []                # 현재 이상 구간 누적 버퍼
        self._clip_recording  = False
        self._clip_start_dt   = None
        self._clip_paths      = []               # 클립 목록 (UI용)

        # ── 재생(Playback) 상태 ─────────────────────────────────────────
        self._pb_frames = []      # list of [ch0..ch15]
        self._pb_idx = 0
        self._pb_playing = False
        self._pb_after_id = None
        self._pb_speed = 1.0
        self._pb_seeking = False  # 슬라이더 프로그래밍 세트 시 콜백 억제

        # --- 관리자 계정 / 기록 보관 폴더 준비 ---
        self.admin_config, is_new_config = load_or_create_admin_config()
        self.admin_authenticated = False
        os.makedirs(RECORDS_DIR, exist_ok=True)

        # --- 왼쪽 네비게이션 레일 + 페이지 컨테이너 ---
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass

        shell = tk.Frame(self.root, bg=T_BG)
        shell.pack(fill=tk.BOTH, expand=True)

        self._nav_rail = tk.Frame(shell, bg=T_HDR, width=84)
        self._nav_rail.pack(side=tk.LEFT, fill=tk.Y)
        self._nav_rail.pack_propagate(False)

        self.page_container = tk.Frame(shell, bg=T_BG)
        self.page_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        measuring_tab   = tk.Frame(self.page_container, bg=T_BG)
        admin_tab       = tk.Frame(self.page_container, bg=T_BG)
        calibration_tab = tk.Frame(self.page_container, bg=T_BG)
        ml_score_tab    = tk.Frame(self.page_container, bg=T_BG)

        self._pages = {
            'measure': measuring_tab,
            'admin':   admin_tab,
            'cal':     calibration_tab,
            'ml':      ml_score_tab,
        }
        self._current_page = 'measure'

        self._nav_buttons = {}
        for key, label in [('measure', '측정'), ('admin', '관리자'),
                            ('cal', '보정'), ('ml', 'ML\n점수')]:
            btn = tk.Button(
                self._nav_rail, text=label, font=("맑은 고딕", 9, "bold"),
                bg=T_HDR, fg=T_DIM, activebackground=T_CARD, activeforeground=T_TEXT,
                relief=tk.FLAT, bd=0, pady=14, justify="center",
                command=lambda k=key: self._show_page(k))
            btn.pack(fill=tk.X, pady=(14 if key == 'measure' else 2, 2), padx=8)
            self._nav_buttons[key] = btn

        self.theme_btn = tk.Button(
            self._nav_rail, text="🌙", bg=T_HDR, fg=T_DIM,
            activebackground=T_HDR, activeforeground=T_TEXT,
            relief=tk.FLAT, font=("Consolas", 14), bd=0,
            command=self._toggle_theme)
        self.theme_btn.pack(side=tk.BOTTOM, pady=16)

        self._build_left_panel(measuring_tab)
        self._build_right_panel(measuring_tab)
        self._build_admin_tab(admin_tab)
        self._build_calibration_tab(calibration_tab)
        self._build_ml_tab(ml_score_tab)

        self._show_page('measure')
        self._load_settings()   # 저장된 설정 복원 (_load_ml_model 전에 sigma_k 등 복원)
        self._load_ml_model()
        self.root.after(10, self.poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if is_new_config:
            messagebox.showinfo(
                "관리자 계정 생성됨",
                f"관리자 설정파일({os.path.basename(ADMIN_CONFIG_PATH)})이 없어서 기본 계정으로 새로 만들었습니다.\n\n"
                f"아이디: {DEFAULT_ADMIN_ID}\n비밀번호: {DEFAULT_ADMIN_PW}\n\n"
                "관리자 탭에 로그인한 뒤 비밀번호를 바꾸는 걸 권장합니다."
            )

    # ---------------- 공통: 섹션 헤더 ----------------
    def _sec(self, parent, text, bg=T_PANEL):
        """섹션 헤더: 파란 레이블 + 가로 구분선. 탭 전체에서 공용으로 사용."""
        f = tk.Frame(parent, bg=bg)
        f.pack(fill=tk.X, padx=14, pady=(13, 5))
        tk.Label(f, text=text, bg=bg, fg=T_BLUE,
                 font=("Consolas", 7, "bold")).pack(side=tk.LEFT)
        tk.Frame(f, bg=T_BORD, height=1).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=4)
        return f

    # ---------------- 왼쪽: 제어판 ----------------
    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=T_PANEL, width=272)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        self._build_pinned_card(left)
        self._build_section_tabs(left)

    def _build_pinned_card(self, left):
        """항상 보이는 영역: 헤더 + CONNECTION + DATA (스크롤 없이 접근)."""
        hdr = tk.Frame(left, bg=T_HDR, height=58)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="PRESSURE MONITOR", bg=T_HDR, fg=T_TEXT,
                 font=("Consolas", 10, "bold")).place(relx=0.5, rely=0.36, anchor="center")
        tk.Label(hdr, text="1×16 ch  ·  STM32 + AD7175",
                 bg=T_HDR, fg=T_DIM, font=("Consolas", 7)).place(
            relx=0.5, rely=0.74, anchor="center")

        self._sec(left, "CONNECTION")

        tk.Label(left, text="Serial Port", bg=T_PANEL, fg=T_DIM,
                 font=("Consolas", 7)).pack(anchor="w", padx=14)
        self.port_entry = tk.Entry(
            left, font=("Consolas", 11), justify="center",
            bg=T_CARD, fg=T_TEXT, insertbackground=T_TEXT,
            relief=tk.FLAT, highlightbackground=T_BORD, highlightthickness=1)
        self.port_entry.insert(0, DEFAULT_PORT)
        self.port_entry.pack(padx=14, pady=(2, 8), fill=tk.X, ipady=5)

        btn_row = tk.Frame(left, bg=T_PANEL)
        btn_row.pack(fill=tk.X, padx=14, pady=(0, 6))
        self.start_btn = tk.Button(
            btn_row, text="▶  Start", font=("맑은 고딕", 10, "bold"),
            bg=T_GREEN, fg=T_TEXT, activebackground=T_GRNH, activeforeground=T_TEXT,
            relief=tk.FLAT, pady=7, command=self.start_stream)
        self.start_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.stop_btn = tk.Button(
            btn_row, text="■  Stop", font=("맑은 고딕", 10, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_RED, activeforeground=T_TEXT,
            relief=tk.FLAT, pady=7, state=tk.DISABLED, command=self.stop_stream)
        self.stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._sec(left, "DATA")

        self.save_btn = tk.Button(
            left, text="CSV 저장", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, state=tk.DISABLED, command=self.save_csv)
        self.save_btn.pack(padx=14, pady=(0, 5), fill=tk.X)
        self.record_count_label = tk.Label(
            left, text="기록  0 줄", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 8))
        self.record_count_label.pack(anchor="w", padx=14, pady=(0, 8))

    def _build_section_tabs(self, left):
        """가끔 쓰는 설정들: 알약형 탭으로 전환하며 한 번에 하나씩만 표시."""
        pill_bar = tk.Frame(left, bg=T_PANEL)
        pill_bar.pack(fill=tk.X, padx=12, pady=(4, 0))

        sections = [
            ('display', '표시'), ('status', '상태'), ('channel', '채널'),
            ('alarm', '알람'), ('calpb', '보정'),
        ]
        self._section_buttons = {}
        for key, label in sections:
            btn = tk.Button(
                pill_bar, text=label, font=("맑은 고딕", 8, "bold"),
                bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                relief=tk.FLAT, bd=0, padx=8, pady=5,
                command=lambda k=key: self._show_section(k))
            btn.pack(side=tk.LEFT, padx=(0, 2))
            self._section_buttons[key] = btn

        body = tk.Frame(left, bg=T_PANEL)
        body.pack(fill=tk.BOTH, expand=True)

        self._section_frames = {
            'display': self._build_section_display(body),
            'status':  self._build_section_status(body),
            'channel': self._build_section_channel(body),
            'alarm':   self._build_section_alarm(body),
            'calpb':   self._build_section_calpb(body),
        }
        self._show_section('display')

    def _show_section(self, key):
        for k, f in self._section_frames.items():
            f.pack_forget()
        self._section_frames[key].pack(fill=tk.BOTH, expand=True)
        for k, btn in self._section_buttons.items():
            if k == key:
                btn.config(bg=T_BLUE, fg="#ffffff")
            else:
                btn.config(bg=T_CARD, fg=T_DIM)

    def _build_section_display(self, parent):
        f = tk.Frame(parent, bg=T_PANEL)

        self._sec(f, "DISPLAY")

        icard = tk.Frame(f, bg=T_CARD,
                         highlightbackground=T_BORD, highlightthickness=1)
        icard.pack(padx=14, pady=(0, 6), fill=tk.X)
        ih = tk.Frame(icard, bg=T_CARD)
        ih.pack(fill=tk.X, padx=10, pady=(7, 2))
        tk.Label(ih, text="부드러움", bg=T_CARD, fg=T_DIM,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self.interp_value_label = tk.Label(ih, text=str(self.smoothness),
                                           bg=T_CARD, fg=T_BLUE,
                                           font=("Consolas", 9, "bold"))
        self.interp_value_label.pack(side=tk.RIGHT)
        self.interp_slider = tk.Scale(
            icard, from_=0, to=100, orient=tk.HORIZONTAL,
            bg=T_CARD, fg=T_BLUE, troughcolor=T_BORD,
            highlightthickness=0, showvalue=False, sliderrelief=tk.FLAT,
            command=self._on_interp_change)
        self.interp_slider.set(self.smoothness)
        self.interp_slider.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(icard, text="0: 계단식  ·  100: 부드러움",
                 bg=T_CARD, fg=T_DIM, font=("Consolas", 7)).pack(
            anchor="w", padx=10, pady=(0, 7))

        # 컬러맵 선택 (미리보기 팝업)
        cmap_card = tk.Frame(f, bg=T_CARD,
                             highlightbackground=T_BORD, highlightthickness=1)
        cmap_card.pack(padx=14, pady=(0, 6), fill=tk.X)
        cmap_hdr = tk.Frame(cmap_card, bg=T_CARD)
        cmap_hdr.pack(fill=tk.X, padx=10, pady=(6, 5))
        tk.Label(cmap_hdr, text="컬러맵", bg=T_CARD, fg=T_DIM,
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        self._cmap_btn = tk.Button(
            cmap_hdr, text=f"{self.cmap_name}  ▾",
            bg=T_CARD, fg=T_BLUE, activebackground=T_BORD,
            activeforeground=T_TEXT, relief=tk.FLAT,
            font=("Consolas", 8), bd=0, highlightthickness=0,
            command=self._open_cmap_popup)
        self._cmap_btn.pack(side=tk.RIGHT)

        self.reset_btn = tk.Button(
            f, text="실시간 범위 초기화", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=4, command=self.reset_live_range)
        self.reset_btn.pack(padx=14, pady=(0, 2), fill=tk.X)

        return f

    def _build_section_status(self, parent):
        f = tk.Frame(parent, bg=T_PANEL)

        self._sec(f, "STATUS")

        self.status_label = tk.Label(
            f, text="●  연결 안 됨", bg=T_PANEL, fg=T_DIM,
            font=("맑은 고딕", 9))
        self.status_label.pack(anchor="w", padx=14, pady=(0, 2))
        self.info_label = tk.Label(
            f, text="", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7), justify="left")
        self.info_label.pack(anchor="w", padx=14)
        self.scale_label = tk.Label(
            f, text=f"범위  {SCALE_MIN} – {SCALE_MAX}", bg=T_PANEL,
            fg=T_DIM, font=("Consolas", 7))
        self.scale_label.pack(anchor="w", padx=14, pady=(4, 0))
        self.range_label = tk.Label(
            f, text="실시간  –", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7))
        self.range_label.pack(anchor="w", padx=14, pady=(1, 0))
        tk.Button(
            f, text="이상 로그 열기", font=("Consolas", 7),
            bg=T_PANEL, fg=T_DIM, activebackground=T_CARD,
            relief=tk.FLAT, pady=2, command=self._open_anomaly_log
        ).pack(anchor="w", padx=14, pady=(5, 0))

        # ML 이상 점수 바
        self._ml_label = tk.Label(
            f, text="ML: 모델 없음  (훈련 필요)",
            bg=T_PANEL, fg=T_DIM, font=("Consolas", 7))
        self._ml_label.pack(anchor="w", padx=14, pady=(6, 0))
        self._ml_canvas = tk.Canvas(
            f, bg=T_BORD, height=7, highlightthickness=0)
        self._ml_canvas.pack(fill=tk.X, padx=14, pady=(2, 0))
        self._ml_bar = self._ml_canvas.create_rectangle(
            0, 0, 0, 7, fill=T_GREEN, outline="")
        # 민감도(σ 배수) 슬라이더
        sigma_hdr = tk.Frame(f, bg=T_PANEL)
        sigma_hdr.pack(fill=tk.X, padx=14, pady=(4, 0))
        tk.Label(sigma_hdr, text="민감도 (σ 배수)",
                 bg=T_PANEL, fg=T_DIM, font=("Consolas", 7)).pack(side=tk.LEFT)
        self._ml_sigma_label = tk.Label(
            sigma_hdr, text="3.0 σ", bg=T_PANEL, fg=T_BLUE,
            font=("Consolas", 7, "bold"))
        self._ml_sigma_label.pack(side=tk.RIGHT)
        self._ml_sigma_slider = tk.Scale(
            f, from_=1.0, to=6.0, resolution=0.5, orient=tk.HORIZONTAL,
            bg=T_PANEL, fg=T_BLUE, troughcolor=T_BORD,
            highlightthickness=0, showvalue=False, sliderrelief=tk.FLAT,
            state=tk.DISABLED, command=self._on_sigma_change)
        self._ml_sigma_slider.set(3.0)
        self._ml_sigma_slider.pack(fill=tk.X, padx=14, pady=(0, 2))

        # σ 구간 안내 레이블
        sigma_hint = tk.Frame(f, bg=T_PANEL)
        sigma_hint.pack(fill=tk.X, padx=14, pady=(0, 2))
        tk.Label(sigma_hint, text="1σ (민감)", bg=T_PANEL, fg=T_DIM,
                 font=("Consolas", 7)).pack(side=tk.LEFT)
        tk.Label(sigma_hint, text="6σ (둔감)", bg=T_PANEL, fg=T_DIM,
                 font=("Consolas", 7)).pack(side=tk.RIGHT)

        ml_btn_row = tk.Frame(f, bg=T_PANEL)
        ml_btn_row.pack(fill=tk.X, padx=14, pady=(1, 10))
        tk.Button(
            ml_btn_row, text="ML 모델 훈련", font=("Consolas", 7),
            bg=T_PANEL, fg=T_DIM, activebackground=T_CARD,
            relief=tk.FLAT, pady=1, command=self._train_ml_model
        ).pack(side=tk.LEFT)
        tk.Button(
            ml_btn_row, text="ML 로그 열기", font=("Consolas", 7),
            bg=T_PANEL, fg=T_DIM, activebackground=T_CARD,
            relief=tk.FLAT, pady=1, command=self._open_ml_log
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(
            ml_btn_row, text="모델 정보", font=("Consolas", 7),
            bg=T_PANEL, fg=T_DIM, activebackground=T_CARD,
            relief=tk.FLAT, pady=1, command=self._open_ml_report
        ).pack(side=tk.LEFT, padx=(6, 0))

        return f

    def _build_section_channel(self, parent):
        f = tk.Frame(parent, bg=T_PANEL)

        self._sec(f, "CH FILTER")

        _DEAD = self._DEAD_CH_SET
        self._ch_filter_btns = []
        ch_grid = tk.Frame(f, bg=T_PANEL)
        ch_grid.pack(fill=tk.X, padx=14, pady=(0, 2))
        for i in range(16):
            row_i, col_i = divmod(i, 4)
            is_dead = i in _DEAD
            btn = tk.Button(
                ch_grid, text=f"{i:02d}",
                font=("Consolas", 7, "bold"),
                relief=tk.FLAT, pady=1, padx=0, width=4,
                bg=T_BORD if is_dead else T_GREEN,
                fg=T_DIM if is_dead else "#ffffff",
                activebackground=T_BORD,
                state=tk.DISABLED if is_dead else tk.NORMAL,
                command=lambda ch=i: self._toggle_ch_filter(ch))
            btn.grid(row=row_i, column=col_i, padx=1, pady=1, sticky="ew")
            self._ch_filter_btns.append(btn)

        ch_btn_row = tk.Frame(f, bg=T_PANEL)
        ch_btn_row.pack(fill=tk.X, padx=14, pady=(2, 10))
        tk.Button(ch_btn_row, text="전체 ON", font=("Consolas", 7),
                  bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                  relief=tk.FLAT, pady=1,
                  command=self._ch_filter_all_on).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(ch_btn_row, text="전체 OFF", font=("Consolas", 7),
                  bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                  relief=tk.FLAT, pady=1,
                  command=self._ch_filter_all_off).pack(side=tk.LEFT)

        return f

    def _build_section_alarm(self, parent):
        f = tk.Frame(parent, bg=T_PANEL)

        self._sec(f, "ALARM")

        alarm_top = tk.Frame(f, bg=T_PANEL)
        alarm_top.pack(fill=tk.X, padx=14, pady=(2, 0))

        self._alarm_btn = tk.Button(
            alarm_top, text="알람 OFF", font=("Consolas", 7),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=1, width=9,
            command=self._toggle_alarm)
        self._alarm_btn.pack(side=tk.LEFT)

        self._alarm_mode_var = tk.StringVar(value=self._alarm_mode)
        alarm_menu = tk.OptionMenu(
            alarm_top, self._alarm_mode_var,
            "소리", "토스트", "소리+토스트",
            command=lambda v: setattr(self, '_alarm_mode', v))
        alarm_menu.config(
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            font=("Consolas", 7), relief=tk.FLAT,
            highlightthickness=0, pady=0)
        alarm_menu["menu"].config(bg=T_CARD, fg=T_DIM, font=("Consolas", 7))
        alarm_menu.pack(side=tk.LEFT, padx=(4, 0))

        cooldown_row = tk.Frame(f, bg=T_PANEL)
        cooldown_row.pack(fill=tk.X, padx=14, pady=(2, 2))
        tk.Label(cooldown_row, text="쿨다운:", bg=T_PANEL, fg=T_DIM,
                 font=("Consolas", 7)).pack(side=tk.LEFT)
        self._alarm_cooldown_var = tk.DoubleVar(value=self._alarm_cooldown)
        tk.Scale(
            cooldown_row, variable=self._alarm_cooldown_var,
            from_=5, to=120, resolution=5,
            orient=tk.HORIZONTAL, length=105,
            bg=T_PANEL, fg=T_DIM, troughcolor=T_CARD,
            highlightthickness=0, bd=0, showvalue=False,
            command=self._on_alarm_cooldown_change
        ).pack(side=tk.LEFT, padx=(4, 2))
        self._alarm_cooldown_lbl = tk.Label(
            cooldown_row, text=f"{int(self._alarm_cooldown)}s",
            bg=T_PANEL, fg=T_DIM, font=("Consolas", 7), width=4)
        self._alarm_cooldown_lbl.pack(side=tk.LEFT)

        return f

    def _build_section_calpb(self, parent):
        """CALIBRATION 퀵 컨트롤 + PLAYBACK: 둘 다 상대적으로 덜 자주 쓰는 흐름이라 한 탭에 묶음."""
        f = tk.Frame(parent, bg=T_PANEL)

        self._sec(f, "CALIBRATION")

        tk.Button(
            f, text="보정 파일 불러오기", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, command=self._load_calibration_file
        ).pack(padx=14, pady=(0, 4), fill=tk.X)
        self.cal_loaded_label = tk.Label(
            f, text="파일 없음", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7), wraplength=240, justify="left")
        self.cal_loaded_label.pack(anchor="w", padx=14)
        tk.Label(f, text="보정값 로드 후 적용 ON 필요",
                 bg=T_PANEL, fg=T_DIM, font=("Consolas", 7)
                  ).pack(anchor="w", padx=14, pady=(2, 0))
        self.cal_apply_btn = tk.Button(
            f, text="보정 적용: OFF", font=("맑은 고딕", 7, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=3, state=tk.DISABLED,
            command=self._toggle_cal_apply
        )
        self.cal_apply_btn.pack(padx=14, pady=(4, 0), fill=tk.X)
        tk.Button(
            f, text="오프셋 편집기 ▶", font=("Consolas", 7),
            bg=T_CARD, fg=T_BLUE, activebackground=T_BORD,
            relief=tk.FLAT, pady=1, command=self._open_cal_graphic
        ).pack(anchor="w", padx=14, pady=(4, 0))

        self._sec(f, "PLAYBACK")

        pb_top = tk.Frame(f, bg=T_PANEL)
        pb_top.pack(fill=tk.X, padx=14, pady=(0, 3))
        tk.Button(
            pb_top, text="CSV 불러오기", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=4, command=self._pb_load
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._pb_file_label = tk.Label(
            f, text="파일 없음", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7), wraplength=240, justify="left")
        self._pb_file_label.pack(anchor="w", padx=14)

        self._pb_frame_label = tk.Label(
            f, text="프레임  –", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7))
        self._pb_frame_label.pack(anchor="w", padx=14, pady=(2, 0))

        self._pb_slider = tk.Scale(
            f, from_=0, to=0, orient=tk.HORIZONTAL,
            bg=T_PANEL, fg=T_BLUE, troughcolor=T_BORD,
            highlightthickness=0, showvalue=False, sliderrelief=tk.FLAT,
            state=tk.DISABLED, command=self._pb_seek)
        self._pb_slider.pack(fill=tk.X, padx=14, pady=(1, 4))

        ctrl = tk.Frame(f, bg=T_PANEL)
        ctrl.pack(fill=tk.X, padx=14, pady=(0, 10))
        self._pb_play_btn = tk.Button(
            ctrl, text="▶", font=("맑은 고딕", 9, "bold"),
            bg=T_GREEN, fg=T_TEXT, activebackground=T_GRNH,
            relief=tk.FLAT, pady=4, state=tk.DISABLED,
            command=self._pb_toggle_play)
        self._pb_play_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        self._pb_stop_btn = tk.Button(
            ctrl, text="■", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=4, state=tk.DISABLED,
            command=self._pb_stop)
        self._pb_stop_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self._pb_speed_var = tk.StringVar(value="1×")
        pb_speed_menu = tk.OptionMenu(
            ctrl, self._pb_speed_var,
            "0.5×", "1×", "2×", "5×", "10×",
            command=self._on_pb_speed)
        pb_speed_menu.config(
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            activeforeground=T_TEXT, relief=tk.FLAT,
            font=("Consolas", 7), bd=0, highlightthickness=0, width=3)
        pb_speed_menu["menu"].config(
            bg=T_CARD, fg=T_TEXT,
            activebackground=T_BLUE, activeforeground=T_TEXT)
        pb_speed_menu.pack(side=tk.LEFT)

        return f

    # ---------------- 오른쪽: 원본 | 보정 적용 2분할 컨투어 ----------------
    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=T_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        hdr = tk.Frame(right, bg=T_BG)
        hdr.pack(fill=tk.X, pady=(12, 0))
        tk.Label(hdr, text="RAW", bg=T_BG, fg=T_DIM,
                 font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=(70, 0))
        tk.Label(hdr, text="CALIBRATED", bg=T_BG, fg=T_BLUE,
                 font=("Consolas", 10, "bold")).pack(side=tk.RIGHT, padx=(0, 70))

        self.fig = Figure(figsize=(7.2, 4.2), dpi=100, facecolor=T_FIG)

        # 좌: 원본 / 우: 보정 적용 — 컬러바 포함 4축 고정 배치
        self.ax_raw = self.fig.add_axes([0.05, 0.14, 0.36, 0.76])
        self.cax_raw = self.fig.add_axes([0.42, 0.14, 0.025, 0.76])
        self.ax_cal = self.fig.add_axes([0.52, 0.14, 0.36, 0.76])
        self.cax_cal = self.fig.add_axes([0.89, 0.14, 0.025, 0.76])

        # 격자/보간 사전 계산 (변경 없음)
        SEGMENTS_PER_CHANNEL = 20
        self.x_fine = np.linspace(0, NUM_CHANNELS - 1,
                                  (NUM_CHANNELS - 1) * SEGMENTS_PER_CHANNEL + 1)
        self.nearest_idx = np.clip(np.round(self.x_fine).astype(int), 0, NUM_CHANNELS - 1)
        self.cell_offset = self.x_fine - self.nearest_idx
        self.y_axis = np.linspace(0, 1, self.STRIP_ROWS)
        self.contour_levels = np.linspace(SCALE_MIN, SCALE_MAX, 26)

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget.get_tk_widget().pack(pady=6, padx=6, fill=tk.BOTH, expand=True)

        # 컬러바는 각 축에 한 번만 생성 (매 프레임 재생성 금지)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(norm=Normalize(vmin=SCALE_MIN, vmax=SCALE_MAX), cmap=self.cmap_name)
        sm.set_array([])
        self.cb_raw = self.fig.colorbar(sm, cax=self.cax_raw)
        self.cb_cal = self.fig.colorbar(sm, cax=self.cax_cal)
        for cb in (self.cb_raw, self.cb_cal):
            cb.ax.yaxis.set_tick_params(color=T_TEXT)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color(T_TEXT)

        # ── 채널별 실시간 수치 테이블 ──────────────────────────────────
        tbl = tk.Frame(right, bg=T_BG)
        tbl.pack(fill=tk.X, padx=8, pady=(0, 4))

        FONT = ("Consolas", 7)
        COL_W = 5

        tk.Label(tbl, text="", bg=T_BG, width=4, font=FONT).grid(
            row=0, column=0, padx=1, pady=1)
        for i in range(NUM_CHANNELS):
            tk.Label(tbl, text=f"{i:02d}", bg=T_BG, fg=T_DIM,
                     font=FONT, width=COL_W).grid(row=0, column=i + 1, padx=1)

        tk.Label(tbl, text="RAW", bg=T_BG, fg=T_DIM,
                 font=FONT, width=4).grid(row=1, column=0, padx=1, pady=1)
        self.raw_val_labels = []
        for i in range(NUM_CHANNELS):
            lbl = tk.Label(tbl, text="-", bg=T_CARD, fg=T_TEXT,
                           font=FONT, width=COL_W)
            lbl.grid(row=1, column=i + 1, padx=1, pady=1)
            self.raw_val_labels.append(lbl)

        tk.Label(tbl, text="CAL", bg=T_BG, fg=T_BLUE,
                 font=FONT, width=4).grid(row=2, column=0, padx=1, pady=1)
        self.cal_val_labels = []
        for i in range(NUM_CHANNELS):
            lbl = tk.Label(tbl, text="-", bg=T_CARD, fg=T_BLUE,
                           font=FONT, width=COL_W)
            lbl.grid(row=2, column=i + 1, padx=1, pady=1)
            self.cal_val_labels.append(lbl)

        # ── ML 채널 기여도 행 ─────────────────────────────────────────
        tk.Label(tbl, text="ERR", bg=T_BG, fg=T_RED,
                 font=FONT, width=4).grid(row=3, column=0, padx=1, pady=1)
        self._ml_contrib_labels = []
        _DEAD_SET = set(range(NUM_CHANNELS)) - set(ML_ACTIVE_CH)
        for i in range(NUM_CHANNELS):
            is_dead = i in _DEAD_SET
            lbl = tk.Label(tbl,
                           text="─" if is_dead else "·",
                           bg=T_BORD if is_dead else T_CARD,
                           fg=T_DIM, font=FONT, width=COL_W)
            lbl.grid(row=3, column=i + 1, padx=1, pady=1)
            self._ml_contrib_labels.append(lbl)

        # ── 채널 상태 바 ──────────────────────────────────────────────
        self.ch_status_label = tk.Label(
            right, text="●  수신 대기",
            bg=T_BG, fg=T_DIM,
            font=("Consolas", 8), anchor="w"
        )
        self.ch_status_label.pack(fill=tk.X, padx=12, pady=(0, 5))

        self._draw_contour([SCALE_MAX] * NUM_CHANNELS, force=True)

    # ---------------- 관리자 탭 ----------------
    def _build_admin_tab(self, parent):
        # 로그인 화면과 기록관리 화면, 두 개를 같은 자리에 만들어두고
        # 인증 여부에 따라 하나만 보이도록 전환한다.
        self.admin_login_frame = tk.Frame(parent, bg=T_BG)
        self.admin_content_frame = tk.Frame(parent, bg=T_BG)

        self._build_admin_login_view(self.admin_login_frame)
        self._build_admin_content_view(self.admin_content_frame)

        self.admin_login_frame.pack(fill=tk.BOTH, expand=True)

    def _build_admin_login_view(self, parent):
        box = tk.Frame(parent, bg=T_PANEL, highlightbackground=T_BORD,
                       highlightthickness=1)
        box.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(box, text="🔒  관리자 로그인", bg=T_PANEL, fg=T_TEXT,
                 font=("맑은 고딕", 15, "bold")).pack(pady=(30, 20), padx=60)

        tk.Label(box, text="아이디", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 10)).pack(anchor="w", padx=30)
        self.admin_id_entry = tk.Entry(box, font=("Consolas", 12), width=22)
        self.admin_id_entry.pack(padx=30, pady=(2, 12))

        tk.Label(box, text="비밀번호", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 10)).pack(anchor="w", padx=30)
        self.admin_pw_entry = tk.Entry(box, font=("Consolas", 12), width=22, show="●")
        self.admin_pw_entry.pack(padx=30, pady=(2, 6))
        self.admin_pw_entry.bind("<Return>", lambda e: self._try_admin_login())

        self.admin_login_error = tk.Label(
            box, text="", bg=T_PANEL, fg=T_RED, font=("맑은 고딕", 9)
        )
        self.admin_login_error.pack(pady=(0, 6))

        tk.Button(
            box, text="로그인", font=("맑은 고딕", 11, "bold"),
            bg=T_GREEN, fg="#ffffff", activebackground=T_GRNH,
            relief=tk.FLAT, command=self._try_admin_login
        ).pack(padx=30, pady=(6, 30), fill=tk.X)

    def _try_admin_login(self):
        entered_id = self.admin_id_entry.get().strip()
        entered_pw = self.admin_pw_entry.get()

        if (entered_id == self.admin_config.get("admin_id")
                and _hash_password(entered_pw) == self.admin_config.get("admin_pw_hash")):
            self.admin_authenticated = True
            self.admin_login_error.config(text="")
            self.admin_pw_entry.delete(0, tk.END)
            self.admin_login_frame.pack_forget()
            self.admin_content_frame.pack(fill=tk.BOTH, expand=True)
            self._refresh_records_list()
        else:
            self.admin_login_error.config(text="아이디 또는 비밀번호가 올바르지 않습니다.")
            self.admin_pw_entry.delete(0, tk.END)

    def _admin_logout(self):
        self.admin_authenticated = False
        self.admin_content_frame.pack_forget()
        self.admin_login_frame.pack(fill=tk.BOTH, expand=True)

    def _build_admin_content_view(self, parent):
        top = tk.Frame(parent, bg=T_BG)
        top.pack(fill=tk.X, padx=20, pady=(16, 8))

        tk.Label(top, text="CSV 기록 관리", bg=T_BG, fg=T_TEXT,
                 font=("맑은 고딕", 14, "bold")).pack(side=tk.LEFT)
        tk.Button(top, text="로그아웃", font=("맑은 고딕", 9),
                 bg=T_CARD, fg=T_TEXT, relief=tk.FLAT,
                 command=self._admin_logout).pack(side=tk.RIGHT)

        tk.Label(top, text=f"({os.path.abspath(RECORDS_DIR)})", bg=T_BG,
                fg=T_DIM, font=("Consolas", 8)).pack(side=tk.RIGHT, padx=10)

        # --- 파일 목록 ---
        list_card = tk.Frame(parent, bg=T_PANEL,
                             highlightbackground=T_BORD, highlightthickness=1)
        list_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 8))
        self._sec(list_card, "RECORDS")

        list_frame = tk.Frame(list_card, bg=T_PANEL)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 10))

        columns = ("name", "size", "mtime")
        style = ttk.Style()
        style.configure("Admin.Treeview", background=T_PANEL, fieldbackground=T_PANEL,
                        foreground=T_TEXT, rowheight=24)
        style.configure("Admin.Treeview.Heading", background=T_CARD, foreground=T_TEXT)

        self.records_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", style="Admin.Treeview"
        )
        self.records_tree.heading("name", text="파일명")
        self.records_tree.heading("size", text="크기")
        self.records_tree.heading("mtime", text="저장 시각")
        self.records_tree.column("name", width=340)
        self.records_tree.column("size", width=100, anchor="e")
        self.records_tree.column("mtime", width=170, anchor="center")
        self.records_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.records_tree.bind("<<TreeviewSelect>>", self._on_record_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.records_tree.yview)
        self.records_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # --- 미리보기 / 버튼 ---
        bottom = tk.Frame(parent, bg=T_BG)
        bottom.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.record_preview_label = tk.Label(
            bottom, text="파일을 선택하면 정보가 여기 표시됩니다.",
            bg=T_BG, fg=T_DIM, font=("Consolas", 9), justify="left"
        )
        self.record_preview_label.pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(bottom, bg=T_BG)
        btn_row.pack(fill=tk.X)

        tk.Button(btn_row, text="새로고침", font=("맑은 고딕", 10),
                 bg=T_CARD, fg=T_TEXT, relief=tk.FLAT,
                 command=self._refresh_records_list).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_row, text="폴더 열기", font=("맑은 고딕", 10),
                 bg=T_CARD, fg=T_TEXT, relief=tk.FLAT,
                 command=self._open_records_folder).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="선택 파일 열기", font=("맑은 고딕", 10),
                 bg=T_CARD, fg=T_TEXT, relief=tk.FLAT,
                 command=self._open_selected_record).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="선택 파일 삭제", font=("맑은 고딕", 10),
                 bg=T_RED, fg="#ffffff", relief=tk.FLAT,
                 command=self._delete_selected_record).pack(side=tk.LEFT, padx=6)

        # --- 비밀번호 변경 ---
        pw_frame = tk.Frame(parent, bg=T_PANEL,
                            highlightbackground=T_BORD, highlightthickness=1)
        pw_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        self._sec(pw_frame, "관리자 비밀번호 변경")

        row = tk.Frame(pw_frame, bg=T_PANEL)
        row.pack(fill=tk.X, padx=14, pady=(0, 14))

        tk.Label(row, text="새 비밀번호", bg=T_PANEL, fg=T_DIM,
                font=("맑은 고딕", 9)).pack(side=tk.LEFT)
        self.new_pw_entry = tk.Entry(row, font=("Consolas", 10), show="●", width=16)
        self.new_pw_entry.pack(side=tk.LEFT, padx=(6, 16))

        tk.Label(row, text="확인", bg=T_PANEL, fg=T_DIM,
                font=("맑은 고딕", 9)).pack(side=tk.LEFT)
        self.new_pw_confirm_entry = tk.Entry(row, font=("Consolas", 10), show="●", width=16)
        self.new_pw_confirm_entry.pack(side=tk.LEFT, padx=(6, 16))

        tk.Button(row, text="변경", font=("맑은 고딕", 9, "bold"),
                 bg=T_GREEN, fg="#ffffff", relief=tk.FLAT,
                 command=self._change_admin_password).pack(side=tk.LEFT)

    def _list_record_files(self):
        try:
            names = [f for f in os.listdir(RECORDS_DIR) if f.lower().endswith(".csv")]
        except OSError:
            return []
        files = []
        for name in names:
            path = os.path.join(RECORDS_DIR, name)
            try:
                stat = os.stat(path)
                files.append((name, path, stat.st_size, stat.st_mtime))
            except OSError:
                continue
        files.sort(key=lambda x: x[3], reverse=True)  # 최신 순
        return files

    def _refresh_records_list(self):
        for item in self.records_tree.get_children():
            self.records_tree.delete(item)

        for name, path, size, mtime in self._list_record_files():
            size_str = f"{size/1024:.1f} KB"
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.records_tree.insert("", tk.END, iid=path, values=(name, size_str, time_str))

        self.record_preview_label.config(text="파일을 선택하면 정보가 여기 표시됩니다.")

    def _get_selected_record_path(self):
        selection = self.records_tree.selection()
        if not selection:
            return None
        return selection[0]  # iid에 전체 경로를 넣어뒀음

    def _on_record_select(self, event=None):
        path = self._get_selected_record_path()
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                row_count = sum(1 for _ in f) - 1  # 헤더 제외
            self.record_preview_label.config(
                text=f"{os.path.basename(path)}  —  약 {max(row_count, 0)}줄"
            )
        except OSError as e:
            self.record_preview_label.config(text=f"읽기 실패: {e}")

    def _open_records_folder(self):
        self._open_path(os.path.abspath(RECORDS_DIR))

    def _open_selected_record(self):
        path = self._get_selected_record_path()
        if not path:
            messagebox.showinfo("선택 필요", "먼저 목록에서 파일을 선택하세요.")
            return
        self._open_path(path)

    @staticmethod
    def _open_path(path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except OSError as e:
            messagebox.showerror("열기 실패", str(e))

    def _delete_selected_record(self):
        path = self._get_selected_record_path()
        if not path:
            messagebox.showinfo("선택 필요", "먼저 목록에서 파일을 선택하세요.")
            return
        if not messagebox.askyesno("삭제 확인", f"{os.path.basename(path)} 파일을 삭제할까요?"):
            return
        try:
            os.remove(path)
        except OSError as e:
            messagebox.showerror("삭제 실패", str(e))
            return
        self._refresh_records_list()

    def _change_admin_password(self):
        new_pw = self.new_pw_entry.get()
        confirm_pw = self.new_pw_confirm_entry.get()

        if len(new_pw) < 4:
            messagebox.showwarning("비밀번호 변경", "비밀번호는 4자 이상이어야 합니다.")
            return
        if new_pw != confirm_pw:
            messagebox.showwarning("비밀번호 변경", "두 비밀번호가 서로 다릅니다.")
            return

        self.admin_config["admin_pw_hash"] = _hash_password(new_pw)
        save_admin_config(self.admin_config)
        self.new_pw_entry.delete(0, tk.END)
        self.new_pw_confirm_entry.delete(0, tk.END)
        messagebox.showinfo("비밀번호 변경", "비밀번호가 변경되었습니다.")

    def _style_axes(self, ax):
        ax.set_facecolor(T_FIG)
        ax.set_xticks(range(NUM_CHANNELS))
        ax.set_xlabel("ch", color=T_DIM, fontsize=7)
        ax.set_yticks([])
        ax.tick_params(colors=T_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(T_BORD)

    def _retheme_widget(self, widget, color_map):
        for opt in ("bg", "fg", "background", "foreground",
                    "activebackground", "activeforeground",
                    "insertbackground", "troughcolor", "highlightbackground"):
            try:
                cur = widget.cget(opt)
                if cur in color_map:
                    widget.config(**{opt: color_map[cur]})
            except (tk.TclError, AttributeError):
                pass
        for child in widget.winfo_children():
            self._retheme_widget(child, color_map)

    def _toggle_theme(self):
        global T_BG, T_PANEL, T_CARD, T_BORD, T_TEXT, T_DIM
        global T_GREEN, T_GRNH, T_BLUE, T_ORNG, T_RED, T_FIG, T_HDR
        old_t = DARK if self._is_dark else LIGHT
        new_t = LIGHT if self._is_dark else DARK
        self._is_dark = not self._is_dark

        color_map = {v: new_t[k] for k, v in old_t.items()}
        self._retheme_widget(self.root, color_map)

        T_BG = new_t["BG"]; T_PANEL = new_t["PANEL"]; T_CARD = new_t["CARD"]
        T_BORD = new_t["BORD"]; T_TEXT = new_t["TEXT"]; T_DIM = new_t["DIM"]
        T_GREEN = new_t["GREEN"]; T_GRNH = new_t["GRNH"]; T_BLUE = new_t["BLUE"]
        T_ORNG = new_t["ORNG"]; T_RED = new_t["RED"]
        T_FIG = new_t["FIG"]; T_HDR = new_t["HDR"]

        self._warn_bg = new_t["WARN_BG"]; self._warn_fg = new_t["WARN_FG"]
        self._cal_fg_press = new_t["CAL_FG_PRESS"]

        # matplotlib 메인 figure 업데이트
        self.fig.set_facecolor(T_FIG)
        for ax in (self.ax_raw, self.ax_cal):
            self._style_axes(ax)
        for cb in (self.cb_raw, self.cb_cal):
            cb.ax.set_facecolor(T_FIG)
            cb.ax.yaxis.set_tick_params(color=T_TEXT)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color(T_TEXT)
        self.canvas_widget.draw_idle()

        # matplotlib 보정 figure 업데이트
        self.cal_fig.set_facecolor(T_FIG)
        for ax in self.cal_axes:
            ax.set_facecolor(T_CARD)
            title = ax.get_title()
            ax.set_title(title, color=T_DIM, fontsize=7, pad=2)
            ax.tick_params(colors=T_DIM, labelsize=5)
            for spine in ax.spines.values():
                spine.set_color(T_BORD)
        self.cal_canvas.draw_idle()

        # 토글 버튼 아이콘
        self.theme_btn.config(text="🌙" if self._is_dark else "☀",
                              fg=T_DIM, bg=T_HDR, activebackground=T_HDR)

    # ---------------- ML 이상 감지 ----------------
    def _load_ml_model(self):
        if not _ensure_torch():
            msg = "ML: PyTorch 로드 실패"
            if _TORCH_ERROR:
                msg = f"{msg} - {_TORCH_ERROR[:60]}"
            self._ml_label.config(text=msg, fg=T_DIM)
            return
        mp, sp = ML_MODEL_PATH, ML_STATS_PATH
        if not (os.path.exists(mp) and os.path.exists(sp)):
            return
        try:
            model = _create_lstm_autoencoder()
            if model is None:
                msg = "ML: PyTorch 로드 실패"
                if _TORCH_ERROR:
                    msg = f"{msg} - {_TORCH_ERROR[:60]}"
                self._ml_label.config(text=msg, fg=T_DIM)
                return
            model.load_state_dict(
                torch.load(mp, map_location='cpu', weights_only=True))
            model.eval()
            stats = np.load(sp)
            self._ml_model    = model
            self._ml_mean_err = float(stats['mean_err'])
            self._ml_std_err  = float(stats['std_err'])
            # 현재 sigma k 로 임계값 재계산 (슬라이더 기억)
            self._ml_threshold = self._ml_mean_err + self._ml_sigma_k * self._ml_std_err
            self._ml_sigma_slider.config(state=tk.NORMAL)
            self._ml_sigma_slider.set(self._ml_sigma_k)
            self._ml_sigma_label.config(text=f"{self._ml_sigma_k:.1f} σ")
            self._ml_label.config(
                text=f"ML: 로드됨  임계={self._ml_threshold:.4f}", fg=T_BLUE)
        except Exception:
            self._ml_label.config(text="ML: 로드 실패", fg=T_RED)

    # ─────────────────────────────────────────────────────────────────
    # 알람 (소리 / Windows 토스트)
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # 채널 ON/OFF 필터
    # ─────────────────────────────────────────────────────────────────
    _DEAD_CH_SET = {1, 5, 14, 15}

    def _toggle_ch_filter(self, ch):
        if ch in self._DEAD_CH_SET:
            return
        self._ch_enabled[ch] = not self._ch_enabled[ch]
        btn = self._ch_filter_btns[ch]
        if self._ch_enabled[ch]:
            btn.config(bg=T_GREEN, fg="#ffffff")
        else:
            self._prev_anomaly[ch] = None
            self._ch_history[ch].clear()
            btn.config(bg=T_CARD, fg=T_DIM)

    def _ch_filter_all_on(self):
        for ch in range(NUM_CHANNELS):
            if ch not in self._DEAD_CH_SET:
                self._ch_enabled[ch] = True
                self._ch_filter_btns[ch].config(bg=T_GREEN, fg="#ffffff")

    def _ch_filter_all_off(self):
        for ch in range(NUM_CHANNELS):
            if ch not in self._DEAD_CH_SET:
                self._ch_enabled[ch] = False
                self._prev_anomaly[ch] = None
                self._ch_history[ch].clear()
                self._ch_filter_btns[ch].config(bg=T_CARD, fg=T_DIM)

    def _toggle_alarm(self):
        self._alarm_enabled = not self._alarm_enabled
        if self._alarm_enabled:
            self._alarm_btn.config(text="알람 ON",  bg=T_RED,  fg="#ffffff")
        else:
            self._alarm_btn.config(text="알람 OFF", bg=T_CARD, fg=T_DIM)

    def _on_alarm_cooldown_change(self, val):
        self._alarm_cooldown = float(val)
        self._alarm_cooldown_lbl.config(text=f"{int(float(val))}s")

    def _fire_alarm(self, reason=None):
        """이상 감지 알람. reason이 없으면 ML 점수 기반 문구를 사용(ML 경로),
        있으면 그대로 토스트 본문에 사용(규칙 기반 채널 이상 경로)."""
        if not self._alarm_enabled:
            return
        now = time.time()
        if now - self._alarm_last_t < self._alarm_cooldown:
            return
        self._alarm_last_t = now
        mode = self._alarm_mode
        if "소리" in mode:
            threading.Thread(target=self._do_beep, daemon=True).start()
        if "토스트" in mode:
            threading.Thread(target=self._do_toast, args=(reason,), daemon=True).start()

    def _do_beep(self):
        try:
            import winsound
            for freq, dur in [(880, 200), (1100, 200), (1320, 350)]:
                winsound.Beep(freq, dur)
                time.sleep(0.05)
        except Exception:
            pass

    def _do_toast(self, reason=None):
        title = "Pressure Anomaly Detected"
        if reason:
            body = reason
        else:
            score  = self._ml_score
            thresh = self._ml_threshold if self._ml_threshold else 0.0
            body   = f"ML score {score:.4f} > threshold {thresh:.4f}"
        try:
            cmd = (
                'Add-Type -AssemblyName System.Windows.Forms; '
                '$n = New-Object System.Windows.Forms.NotifyIcon; '
                '$n.Icon = [System.Drawing.SystemIcons]::Warning; '
                '$n.Visible = $True; '
                f'$n.BalloonTipTitle = "{title}"; '
                f'$n.BalloonTipText  = "{body}"; '
                '$n.ShowBalloonTip(6000); '
                'Start-Sleep -Seconds 7; '
                '$n.Dispose()'
            )
            subprocess.Popen(
                ['powershell', '-WindowStyle', 'Hidden', '-Command', cmd],
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            pass

    def _show_page(self, name):
        """왼쪽 네비 레일 페이지 전환: notebook 대신 frame pack/pack_forget 스왑."""
        if name not in self._pages:
            return
        for frame in self._pages.values():
            frame.pack_forget()
        self._pages[name].pack(fill=tk.BOTH, expand=True)
        self._current_page = name
        for key, btn in self._nav_buttons.items():
            if key == name:
                btn.config(bg=T_BLUE, fg="#ffffff")
            else:
                btn.config(bg=T_HDR, fg=T_DIM)
        if name == 'ml':
            self._update_ml_graph()

    def _on_sigma_change(self, val):
        k = float(val)
        self._ml_sigma_k = k
        self._ml_sigma_label.config(text=f"{k:.1f} σ")
        if self._ml_std_err > 0:
            self._ml_threshold = self._ml_mean_err + k * self._ml_std_err
            # 현재 점수 기준으로 레이블 즉시 갱신
            score = self._ml_score
            thresh = self._ml_threshold
            tag = "⚠ 이상" if score > thresh else "정상"
            self._ml_label.config(
                text=f"ML: {score:.4f} / {thresh:.4f}  {tag}  [{k:.1f}σ]",
                fg=T_RED if score > thresh else T_DIM)
            ratio = min(1.0, score / thresh)
            color = T_RED if score > thresh else (T_ORNG if ratio > 0.7 else T_GREEN)
            w = max(1, self._ml_canvas.winfo_width())
            self._ml_canvas.coords(self._ml_bar, 0, 0, int(w * ratio), 7)
            self._ml_canvas.itemconfig(self._ml_bar, fill=color)
            self._update_ml_graph()   # σ 변경 시 임계선 즉시 갱신

    def _run_ml_inference(self):
        raw      = np.array(self._ml_buffer, dtype=np.float32)
        active   = raw[:, ML_ACTIVE_CH]
        pressure = (4095 - active) / 4095.0
        # 비활성화된 채널은 압력 0(정상값)으로 zeroing
        for _ai, _ci in enumerate(ML_ACTIVE_CH):
            if not self._ch_enabled[_ci]:
                pressure[:, _ai] = 0.0
        t = torch.from_numpy(pressure).unsqueeze(0)  # (1, 30, 12)
        with torch.no_grad():
            pred    = self._ml_model(t)
            sq_err  = (pred - t) ** 2              # (1, 30, 12)
            score   = float(sq_err.mean())
            ch_err  = sq_err.mean(dim=(0, 1)).numpy()   # (12,) per active channel

        # 16채널 기여도 배열 구성 (사망 채널 = 0)
        contrib16 = np.zeros(NUM_CHANNELS, dtype=float)
        for ai, ci in enumerate(ML_ACTIVE_CH):
            contrib16[ci] = float(ch_err[ai])

        self._ml_score = score
        thresh     = self._ml_threshold
        is_anomaly = score > thresh
        ratio      = min(1.0, score / thresh)
        color      = T_RED if is_anomaly else (T_ORNG if ratio > 0.7 else T_GREEN)
        w = max(1, self._ml_canvas.winfo_width())
        self._ml_canvas.coords(self._ml_bar, 0, 0, int(w * ratio), 7)
        self._ml_canvas.itemconfig(self._ml_bar, fill=color)
        tag = "⚠ 이상" if is_anomaly else "정상"
        self._ml_label.config(
            text=f"ML: {score:.4f} / {thresh:.4f}  {tag}  [{self._ml_sigma_k:.1f}σ]",
            fg=T_RED if is_anomaly else T_DIM)

        # 채널 기여도 히트맵 갱신
        self._update_ml_contrib(contrib16, is_anomaly)

        frame_data = raw[-1].tolist()

        # 상태 변화 시에만 로그 기록 + 클립 처리
        if is_anomaly != self._ml_was_anomaly:
            event = "감지" if is_anomaly else "복구"
            self._log_ml_anomaly(event, score, raw[-1])
            self._ml_was_anomaly = is_anomaly
            if is_anomaly:
                self._fire_alarm()
                # 이상 감지 시작 — 직전 문맥 포함해서 녹화 시작
                self._clip_recording = True
                self._clip_start_dt  = datetime.datetime.now()
                self._clip_rec_buf   = [(f, s, False) for f, s in self._clip_pre_buf]
                self._clip_rec_buf.append((frame_data, score, True))
            else:
                # 복구 — 첫 정상 프레임 추가 후 저장
                self._clip_rec_buf.append((frame_data, score, False))
                self._clip_recording = False
                self._save_clip()
        elif self._clip_recording:
            self._clip_rec_buf.append((frame_data, score, True))
        else:
            self._clip_pre_buf.append((frame_data, score))

        # ML 점수 탭 추이 그래프 업데이트
        self._ml_score_history.append(score)
        self._ml_anomaly_history.append(is_anomaly)
        self._update_ml_graph()

    @staticmethod
    def _lerp_color(c1, c2, t):
        """두 hex 색상 사이를 t(0~1)로 선형 보간."""
        c1 = c1.lstrip('#'); c2 = c2.lstrip('#')
        r = int(int(c1[0:2], 16) * (1 - t) + int(c2[0:2], 16) * t)
        g = int(int(c1[2:4], 16) * (1 - t) + int(c2[2:4], 16) * t)
        b = int(int(c1[4:6], 16) * (1 - t) + int(c2[4:6], 16) * t)
        return f'#{r:02x}{g:02x}{b:02x}'

    # ─────────────────────────────────────────────────────────────────
    # ML 점수 탭 — 실시간 추이 그래프
    # ─────────────────────────────────────────────────────────────────

    def _make_ml_metric_card(self, parent, title, value, sub, accent):
        card = tk.Frame(parent, bg=T_PANEL, highlightbackground=T_BORD,
                        highlightthickness=1)
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

        tk.Label(card, text=title, bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 8, "bold"), anchor="w"
                 ).pack(fill=tk.X, padx=12, pady=(8, 0))
        value_lbl = tk.Label(card, text=value, bg=T_PANEL, fg=accent,
                             font=("Consolas", 19, "bold"), anchor="w")
        value_lbl.pack(fill=tk.X, padx=12, pady=(1, 0))
        sub_lbl = tk.Label(card, text=sub, bg=T_PANEL, fg=T_DIM,
                           font=("맑은 고딕", 7), anchor="w")
        sub_lbl.pack(fill=tk.X, padx=12, pady=(0, 8))
        return {"frame": card, "value": value_lbl, "sub": sub_lbl}

    def _set_ml_metric_card(self, card, value, sub, accent):
        card["value"].config(text=value, fg=accent)
        card["sub"].config(text=sub)
        card["frame"].config(highlightbackground=accent if accent != T_DIM else T_BORD)

    def _build_ml_tab(self, parent):
        # ── 상단 지표 카드 ───────────────────────────────────────────
        action_bar = tk.Frame(parent, bg=T_PANEL, highlightbackground=T_BORD,
                              highlightthickness=1)
        action_bar.pack(fill=tk.X, padx=14, pady=(12, 6))

        tk.Label(
            action_bar, text="ML MODEL", bg=T_PANEL, fg=T_BLUE,
            font=("Consolas", 10, "bold")
        ).pack(side=tk.LEFT, padx=(14, 10), pady=10)

        tk.Button(
            action_bar, text="ML 모델 훈련", font=("맑은 고딕", 13, "bold"),
            bg=T_BLUE, fg="#ffffff", activebackground=T_BORD,
            activeforeground="#ffffff", relief=tk.FLAT,
            padx=24, pady=9, command=self._train_ml_model
        ).pack(side=tk.LEFT, pady=8)

        tk.Button(
            action_bar, text="모델 정보", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, padx=14, pady=7, command=self._open_ml_report
        ).pack(side=tk.LEFT, padx=(8, 0), pady=8)

        tk.Button(
            action_bar, text="ML 로그 열기", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, padx=14, pady=7, command=self._open_ml_log
        ).pack(side=tk.LEFT, padx=(8, 0), pady=8)

        tk.Label(
            action_bar, text="CSV를 선택하면 anomaly_model.pt와 anomaly_stats.npz를 생성합니다.",
            bg=T_PANEL, fg=T_DIM, font=("맑은 고딕", 9)
        ).pack(side=tk.RIGHT, padx=14, pady=10)

        hdr = tk.Frame(parent, bg=T_BG)
        hdr.pack(fill=tk.X, padx=14, pady=(0, 8))

        self._ml_score_card = self._make_ml_metric_card(
            hdr, "CURRENT SCORE", "—", "ML 모델 대기", T_DIM)
        self._ml_thresh_card = self._make_ml_metric_card(
            hdr, "THRESHOLD", "—", f"{self._ml_sigma_k:.1f}σ 기준", T_DIM)
        self._ml_state_card = self._make_ml_metric_card(
            hdr, "STATUS", "모델 없음", "실시간 수신 대기", T_DIM)

        # ── 규칙 임계값 슬라이더 행 ──────────────────────────────────
        rule_row = tk.Frame(parent, bg=T_PANEL, highlightbackground=T_BORD,
                            highlightthickness=1)
        rule_row.pack(fill=tk.X, padx=18, pady=(0, 8))

        tk.Label(rule_row, text="규칙 임계", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 8, "bold")).pack(side=tk.LEFT, padx=(12, 8), pady=6)

        self._rule_thresh_var = tk.DoubleVar(value=self._rule_thresh)
        self._rule_thresh_slider = tk.Scale(
            rule_row,
            variable=self._rule_thresh_var,
            from_=0.05, to=0.95, resolution=0.01,
            orient=tk.HORIZONTAL, length=340,
            bg=T_PANEL, fg=T_DIM, troughcolor=T_CARD,
            highlightthickness=0, bd=0, showvalue=False,
            command=self._on_rule_thresh_change)
        self._rule_thresh_slider.pack(side=tk.LEFT, padx=(0, 8), pady=2)

        self._rule_thresh_val_lbl = tk.Label(
            rule_row, text=f"{self._rule_thresh:.2f}",
            bg=T_PANEL, fg=T_ORNG, font=("Consolas", 11, "bold"), width=5)
        self._rule_thresh_val_lbl.pack(side=tk.LEFT, pady=6)

        tk.Label(rule_row, text="민감  ←  →  둔감", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 7)).pack(side=tk.RIGHT, padx=(0, 12), pady=6)

        # ── 클립 목록 패널 (bottom 선 확보, canvas 이전에 pack) ────────
        clip_panel = tk.Frame(parent, bg=T_PANEL,
                              highlightbackground=T_BORD, highlightthickness=1)
        clip_panel.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=(4, 8))

        clip_hdr = tk.Frame(clip_panel, bg=T_PANEL)
        clip_hdr.pack(fill=tk.X, padx=10, pady=(6, 3))
        self._clip_count_lbl = tk.Label(
            clip_hdr, text="저장된 클립 없음", bg=T_PANEL, fg=T_DIM,
            font=("맑은 고딕", 8, "bold"))
        self._clip_count_lbl.pack(side=tk.LEFT)
        tk.Button(
            clip_hdr, text="폴더 열기", font=("맑은 고딕", 7),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=2, command=self._open_clips_dir
        ).pack(side=tk.RIGHT)
        tk.Button(
            clip_hdr, text="클립 비교", font=("맑은 고딕", 7),
            bg=T_CARD, fg=T_BLUE, activebackground=T_BORD,
            relief=tk.FLAT, pady=2, command=self._open_clip_compare
        ).pack(side=tk.RIGHT, padx=(0, 6))

        self._clip_listbox = tk.Listbox(
            clip_panel, height=3,
            bg=T_PANEL, fg=T_TEXT, selectbackground=T_CARD,
            selectforeground=T_BLUE, font=("Consolas", 8),
            bd=0, highlightthickness=0, relief=tk.FLAT,
            selectmode=tk.EXTENDED)
        self._clip_listbox.pack(fill=tk.X, padx=10, pady=(0, 6))
        self._clip_listbox.bind('<Double-Button-1>', self._open_selected_clip)

        self._refresh_clip_list()  # 기존 클립 로드

        # ── matplotlib 2-subplot 그래프 (ML 위 / 규칙 기반 아래) ─────
        self._ml_score_fig = Figure(figsize=(8.6, 5.8), dpi=100, facecolor=T_FIG)
        gs = self._ml_score_fig.add_gridspec(
            2, 1, height_ratios=[3.2, 2], hspace=0.38,
            left=0.08, right=0.975, top=0.93, bottom=0.085)
        self._ml_score_ax = self._ml_score_fig.add_subplot(gs[0])
        self._rule_ax     = self._ml_score_fig.add_subplot(gs[1])

        self._ml_score_canvas = FigureCanvasTkAgg(self._ml_score_fig, master=parent)
        self._ml_score_canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # 초기 빈 화면
        self._ml_draw_placeholder()

    # ─────────────────────────────────────────────────────────────────
    # 이상 구간 자동 클립 저장
    # ─────────────────────────────────────────────────────────────────

    def _save_clip(self):
        if not self._clip_rec_buf or self._clip_start_dt is None:
            self._clip_rec_buf = []
            self._clip_recording = False
            self._clip_start_dt = None
            return
        os.makedirs(ML_CLIPS_DIR, exist_ok=True)
        ts   = self._clip_start_dt.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(ML_CLIPS_DIR, f'clip_{ts}.csv')
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                w.writerow(
                    ['frame'] + [f'ch{i}' for i in range(16)] +
                    ['ml_score', 'is_anomaly'])
                for idx, (frame_data, sc, anom) in enumerate(self._clip_rec_buf):
                    w.writerow(
                        [idx] + [f'{v:.0f}' for v in frame_data] +
                        [f'{sc:.6f}', '1' if anom else '0'])
        except OSError:
            pass
        finally:
            self._clip_rec_buf = []
            self._clip_recording = False
            self._clip_start_dt = None
        self._refresh_clip_list()

    def _flush_active_clip(self):
        if self._clip_recording and self._clip_rec_buf:
            self._save_clip()

    def _refresh_clip_list(self):
        if not hasattr(self, '_clip_listbox'):
            return
        pattern = os.path.join(ML_CLIPS_DIR, 'clip_*.csv')
        clips = sorted(_glob.glob(pattern), reverse=True)[:10]
        self._clip_paths = clips
        self._clip_listbox.delete(0, tk.END)
        if not clips:
            self._clip_count_lbl.config(text="저장된 클립 없음")
            return
        self._clip_count_lbl.config(text=f"클립 {len(clips)}개 (최근 10개)")
        for p in clips:
            name = os.path.basename(p)
            try:
                with open(p, 'r', encoding='utf-8-sig') as f:
                    n_rows = max(0, sum(1 for _ in f) - 1)
            except OSError:
                n_rows = 0
            # 이상 프레임 수 계산
            try:
                with open(p, 'r', encoding='utf-8-sig') as f:
                    rdr = csv.DictReader(f)
                    anom_cnt = sum(1 for r in rdr if r.get('is_anomaly') == '1')
            except OSError:
                anom_cnt = 0
            self._clip_listbox.insert(
                tk.END,
                f"  {name}   {n_rows}프레임 (이상 {anom_cnt}프레임)")

    def _open_clips_dir(self):
        d = os.path.abspath(ML_CLIPS_DIR)
        os.makedirs(d, exist_ok=True)
        self._open_path(d)

    def _open_selected_clip(self, event=None):
        sel = self._clip_listbox.curselection()
        if not sel or sel[0] >= len(self._clip_paths):
            return
        self._open_path(self._clip_paths[sel[0]])

    def _open_clip_compare(self):
        if hasattr(self, '_compare_win') and self._compare_win.winfo_exists():
            self._compare_win.lift()
            return

        popup = tk.Toplevel(self.root)
        popup.title("클립 비교 분석")
        popup.configure(bg=T_BG)
        popup.geometry("960x700")
        popup.resizable(True, True)
        self._compare_win = popup

        # ─── 파일 선택 행 ─────────────────────────────────────────────
        sel_frame = tk.Frame(popup, bg=T_PANEL,
                             highlightbackground=T_BORD, highlightthickness=1)
        sel_frame.pack(fill=tk.X, padx=16, pady=(14, 6))

        clip_a_path = [None]; clip_a_var = tk.StringVar(value="파일 선택...")
        clip_b_path = [None]; clip_b_var = tk.StringVar(value="파일 선택...")

        def _choose(var, holder):
            init = os.path.abspath(ML_CLIPS_DIR) if os.path.isdir(ML_CLIPS_DIR) else '.'
            p = filedialog.askopenfilename(
                parent=popup, title="클립 선택", initialdir=init,
                filetypes=[("CSV 클립", "clip_*.csv"),
                           ("CSV 파일", "*.csv"), ("모든 파일", "*.*")])
            if p:
                holder[0] = p; var.set(os.path.basename(p))

        for row_idx, (label, color, var, holder) in enumerate([
            ("클립 A:", T_BLUE,  clip_a_var, clip_a_path),
            ("클립 B:", T_ORNG,  clip_b_var, clip_b_path),
        ]):
            row = tk.Frame(sel_frame, bg=T_PANEL)
            row.pack(fill=tk.X, padx=10, pady=(6 if row_idx == 0 else 2, 6 if row_idx == 1 else 2))
            tk.Label(row, text=label, bg=T_PANEL, fg=color,
                     font=("Consolas", 8, "bold"), width=8).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg=T_PANEL, fg=T_TEXT,
                     font=("Consolas", 8), anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(row, text="파일 선택", font=("Consolas", 7),
                      bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                      relief=tk.FLAT, pady=2,
                      command=lambda v=var, h=holder: _choose(v, h)
                      ).pack(side=tk.RIGHT)

        cmp_btn = tk.Button(popup, text="비교 분석", font=("맑은 고딕", 9),
                            bg=T_GREEN, fg="#ffffff", activebackground=T_GRNH,
                            relief=tk.FLAT, pady=5)
        cmp_btn.pack(fill=tk.X, padx=16, pady=(0, 6))

        # ─── matplotlib 3-subplot ─────────────────────────────────────
        fig = Figure(figsize=(9.5, 5.2), dpi=100, facecolor=T_FIG)
        gs  = fig.add_gridspec(2, 2, height_ratios=[2, 1.5],
                               hspace=0.48, wspace=0.28,
                               left=0.07, right=0.97, top=0.93, bottom=0.09)
        ax_a   = fig.add_subplot(gs[0, 0])
        ax_b   = fig.add_subplot(gs[0, 1])
        ax_bar = fig.add_subplot(gs[1, :])
        cmp_canvas = FigureCanvasTkAgg(fig, master=popup)
        cmp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 4))

        # ─── 통계 요약 레이블 ─────────────────────────────────────────
        stats_lbl = tk.Label(popup, text="클립을 선택하고 [비교 분석]을 클릭하세요.",
                             bg=T_BG, fg=T_DIM, font=("Consolas", 7), anchor='w')
        stats_lbl.pack(fill=tk.X, padx=16, pady=(0, 4))

        tk.Button(popup, text="닫기", font=("맑은 고딕", 9),
                  bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                  relief=tk.FLAT, pady=4,
                  command=popup.destroy).pack(fill=tk.X, padx=16, pady=(0, 12))

        # ─── 내부 함수 ────────────────────────────────────────────────
        def _load_clip(path):
            with open(path, 'r', encoding='utf-8-sig') as f:
                rows = list(csv.DictReader(f))
            n = len(rows)
            frames  = np.zeros((n, 16))
            scores  = np.zeros(n)
            is_anom = np.zeros(n, dtype=bool)
            for i, r in enumerate(rows):
                for ch in range(16):
                    try: frames[i, ch] = float(r[f'ch{ch}'])
                    except (KeyError, ValueError): pass
                try: scores[i] = float(r['ml_score'])
                except (KeyError, ValueError): pass
                is_anom[i] = r.get('is_anomaly', '0').strip() == '1'
            return {'frames': frames, 'scores': scores, 'is_anom': is_anom, 'n': n}

        def _draw_series(ax, data, title, line_col, thresh):
            ax.cla(); ax.set_facecolor(T_PANEL)
            n = data['n']; xs = range(n)
            scores = data['scores']; is_anom = data['is_anom']
            # 이상 음영
            in_s = False; s0 = 0
            for i, a in enumerate(is_anom):
                if a and not in_s: in_s = True; s0 = i
                elif not a and in_s:
                    ax.axvspan(s0, i, color=T_RED, alpha=0.15, linewidth=0); in_s = False
            if in_s: ax.axvspan(s0, n - 1, color=T_RED, alpha=0.15, linewidth=0)
            ax.axhline(thresh, color=T_RED, linewidth=0.9, linestyle='--',
                       alpha=0.85, label=f'임계값 {thresh:.4f}')
            ax.plot(xs, scores, color=line_col, linewidth=1.1, label='ML 오차')
            ax.set_title(title, color=T_TEXT, fontsize=8, pad=4)
            ax.set_ylabel("복원 오차", color=T_DIM, fontsize=7)
            ax.set_xlabel("프레임", color=T_DIM, fontsize=7)
            ax.tick_params(colors=T_DIM, labelsize=6)
            for sp in ('bottom', 'left'): ax.spines[sp].set_color(T_BORD)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(fontsize=6, facecolor=T_CARD, edgecolor=T_BORD, labelcolor=T_TEXT,
                      loc='upper right')

        def _draw_bar(ax, da, db, na, nb):
            ax.cla(); ax.set_facecolor(T_PANEL)
            active = ML_ACTIVE_CH
            def ch_press(data):
                raw = data['frames'][:, active]
                p = (4095 - raw) / 4095
                mask = data['is_anom']
                return p[mask].mean(axis=0) if mask.any() else p.mean(axis=0)

            ma = ch_press(da); mb = ch_press(db)
            x = np.arange(len(active)); w = 0.38
            ax.bar(x - w/2, ma, w, color=T_BLUE,  alpha=0.85, label=na)
            ax.bar(x + w/2, mb, w, color=T_ORNG, alpha=0.85, label=nb)
            ax.set_xticks(x)
            ax.set_xticklabels([f'ch{c:02d}' for c in active],
                               fontsize=6, color=T_DIM)
            ax.set_ylabel("평균 압력 (정규화)", color=T_DIM, fontsize=7)
            anom_note = "(이상 구간 평균, 이상 없으면 전체 평균)"
            ax.set_title(f"채널별 평균 압력 비교  {anom_note}",
                         color=T_TEXT, fontsize=8, pad=4)
            ax.set_ylim(0, 1.12)
            ax.tick_params(colors=T_DIM, labelsize=6)
            for sp in ('bottom', 'left'): ax.spines[sp].set_color(T_BORD)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.legend(fontsize=7, facecolor=T_CARD, edgecolor=T_BORD, labelcolor=T_TEXT)

        def _run_compare():
            if not clip_a_path[0] or not clip_b_path[0]:
                messagebox.showwarning("파일 미선택", "클립 A와 B를 모두 선택하세요.",
                                       parent=popup)
                return
            try:
                da = _load_clip(clip_a_path[0])
                db = _load_clip(clip_b_path[0])
            except Exception as e:
                messagebox.showerror("로드 오류", str(e), parent=popup)
                return
            thresh = self._ml_threshold if self._ml_threshold else 0.003
            na = os.path.basename(clip_a_path[0])
            nb = os.path.basename(clip_b_path[0])
            short_a = na[:22] + ("…" if len(na) > 22 else "")
            short_b = nb[:22] + ("…" if len(nb) > 22 else "")
            _draw_series(ax_a, da, f"클립 A: {short_a}", T_BLUE, thresh)
            _draw_series(ax_b, db, f"클립 B: {short_b}", T_ORNG, thresh)
            _draw_bar(ax_bar, da, db, "클립 A", "클립 B")
            cmp_canvas.draw()

            def _stat(d, label):
                n = d['n']; anom = int(d['is_anom'].sum())
                pct = anom / n * 100 if n else 0.0
                peak = float(d['scores'].max()) if n else 0.0
                return (f"{label}: {n}프레임  이상 {anom}f ({pct:.1f}%)"
                        f"  peak={peak:.5f}")
            stats_lbl.config(
                text=f"{_stat(da, '클립A')}    │    {_stat(db, '클립B')}")

        cmp_btn.config(command=_run_compare)

        # 리스트박스 선택 항목 2개로 사전 채움
        sel = self._clip_listbox.curselection()
        pre = [self._clip_paths[i] for i in sel if i < len(self._clip_paths)]
        if not pre:
            pre = self._clip_paths[:2]
        for i, p in enumerate(pre[:2]):
            if i == 0: clip_a_path[0] = p; clip_a_var.set(os.path.basename(p))
            else:      clip_b_path[0] = p; clip_b_var.set(os.path.basename(p))
        if clip_a_path[0] and clip_b_path[0]:
            popup.after(100, _run_compare)   # 팝업 렌더 후 자동 비교

    def _on_rule_thresh_change(self, val):
        self._rule_thresh = float(val)
        self._rule_thresh_val_lbl.config(text=f"{self._rule_thresh:.2f}")
        self._update_ml_graph()

    def _ml_draw_placeholder(self):
        if hasattr(self, '_ml_score_card'):
            self._set_ml_metric_card(self._ml_score_card, "—", "ML 모델 대기", T_DIM)
            self._set_ml_metric_card(
                self._ml_thresh_card, "—", f"{self._ml_sigma_k:.1f}σ 기준", T_DIM)
            self._set_ml_metric_card(self._ml_state_card, "모델 없음", "실시간 수신 대기", T_DIM)
        for ax in (self._ml_score_ax, self._rule_ax):
            ax.cla()
            ax.set_facecolor(T_PANEL)
            for sp in ax.spines.values():
                sp.set_color(T_BORD)
            ax.tick_params(colors=T_DIM)
        self._ml_score_ax.text(
            0.5, 0.5,
            "ML 모델이 로드되지 않았습니다.\n'ML 모델 훈련' 버튼으로 학습 후 실시간 수신을 시작하세요.",
            ha='center', va='center', transform=self._ml_score_ax.transAxes,
            color=T_DIM, fontsize=10)
        self._rule_ax.text(
            0.5, 0.5, "규칙 기반: 실시간 수신 시 표시",
            ha='center', va='center', transform=self._rule_ax.transAxes,
            color=T_DIM, fontsize=9)
        self._ml_score_canvas.draw()

    def _update_ml_graph(self):
        """ML 점수 탭이 활성일 때만 그래프를 갱신."""
        if self._current_page != 'ml':
            return
        if self._ml_model is None or not self._ml_score_history:
            return

        scores   = list(self._ml_score_history)
        is_anom  = list(self._ml_anomaly_history)
        n        = len(scores)
        thresh   = self._ml_threshold
        cur_anom = is_anom[-1] if is_anom else False
        plot_bg  = "#101820" if self._is_dark else "#ffffff"
        grid_col = "#2f3b45" if self._is_dark else "#d8dee4"

        def _style_trend_axis(axis):
            axis.set_facecolor(plot_bg)
            axis.grid(True, color=grid_col, linewidth=0.55, alpha=0.42)
            axis.set_axisbelow(True)
            axis.tick_params(colors=T_DIM, labelsize=7, length=0)
            for sp in ('bottom', 'left'):
                axis.spines[sp].set_color(T_BORD)
                axis.spines[sp].set_linewidth(0.8)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)

        ax = self._ml_score_ax
        ax.cla()
        _style_trend_axis(ax)

        xs = list(range(n))

        # 이상 구간 배경 음영
        in_span = False; span_start = 0
        for i, a in enumerate(is_anom):
            if a and not in_span:
                in_span = True; span_start = i
            elif not a and in_span:
                ax.axvspan(span_start, i, color=T_RED, alpha=0.14, linewidth=0)
                in_span = False
        if in_span:
            ax.axvspan(span_start, n - 1, color=T_RED, alpha=0.14, linewidth=0)

        # 임계선
        ax.axhline(thresh, color=T_RED, linewidth=1.15, linestyle=(0, (6, 4)), alpha=0.9,
                   label=f'임계값 ({self._ml_sigma_k:.1f}σ)  {thresh:.5f}')

        # 점수 곡선
        ratio = scores[-1] / thresh if thresh else 0.0
        line_col = T_RED if cur_anom else (T_ORNG if ratio >= 0.70 else T_BLUE)
        ax.plot(xs, scores, color=line_col, linewidth=1.85, alpha=0.96,
                solid_capstyle='round', label='ML 복원오차')
        ax.plot(n - 1, scores[-1], 'o', color=line_col, markeredgecolor=T_TEXT,
                markeredgewidth=0.9, markersize=6, zorder=7)

        # 축 범위·레이블 (상단 ML 서브플롯)
        y_top = max(thresh * 2.2, max(scores) * 1.35, thresh + 0.0005)
        ax.set_xlim(0, 300)
        ax.set_ylim(0, y_top)
        ax.set_ylabel("복원 오차", color=T_DIM, fontsize=8)
        ax.set_title(
            f"ML 복원오차 추이  ({n} / 300 프레임)",
            color=T_TEXT, fontsize=10, fontweight='bold', pad=7)
        ax.set_xticklabels([])   # X 레이블 숨김 (하단 공유)
        ax.legend(loc='upper left', fontsize=7, framealpha=0.86,
                  facecolor=T_PANEL, edgecolor=T_BORD, labelcolor=T_TEXT,
                  borderpad=0.45, handlelength=2.2)

        # ── 하단: 규칙 기반 비교 서브플롯 ────────────────────────────
        ax_r = self._rule_ax
        ax_r.cla()
        _style_trend_axis(ax_r)

        rule_all  = list(self._rule_max_history)
        rule_vals = rule_all[-n:] if len(rule_all) >= n else rule_all
        nr        = len(rule_vals)
        x0        = n - nr   # 정렬 오프셋 (ML 기준)
        xr        = list(range(x0, x0 + nr))
        rule_anom = [v > self._rule_thresh for v in rule_vals]

        # 규칙 감지 배경 (주황)
        in_r = False; rs = x0
        for i, a in enumerate(rule_anom):
            xi = x0 + i
            if a and not in_r:
                in_r = True; rs = xi
            elif not a and in_r:
                ax_r.axvspan(rs, xi, color=T_ORNG, alpha=0.13, linewidth=0)
                in_r = False
        if in_r:
            ax_r.axvspan(rs, x0 + nr - 1, color=T_ORNG, alpha=0.13, linewidth=0)

        # ML 감지 배경 (빨강, 상단과 동일)
        ml_slice = is_anom[-nr:]
        in_m = False; ms = x0
        for i, a in enumerate(ml_slice):
            xi = x0 + i
            if a and not in_m:
                in_m = True; ms = xi
            elif not a and in_m:
                ax_r.axvspan(ms, xi, color=T_RED, alpha=0.09, linewidth=0)
                in_m = False
        if in_m:
            ax_r.axvspan(ms, x0 + nr - 1, color=T_RED, alpha=0.09, linewidth=0)

        # 최대 압력 선 + 규칙 임계선
        ax_r.plot(xr, rule_vals, color=T_ORNG, linewidth=1.55, alpha=0.95,
                  solid_capstyle='round', label='최대 압력 (활성채널)')
        ax_r.axhline(self._rule_thresh, color=T_ORNG, linewidth=1.0, linestyle=(0, (6, 4)),
                      alpha=0.85, label=f'규칙 임계 ({self._rule_thresh:.2f})')

        # 감지 방식 비교 마커
        ml_and_r_xs, ml_and_r_ys = [], []
        ml_only_xs,  ml_only_ys  = [], []
        rule_only_xs, rule_only_ys = [], []
        for i, (ml_a, ru_a) in enumerate(zip(ml_slice, rule_anom)):
            xi = x0 + i; yi = rule_vals[i]
            if ml_a and ru_a:
                ml_and_r_xs.append(xi); ml_and_r_ys.append(yi)
            elif ml_a:
                ml_only_xs.append(xi);  ml_only_ys.append(yi)
            elif ru_a:
                rule_only_xs.append(xi); rule_only_ys.append(yi)

        if ml_and_r_xs:
            ax_r.plot(ml_and_r_xs, ml_and_r_ys, '^', color=T_RED,
                      markeredgecolor=T_TEXT, markeredgewidth=0.5,
                      markersize=6, alpha=0.85, zorder=6, label='ML+규칙 동시')
        if ml_only_xs:
            ax_r.plot(ml_only_xs, ml_only_ys, '^', color=T_BLUE,
                      markeredgecolor=T_TEXT, markeredgewidth=0.4,
                      markersize=5, alpha=0.82, zorder=6, label='ML만 감지')
        if rule_only_xs:
            ax_r.plot(rule_only_xs, rule_only_ys, 'o', color=T_ORNG,
                      markeredgecolor=T_TEXT, markeredgewidth=0.4,
                      markersize=5, alpha=0.82, zorder=6, label='규칙만 감지')

        # 스타일
        ax_r.set_xlim(0, 300)
        ax_r.set_ylim(0, 1.15)
        ax_r.set_xlabel("프레임 (최근 300)", color=T_DIM, fontsize=8)
        ax_r.set_ylabel("정규화 압력", color=T_DIM, fontsize=8)
        n_ml_r = len(ml_and_r_xs); n_ml_o = len(ml_only_xs); n_ru_o = len(rule_only_xs)
        ax_r.set_title(
            f"규칙 기반 비교   ML+규칙:{n_ml_r}  ML만:{n_ml_o}  규칙만:{n_ru_o}",
            color=T_TEXT, fontsize=9, fontweight='bold', pad=7)
        ax_r.legend(loc='upper left', fontsize=6, framealpha=0.86,
                    facecolor=T_PANEL, edgecolor=T_BORD, labelcolor=T_TEXT,
                    borderpad=0.45, handlelength=2.0)

        # ── 상단 통계 레이블 동기화 ────────────────────────────────────
        cur_score   = scores[-1]
        rule_now    = rule_vals[-1] if rule_vals else 0.0
        rule_detect = rule_now > self._rule_thresh

        ratio = cur_score / thresh if thresh else 0.0
        if cur_anom:
            score_accent = T_RED
            score_sub = f"{ratio * 100:.0f}% of threshold"
        elif ratio >= 0.70:
            score_accent = T_ORNG
            score_sub = f"주의 구간  {ratio * 100:.0f}%"
        else:
            score_accent = T_GREEN
            score_sub = f"안정 구간  {ratio * 100:.0f}%"

        status_parts = []
        if cur_anom:      status_parts.append("ML ⚠")
        if rule_detect:   status_parts.append("규칙 ⚠")
        if not status_parts: status_parts = ["정상"]

        state_text = " / ".join(status_parts)
        if cur_anom or rule_detect:
            state_accent = T_RED if cur_anom else T_ORNG
            state_sub = f"rule max {rule_now:.2f}"
        else:
            state_accent = T_GREEN
            state_sub = f"rule max {rule_now:.2f}"

        self._set_ml_metric_card(
            self._ml_score_card, f"{cur_score:.5f}", score_sub, score_accent)
        self._set_ml_metric_card(
            self._ml_thresh_card, f"{thresh:.5f}",
            f"{self._ml_sigma_k:.1f}σ sensitivity", T_BLUE)
        self._set_ml_metric_card(
            self._ml_state_card, state_text, state_sub, state_accent)

        self._ml_score_canvas.draw_idle()

    _DEAD_SET = set(range(NUM_CHANNELS)) - set(ML_ACTIVE_CH)
    _BAR_CHARS = " ·▁▄▇█"

    def _update_ml_contrib(self, contrib16, is_anomaly):
        """ERR 행 셀 색상·문자를 채널 기여도에 맞게 갱신."""
        max_e = contrib16.max() if is_anomaly and contrib16.max() > 0 else 1.0
        for i, lbl in enumerate(self._ml_contrib_labels):
            if i in self._DEAD_SET:
                lbl.config(text="─", bg=T_BORD, fg=T_DIM)
            elif not is_anomaly:
                lbl.config(text="·", bg=T_CARD, fg=T_DIM)
            else:
                ratio = float(contrib16[i]) / max_e
                bg    = self._lerp_color(T_CARD, T_RED, ratio)
                bar   = self._BAR_CHARS[int(ratio * (len(self._BAR_CHARS) - 1))]
                fg    = T_TEXT if ratio > 0.5 else T_DIM
                lbl.config(text=bar, bg=bg, fg=fg)

    def _log_ml_anomaly(self, event, score, last_frame_raw):
        """ML 이상 감지/복구 이벤트를 ml_anomaly_log.csv 에 기록."""
        is_new = (not os.path.exists(ML_LOG_PATH)
                  or os.path.getsize(ML_LOG_PATH) == 0)
        # 마지막 프레임에서 가장 압력이 높은(= raw가 낮은) 활성 채널 찾기
        active_raw = last_frame_raw[ML_ACTIVE_CH]
        peak_idx   = int(np.argmin(active_raw))
        peak_ch    = ML_ACTIVE_CH[peak_idx]
        peak_press = int(4095 - active_raw[peak_idx])
        ratio      = score / self._ml_threshold if self._ml_threshold else 0.0
        try:
            with open(ML_LOG_PATH, 'a', newline='', encoding='utf-8-sig') as f:
                w = csv.writer(f)
                if is_new:
                    w.writerow(['datetime', 'event', 'ml_score', 'threshold',
                                'ratio', 'peak_channel', 'peak_pressure'])
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                w.writerow([ts, event,
                            f'{score:.6f}', f'{self._ml_threshold:.6f}',
                            f'{ratio:.3f}', f'ch{peak_ch:02d}', peak_press])
        except OSError:
            pass

    def _open_ml_log(self):
        path = os.path.abspath(ML_LOG_PATH)
        if not os.path.exists(path):
            messagebox.showinfo("ML 이상 로그", "아직 기록된 ML 이상 이력이 없습니다.\n\n"
                                "모델 로드 후 실시간 수신 시 이상이 감지되면 자동 기록됩니다.")
            return
        self._open_path(path)

    def _open_ml_report(self):
        import math, datetime as dt

        if not (os.path.exists(ML_MODEL_PATH) and os.path.exists(ML_STATS_PATH)):
            messagebox.showinfo("모델 정보", "학습된 모델이 없습니다.\n'ML 모델 훈련' 버튼으로 먼저 훈련하세요.")
            return
        if hasattr(self, '_ml_report_win') and self._ml_report_win.winfo_exists():
            self._ml_report_win.lift()
            return

        stats     = np.load(ML_STATS_PATH)
        mean_e    = float(stats['mean_err'])
        std_e     = float(stats['std_err'])
        n_windows = int(stats['n_windows'])   if 'n_windows' in stats else None
        has_fp    = 'fp_rates' in stats and 'sigma_ks' in stats
        sigma_ks_s  = stats['sigma_ks']  if has_fp else None
        fp_rates_s  = stats['fp_rates']  if has_fp else None
        preview_report = self._load_ml_training_preview_report()

        mtime     = os.path.getmtime(ML_MODEL_PATH)
        mtime_str = dt.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')
        fsize_kb  = os.path.getsize(ML_MODEL_PATH) / 1024

        popup = tk.Toplevel(self.root)
        popup.title("ML 모델 성능 리포트")
        popup.configure(bg=T_BG)
        popup.resizable(False, False)
        self._ml_report_win = popup

        # ─── 헬퍼 ────────────────────────────────────────────────────
        def _section(title):
            f = tk.Frame(popup, bg=T_BG)
            f.pack(fill=tk.X, padx=20, pady=(12, 3))
            tk.Label(f, text=title, bg=T_BG, fg=T_BLUE,
                     font=("Consolas", 8, "bold")).pack(side=tk.LEFT)
            sep = tk.Frame(f, bg=T_BORD, height=1)
            sep.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        def _kv(parent, key, val, row, bold_val=False):
            vfont = ("Consolas", 8, "bold") if bold_val else ("Consolas", 8)
            tk.Label(parent, text=key, bg=T_PANEL, fg=T_DIM,
                     font=("Consolas", 8), width=22, anchor='w'
                     ).grid(row=row, column=0, padx=(10, 4), pady=2, sticky='w')
            tk.Label(parent, text=val, bg=T_PANEL, fg=T_TEXT,
                     font=vfont, anchor='w'
                     ).grid(row=row, column=1, padx=(0, 10), pady=2, sticky='w')

        # ─── 제목 ────────────────────────────────────────────────────
        tk.Label(popup, text="ML 모델 성능 리포트", bg=T_BG, fg=T_TEXT,
                 font=("맑은 고딕", 12, "bold")).pack(pady=(16, 4), padx=20)

        # ─── 파일 정보 ───────────────────────────────────────────────
        _section("파일 정보")
        fi = tk.Frame(popup, bg=T_PANEL, highlightbackground=T_BORD, highlightthickness=1)
        fi.pack(fill=tk.X, padx=20, pady=(0, 2))
        _kv(fi, "모델 파일", f"anomaly_model.pt  ({fsize_kb:.1f} KB)", 0)
        _kv(fi, "최종 수정", mtime_str, 1)
        _kv(fi, "통계 파일", "anomaly_stats.npz", 2)

        # ─── 훈련 통계 ───────────────────────────────────────────────
        _section("훈련 통계")
        si = tk.Frame(popup, bg=T_PANEL, highlightbackground=T_BORD, highlightthickness=1)
        si.pack(fill=tk.X, padx=20, pady=(0, 2))
        dead = sorted(set(range(16)) - set(ML_ACTIVE_CH))
        dead_str = "  ".join(f"ch{c:02d}" for c in dead)
        _kv(si, "활성 채널", f"{len(ML_ACTIVE_CH)}ch  (사망: {dead_str})", 0)
        _kv(si, "윈도우 크기", f"{ML_SEQ_LEN} 프레임  ({ML_SEQ_LEN * 10} ms)", 1)
        r = 2
        if n_windows is not None:
            _kv(si, "훈련 윈도우", f"{n_windows:,} 개", r); r += 1
            _kv(si, "훈련 프레임 (추정)", f"{n_windows + ML_SEQ_LEN:,} 개", r); r += 1
        _kv(si, "복원오차 평균", f"{mean_e:.6f}", r);     r += 1
        _kv(si, "복원오차 표준편차", f"{std_e:.6f}", r);  r += 1
        _kv(si, "기본 임계값 (3σ)", f"{mean_e + 3 * std_e:.6f}", r, bold_val=True)

        # ─── 훈련 CSV 자동 검사 ────────────────────────────────────
        if preview_report:
            check_channels = preview_report.get("check_channels", [])
            flat_channels = preview_report.get("flat_channels", [])
            channel_rows = preview_report.get("channels", [])
            flagged_rows = [r for r in channel_rows if r.get("flag")]

            def _ch_list(values):
                if not values:
                    return "없음"
                return " ".join(f"ch{int(v):02d}" for v in values[:8]) + (
                    f" 외 {len(values) - 8}개" if len(values) > 8 else "")

            _section("훈련 CSV 자동 검사")
            pi = tk.Frame(popup, bg=T_PANEL, highlightbackground=T_BORD, highlightthickness=1)
            pi.pack(fill=tk.X, padx=20, pady=(0, 2))
            _kv(pi, "CSV 파일", preview_report.get("source_file", "unknown"), 0)
            _kv(pi, "프레임 / 시간", f"{int(preview_report.get('frames', 0)):,}개  "
                                    f"{float(preview_report.get('duration_ms', 0.0)) / 1000:.2f}s", 1)
            _kv(pi, "원본 범위", f"{float(preview_report.get('raw_min', 0.0)):.0f} ~ "
                              f"{float(preview_report.get('raw_max', 0.0)):.0f}", 2)
            _kv(pi, "CHECK 채널", _ch_list(check_channels), 3, bold_val=bool(check_channels))
            _kv(pi, "FLAT 채널", _ch_list(flat_channels), 4, bold_val=bool(flat_channels))

            if flagged_rows:
                detail = "   ".join(
                    f"ch{int(s['ch']):02d}:{s.get('flag', '')}/{s.get('reason', '')}"
                    for s in flagged_rows[:6])
                if len(flagged_rows) > 6:
                    detail += f" 외 {len(flagged_rows) - 6}개"
            else:
                detail = "튀는 채널 없음"
            _kv(pi, "판정 요약", detail, 5, bold_val=bool(flagged_rows))

        # ─── σ 배수별 임계값 & 탐지율 ───────────────────────────────
        _section("σ 배수별 임계값 & 탐지율")

        tbl = tk.Frame(popup, bg=T_PANEL, highlightbackground=T_BORD, highlightthickness=1)
        tbl.pack(fill=tk.X, padx=20, pady=(0, 2))

        HFONT = ("Consolas", 7, "bold")
        CFONT = ("Consolas", 8)
        hdrs  = ["σ 배수", "임계값", "이론 확률", "훈련 실측"]
        widths = [8, 12, 11, 11]
        for c, (h, w) in enumerate(zip(hdrs, widths)):
            tk.Label(tbl, text=h, bg=T_CARD, fg=T_DIM,
                     font=HFONT, width=w, anchor='center'
                     ).grid(row=0, column=c, padx=1, pady=(6, 2), sticky='ew')

        ks = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        for ri, k in enumerate(ks):
            thresh_k = mean_e + k * std_e
            theory_p = 0.5 * math.erfc(k / math.sqrt(2)) * 100

            is_cur  = abs(k - self._ml_sigma_k) < 0.01
            is_def  = abs(k - 3.0) < 0.01
            row_bg  = T_CARD  if is_cur else T_PANEL
            lbl_col = T_BLUE  if is_cur else (T_ORNG if is_def else T_DIM)
            val_col = T_TEXT  if is_cur else (T_TEXT if is_def else T_DIM)
            vfont   = (CFONT[0], CFONT[1], "bold") if (is_cur or is_def) else CFONT

            if has_fp and sigma_ks_s is not None:
                idx      = int(np.argmin(np.abs(sigma_ks_s - k)))
                actual_p = float(fp_rates_s[idx]) * 100
                actual_s = f"{actual_p:.3f}%"
            else:
                actual_s = "─"

            row_vals = [f"{k:.1f} σ", f"{thresh_k:.6f}",
                        f"{theory_p:.3f}%", actual_s]
            for c, (v, w) in enumerate(zip(row_vals, widths)):
                fg = lbl_col if c == 0 else val_col
                tk.Label(tbl, text=v, bg=row_bg, fg=fg,
                         font=vfont, width=w, anchor='center'
                         ).grid(row=ri + 1, column=c, padx=1, pady=1, sticky='ew')

        # ─── 주석 ────────────────────────────────────────────────────
        notes = ["● 현재 선택 σ → 파란색  /  기본값 3σ → 주황색"]
        if not has_fp:
            notes.append("● 훈련 실측값: 재훈련 후 표시됩니다")
        notes.append("● 이론 확률: 오차가 정규분포를 따른다고 가정한 이상 프레임 비율")
        tk.Label(popup, text="\n".join(notes), bg=T_BG, fg=T_DIM,
                 font=("Consolas", 6), justify="left").pack(anchor='w', padx=22, pady=(6, 0))

        # ─── 클립보드 복사 + 닫기 ────────────────────────────────────
        def _copy():
            lines = [
                "ML 모델 성능 리포트",
                f"  모델 파일   : anomaly_model.pt ({fsize_kb:.1f} KB)",
                f"  최종 수정   : {mtime_str}",
                f"  활성 채널   : {len(ML_ACTIVE_CH)}ch",
                f"  윈도우 크기 : {ML_SEQ_LEN} 프레임",
            ]
            if n_windows is not None:
                lines.append(f"  훈련 윈도우 : {n_windows:,}개")
            lines += [
                f"  평균 오차   : {mean_e:.6f}",
                f"  표준편차    : {std_e:.6f}",
            ]
            if preview_report:
                check_channels = preview_report.get("check_channels", [])
                flat_channels = preview_report.get("flat_channels", [])

                def _copy_ch_list(values):
                    return "없음" if not values else " ".join(f"ch{int(v):02d}" for v in values)

                lines += [
                    "",
                    "훈련 CSV 자동 검사",
                    f"  CSV 파일    : {preview_report.get('source_file', 'unknown')}",
                    f"  프레임      : {int(preview_report.get('frames', 0)):,}개",
                    f"  시간        : {float(preview_report.get('duration_ms', 0.0)) / 1000:.2f}s",
                    f"  원본 범위   : {float(preview_report.get('raw_min', 0.0)):.0f} ~ "
                    f"{float(preview_report.get('raw_max', 0.0)):.0f}",
                    f"  CHECK 채널  : {_copy_ch_list(check_channels)}",
                    f"  FLAT 채널   : {_copy_ch_list(flat_channels)}",
                ]
            lines += [
                "",
                f"{'σ':>6}  {'임계값':>10}  {'이론확률':>10}  {'훈련실측':>10}",
            ]
            for k in ks:
                thresh_k = mean_e + k * std_e
                tp = 0.5 * math.erfc(k / math.sqrt(2)) * 100
                if has_fp and sigma_ks_s is not None:
                    idx = int(np.argmin(np.abs(sigma_ks_s - k)))
                    ap  = f"{float(fp_rates_s[idx])*100:.3f}%"
                else:
                    ap = "─"
                lines.append(f"{k:>5.1f}σ  {thresh_k:>10.6f}  {tp:>9.3f}%  {ap:>10}")
            popup.clipboard_clear()
            popup.clipboard_append("\n".join(lines))
            copy_btn.config(text="복사됨!", fg=T_GREEN)
            popup.after(1500, lambda: copy_btn.config(text="클립보드 복사", fg=T_DIM))

        def _open_preview_json():
            if not os.path.exists(ML_PREVIEW_REPORT_PATH):
                messagebox.showinfo(
                    "CSV 검사 JSON",
                    "아직 저장된 CSV 검사 리포트가 없습니다.\n\n"
                    "ML 모델 훈련에서 CSV를 선택하면 자동 생성됩니다.",
                    parent=popup
                )
                return
            self._open_path(ML_PREVIEW_REPORT_PATH)

        def _open_preview_html():
            if not os.path.exists(ML_PREVIEW_HTML_PATH):
                messagebox.showinfo(
                    "CSV 검사 HTML",
                    "아직 저장된 HTML 리포트가 없습니다.\n\n"
                    "ML 모델 훈련에서 CSV를 선택하면 자동 생성됩니다.",
                    parent=popup
                )
                return
            self._open_path(ML_PREVIEW_HTML_PATH)

        btn_row = tk.Frame(popup, bg=T_BG)
        btn_row.pack(fill=tk.X, padx=20, pady=(10, 16))
        copy_btn = tk.Button(btn_row, text="클립보드 복사", font=("맑은 고딕", 8),
                             bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                             relief=tk.FLAT, pady=4, command=_copy)
        copy_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        html_btn = tk.Button(
            btn_row, text="HTML 리포트 열기", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_BLUE if os.path.exists(ML_PREVIEW_HTML_PATH) else T_DIM,
            activebackground=T_BORD, relief=tk.FLAT, pady=4,
            state=tk.NORMAL if os.path.exists(ML_PREVIEW_HTML_PATH) else tk.DISABLED,
            command=_open_preview_html
        )
        html_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        json_btn = tk.Button(
            btn_row, text="JSON 열기", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_BLUE if os.path.exists(ML_PREVIEW_REPORT_PATH) else T_DIM,
            activebackground=T_BORD, relief=tk.FLAT, pady=4,
            state=tk.NORMAL if os.path.exists(ML_PREVIEW_REPORT_PATH) else tk.DISABLED,
            command=_open_preview_json
        )
        json_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(btn_row, text="닫기", font=("맑은 고딕", 8),
                  bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                  relief=tk.FLAT, pady=4, command=popup.destroy
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        popup.update_idletasks()
        px = self.root.winfo_x() + (self.root.winfo_width()  - popup.winfo_width())  // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{px}+{py}")

    def _open_ml_train_progress(self, filepath):
        if hasattr(self, "_ml_train_win") and self._ml_train_win.winfo_exists():
            self._ml_train_win.destroy()

        popup = tk.Toplevel(self.root)
        popup.title("ML 모델 훈련")
        popup.configure(bg=T_BG)
        popup.geometry("680x440")
        popup.minsize(560, 360)
        self._ml_train_win = popup

        tk.Label(
            popup, text="ML 모델 훈련 중", bg=T_BG, fg=T_TEXT,
            font=("맑은 고딕", 13, "bold")
        ).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(
            popup, text=os.path.basename(filepath), bg=T_BG, fg=T_DIM,
            font=("Consolas", 9)
        ).pack(anchor="w", padx=16, pady=(0, 8))

        progress_frame = tk.Frame(popup, bg=T_BG)
        progress_frame.pack(fill=tk.X, padx=16, pady=(0, 8))
        self._ml_train_progress_canvas = tk.Canvas(
            progress_frame, bg=T_BORD, height=14, highlightthickness=0
        )
        self._ml_train_progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._ml_train_progress_bar = self._ml_train_progress_canvas.create_rectangle(
            0, 0, 0, 14, fill=T_BLUE, outline=""
        )
        self._ml_train_progress_label = tk.Label(
            progress_frame, text="0%", bg=T_BG, fg=T_DIM,
            font=("Consolas", 10, "bold"), width=5
        )
        self._ml_train_progress_label.pack(side=tk.LEFT, padx=(8, 0))

        self._ml_train_log = tk.Text(
            popup, bg=T_PANEL, fg=T_TEXT, insertbackground=T_TEXT,
            relief=tk.FLAT, height=14, font=("Consolas", 9),
            state=tk.DISABLED
        )
        self._ml_train_log.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 10))

        self._ml_train_close_btn = tk.Button(
            popup, text="훈련 중...", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=6, state=tk.DISABLED,
            command=popup.destroy
        )
        self._ml_train_close_btn.pack(fill=tk.X, padx=16, pady=(0, 14))
        self._set_ml_train_progress(0)
        popup.after(100, lambda: self._set_ml_train_progress(0))
        self._append_ml_train_log("[UI] CSV 선택 완료")
        self._append_ml_train_log("[UI] 훈련을 시작합니다. 완료까지 잠시 기다리세요.")

    def _append_ml_train_log(self, msg):
        if not hasattr(self, "_ml_train_log"):
            return
        try:
            if not self._ml_train_log.winfo_exists():
                return
            self._update_ml_train_progress_from_log(str(msg))
            self._ml_train_log.config(state=tk.NORMAL)
            self._ml_train_log.insert(tk.END, str(msg) + "\n")
            self._ml_train_log.see(tk.END)
            self._ml_train_log.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def _set_ml_train_progress(self, percent):
        percent = max(0, min(100, int(percent)))
        if not hasattr(self, "_ml_train_progress_canvas"):
            return
        try:
            w = max(1, self._ml_train_progress_canvas.winfo_width())
            self._ml_train_progress_canvas.coords(
                self._ml_train_progress_bar, 0, 0, int(w * percent / 100), 14)
            color = T_GREEN if percent >= 100 else T_BLUE
            self._ml_train_progress_canvas.itemconfig(
                self._ml_train_progress_bar, fill=color)
            self._ml_train_progress_label.config(
                text=f"{percent}%", fg=color if percent >= 100 else T_DIM)
        except tk.TclError:
            pass

    def _update_ml_train_progress_from_log(self, msg):
        if "Epoch" not in msg:
            return
        try:
            part = msg.split("Epoch", 1)[1].strip().split()[0]
            cur_s, total_s = part.split("/", 1)
            cur = int(cur_s)
            total = int(total_s)
            if total > 0:
                self._set_ml_train_progress(round(cur / total * 100))
        except (IndexError, ValueError):
            pass

    def _queue_ml_train_log(self, msg):
        self.root.after(0, lambda msg=str(msg): self._append_ml_train_log(msg))

    def _on_ml_train_failed(self, err):
        self._ml_label.config(text=f"ML: 훈련 실패 - {err}", fg=T_RED)
        if hasattr(self, "_ml_train_progress_canvas"):
            self._ml_train_progress_canvas.itemconfig(
                self._ml_train_progress_bar, fill=T_RED)
            self._ml_train_progress_label.config(text="ERR", fg=T_RED)
        self._append_ml_train_log("")
        self._append_ml_train_log("[ERROR] " + str(err))
        if hasattr(self, "_ml_train_close_btn"):
            self._ml_train_close_btn.config(text="닫기", state=tk.NORMAL, fg=T_RED)
        messagebox.showerror("ML 모델 훈련 실패", str(err))

    def _train_ml_model(self):
        if not _ensure_torch():
            detail = _TORCH_ERROR or "현재 실행 환경에서 torch를 불러오지 못했습니다."
            messagebox.showwarning(
                "PyTorch 로드 실패",
                "ML 모델 훈련에 필요한 PyTorch를 불러오지 못했습니다.\n\n"
                f"실행 파일: {sys.executable}\n\n"
                f"원인: {detail}\n\n"
                "venv로 실행 중이면: venv\\Scripts\\python.exe -m pip install torch\n"
                "배포본이면: build.bat full 로 다시 빌드하세요."
            )
            return
        filepath = filedialog.askopenfilename(
            title="훈련용 CSV 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")])
        if not filepath:
            return
        try:
            with open(filepath, newline="", encoding="utf-8-sig") as f:
                preview_rows = list(csv.DictReader(f))
            preview_xs, preview_channels = self._csv_rows_to_channel_arrays(preview_rows)
            preview_stats = self._analyze_csv_channel_quality(preview_channels)
        except Exception as e:
            messagebox.showerror(
                "훈련용 CSV 오류",
                "ML 훈련 전에 CSV를 확인할 수 없습니다.\n\n"
                "ch0~ch15 컬럼과 데이터 행을 확인하세요.\n\n"
                f"원인: {e}"
            )
            return
        self._save_ml_training_preview_report(filepath, preview_xs, preview_channels, preview_stats)
        self._open_csv_channel_graph(filepath, preview_rows)
        self._ml_label.config(text="ML: 훈련 중...", fg=T_ORNG)
        self._open_ml_train_progress(filepath)
        self.root.update()

        def run():
            try:
                from ml_anomaly import train as ml_train
                ml_train(filepath, save_dir=APP_DIR, print_fn=self._queue_ml_train_log)
                self.root.after(0, self._on_ml_train_done)
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda err=err: self._on_ml_train_failed(err))

        threading.Thread(target=run, daemon=True).start()

    def _on_ml_train_done(self):
        self._ml_model = None
        self._ml_buffer.clear()
        self._load_ml_model()
        self._append_ml_train_log("")
        self._append_ml_train_log("[DONE] ML 모델 훈련 완료")
        self._set_ml_train_progress(100)
        if hasattr(self, "_ml_train_close_btn"):
            self._ml_train_close_btn.config(text="닫기", state=tk.NORMAL, fg=T_GREEN)
        self.root.after(250, self._open_ml_report)
        messagebox.showinfo("훈련 완료",
            "ML 모델 훈련 완료.\n\n"
            "이제 실시간 수신 중 STATUS 섹션의\n"
            "ML 점수 바로 이상 수준을 확인할 수 있습니다.")

    # ---------------- 재생(Playback) 로직 ----------------
    def _csv_rows_to_channel_arrays(self, rows):
        xs = []
        channels = [[] for _ in range(NUM_CHANNELS)]
        for row_idx, row in enumerate(rows):
            try:
                values = [float(row[f"ch{i}"]) for i in range(NUM_CHANNELS)]
            except (KeyError, ValueError):
                continue

            x_val = None
            for key in ("elapsed_ms", "timestamp_ms", "raw_timestamp_ms", "index"):
                try:
                    if row.get(key, "") != "":
                        x_val = float(row[key])
                        break
                except (TypeError, ValueError):
                    pass
            if x_val is None:
                x_val = row_idx * 10.0

            xs.append(x_val)
            for ch, value in enumerate(values):
                channels[ch].append(value)

        if not xs:
            raise ValueError("ch0~ch15 columns were not found.")
        return np.array(xs, dtype=float), [np.array(v, dtype=float) for v in channels]

    def _analyze_csv_channel_quality(self, channels):
        stats = []
        peaks = []
        ranges = []
        stds = []
        for ch, arr in enumerate(channels):
            min_val = float(arr.min())
            max_val = float(arr.max())
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            range_val = max_val - min_val
            peak_val = float(SCALE_MAX - min_val)
            item = {
                "ch": ch,
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "range": range_val,
                "peak": peak_val,
                "flag": "",
                "reason": "",
            }
            stats.append(item)
            peaks.append(peak_val)
            ranges.append(range_val)
            stds.append(std_val)

        def robust_z(values):
            arr = np.array(values, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            scale = (1.4826 * mad) if mad > 1e-9 else float(arr.std())
            if scale <= 1e-9:
                return np.zeros_like(arr)
            return (arr - med) / scale

        peak_z = robust_z(peaks)
        range_z = robust_z(ranges)
        std_z = robust_z(stds)

        for i, item in enumerate(stats):
            if item["range"] <= 5.0 and item["std"] <= 2.0:
                item["flag"] = "FLAT"
                item["reason"] = "no movement"
                continue

            reasons = []
            if peak_z[i] >= 2.5 and item["range"] > 50.0:
                reasons.append("peak")
            if range_z[i] >= 2.5:
                reasons.append("range")
            if std_z[i] >= 2.5:
                reasons.append("noise")

            if reasons:
                item["flag"] = "CHECK"
                item["reason"] = "/".join(reasons)

        return stats

    def _save_ml_training_preview_report(self, filepath, xs, channels, channel_stats):
        flagged = [s for s in channel_stats if s.get("flag")]
        check = [s for s in flagged if s.get("flag") == "CHECK"]
        flat = [s for s in flagged if s.get("flag") == "FLAT"]
        duration_ms = float(xs[-1] - xs[0]) if len(xs) > 1 else 0.0
        all_values = np.concatenate(channels)

        report = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_file": os.path.basename(filepath),
            "source_path": os.path.abspath(filepath),
            "frames": int(len(xs)),
            "duration_ms": duration_ms,
            "raw_min": float(all_values.min()),
            "raw_max": float(all_values.max()),
            "check_channels": [int(s["ch"]) for s in check],
            "flat_channels": [int(s["ch"]) for s in flat],
            "channels": [
                {
                    "ch": int(s["ch"]),
                    "mean": round(float(s["mean"]), 3),
                    "std": round(float(s["std"]), 3),
                    "min": round(float(s["min"]), 3),
                    "max": round(float(s["max"]), 3),
                    "range": round(float(s["range"]), 3),
                    "peak": round(float(s["peak"]), 3),
                    "flag": s.get("flag", ""),
                    "reason": s.get("reason", ""),
                }
                for s in channel_stats
            ],
        }

        try:
            with open(ML_PREVIEW_REPORT_PATH, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
        self._save_ml_training_preview_html(report)

    def _save_ml_training_preview_html(self, report):
        import html

        def esc(value):
            return html.escape(str(value), quote=True)

        def ch_list(values):
            if not values:
                return '<span class="muted">없음</span>'
            return " ".join(f'<span class="chip">ch{int(v):02d}</span>' for v in values)

        channel_rows = []
        for item in report.get("channels", []):
            flag = item.get("flag", "")
            flag_class = "ok"
            flag_text = "OK"
            if flag == "CHECK":
                flag_class = "check"
                flag_text = "CHECK"
            elif flag == "FLAT":
                flag_class = "flat"
                flag_text = "FLAT"
            channel_rows.append(
                "<tr>"
                f"<td>ch{int(item.get('ch', 0)):02d}</td>"
                f"<td><span class=\"badge {flag_class}\">{flag_text}</span></td>"
                f"<td>{esc(item.get('reason', ''))}</td>"
                f"<td>{float(item.get('mean', 0.0)):.1f}</td>"
                f"<td>{float(item.get('std', 0.0)):.1f}</td>"
                f"<td>{float(item.get('min', 0.0)):.0f}</td>"
                f"<td>{float(item.get('max', 0.0)):.0f}</td>"
                f"<td>{float(item.get('range', 0.0)):.0f}</td>"
                f"<td>{float(item.get('peak', 0.0)):.0f}</td>"
                "</tr>"
            )

        check_count = len(report.get("check_channels", []))
        flat_count = len(report.get("flat_channels", []))
        status_class = "good" if check_count == 0 else "bad"
        status_text = "훈련 가능" if check_count == 0 else "확인 필요"

        doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>ML Training CSV Report</title>
  <style>
    body {{ margin: 0; background: #f3f6fb; color: #172033; font-family: 'Malgun Gothic', Arial, sans-serif; }}
    .page {{ max-width: 1120px; margin: 0 auto; padding: 28px; }}
    .top {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-start; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ color: #5f6f82; font-size: 13px; }}
    .status {{ padding: 10px 14px; border-radius: 6px; font-weight: 700; }}
    .status.good {{ background: #e7f7ee; color: #1f8f4d; }}
    .status.bad {{ background: #fdecec; color: #d1242f; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #c8d2df; border-radius: 6px; padding: 12px; }}
    .label {{ color: #5f6f82; font-size: 12px; margin-bottom: 4px; }}
    .value {{ font-size: 18px; font-weight: 700; }}
    .section {{ margin-top: 18px; }}
    h2 {{ font-size: 16px; margin: 0 0 8px; }}
    .chips {{ background: #fff; border: 1px solid #c8d2df; border-radius: 6px; padding: 12px; line-height: 2.1; }}
    .chip {{ display: inline-block; padding: 2px 8px; margin: 2px; background: #eef3f8; border-radius: 999px; font-family: Consolas, monospace; }}
    .muted {{ color: #5f6f82; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #c8d2df; border-radius: 6px; overflow: hidden; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5ebf1; text-align: right; font-family: Consolas, 'Malgun Gothic', monospace; font-size: 13px; }}
    th {{ background: #e8eef6; color: #5f6f82; font-weight: 700; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    .badge {{ display: inline-block; min-width: 54px; padding: 2px 8px; border-radius: 999px; color: #fff; font-weight: 700; text-align: center; }}
    .badge.ok {{ background: #1f8f4d; }}
    .badge.check {{ background: #d1242f; }}
    .badge.flat {{ background: #b66a00; }}
  </style>
</head>
<body>
  <main class="page">
    <div class="top">
      <div>
        <h1>ML Training CSV Report</h1>
        <div class="sub">{esc(report.get('source_file', 'unknown'))}</div>
        <div class="sub">created at {esc(report.get('created_at', ''))}</div>
      </div>
      <div class="status {status_class}">{status_text}</div>
    </div>

    <div class="grid">
      <div class="card"><div class="label">Frames</div><div class="value">{int(report.get('frames', 0)):,}</div></div>
      <div class="card"><div class="label">Duration</div><div class="value">{float(report.get('duration_ms', 0.0)) / 1000:.2f}s</div></div>
      <div class="card"><div class="label">Raw Range</div><div class="value">{float(report.get('raw_min', 0.0)):.0f}~{float(report.get('raw_max', 0.0)):.0f}</div></div>
      <div class="card"><div class="label">Flags</div><div class="value">CHECK {check_count} / FLAT {flat_count}</div></div>
    </div>

    <section class="section">
      <h2>CHECK Channels</h2>
      <div class="chips">{ch_list(report.get('check_channels', []))}</div>
    </section>

    <section class="section">
      <h2>FLAT Channels</h2>
      <div class="chips">{ch_list(report.get('flat_channels', []))}</div>
    </section>

    <section class="section">
      <h2>Channel Detail</h2>
      <table>
        <thead>
          <tr><th>Channel</th><th>Status</th><th>Reason</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Range</th><th>Peak</th></tr>
        </thead>
        <tbody>
          {''.join(channel_rows)}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
        try:
            with open(ML_PREVIEW_HTML_PATH, "w", encoding="utf-8") as f:
                f.write(doc)
        except OSError:
            pass

    def _load_ml_training_preview_report(self):
        try:
            with open(ML_PREVIEW_REPORT_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return None

    def _open_csv_channel_graph(self, filepath, rows):
        try:
            xs, channels = self._csv_rows_to_channel_arrays(rows)
        except Exception as e:
            messagebox.showwarning("CSV graph", f"Cannot draw 16 channel graphs.\n{e}")
            return

        if hasattr(self, "_csv_graph_win") and self._csv_graph_win.winfo_exists():
            self._csv_graph_win.destroy()

        popup = tk.Toplevel(self.root)
        popup.title("CSV 16 Channel Graphs")
        popup.configure(bg=T_BG)
        popup.geometry("1180x760")
        popup.minsize(980, 620)
        self._csv_graph_win = popup

        header = tk.Frame(popup, bg=T_BG)
        header.pack(fill=tk.X, padx=14, pady=(12, 4))

        name = os.path.basename(filepath)
        duration_ms = float(xs[-1] - xs[0]) if len(xs) > 1 else 0.0
        all_values = np.concatenate(channels)
        peak_ch = int(np.argmin([arr.min() for arr in channels]))
        peak_pressure = int(SCALE_MAX - channels[peak_ch].min())
        channel_stats = self._analyze_csv_channel_quality(channels)
        flagged_stats = [s for s in channel_stats if s["flag"]]
        check_channels = [s for s in flagged_stats if s["flag"] == "CHECK"]
        flat_channels = [s for s in flagged_stats if s["flag"] == "FLAT"]

        tk.Label(
            header, text=name, bg=T_BG, fg=T_TEXT,
            font=("맑은 고딕", 12, "bold")
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            text=(f"{len(xs):,} frames   {duration_ms / 1000:.2f}s   "
                  f"raw {all_values.min():.0f}~{all_values.max():.0f}   "
                  f"peak ch{peak_ch:02d} pressure {peak_pressure}"),
            bg=T_BG, fg=T_DIM, font=("Consolas", 9)
        ).pack(side=tk.RIGHT)

        flag_text = "자동 검사: 튀는 채널 없음"
        flag_fg = T_GREEN
        if flagged_stats:
            parts = []
            if check_channels:
                parts.append("CHECK " + ", ".join(f"ch{s['ch']:02d}({s['reason']})" for s in check_channels[:6]))
                if len(check_channels) > 6:
                    parts[-1] += f" 외 {len(check_channels) - 6}개"
            if flat_channels:
                parts.append("FLAT " + ", ".join(f"ch{s['ch']:02d}" for s in flat_channels[:6]))
                if len(flat_channels) > 6:
                    parts[-1] += f" 외 {len(flat_channels) - 6}개"
            flag_text = "자동 검사: " + "   |   ".join(parts)
            flag_fg = T_RED if check_channels else T_ORNG

        tk.Label(
            popup, text=flag_text, bg=T_BG, fg=flag_fg,
            font=("맑은 고딕", 9, "bold"), anchor="w", justify="left",
            wraplength=1120
        ).pack(fill=tk.X, padx=14, pady=(0, 6))
        tk.Label(
            popup, text="그래프를 클릭하면 해당 채널을 크게 볼 수 있습니다.",
            bg=T_BG, fg=T_DIM, font=("맑은 고딕", 8), anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(0, 4))

        fig = Figure(figsize=(11.6, 6.8), dpi=100, facecolor=T_FIG)
        plot_bg = "#101820" if self._is_dark else "#ffffff"
        grid_col = "#2f3b45" if self._is_dark else "#d8dee4"

        x_plot = xs - xs[0]
        x_label = "elapsed (ms)"
        if len(x_plot) > 1 and x_plot[-1] >= 5000:
            x_plot = x_plot / 1000.0
            x_label = "elapsed (s)"

        axis_to_channel = {}
        for ch in range(NUM_CHANNELS):
            ax = fig.add_subplot(4, 4, ch + 1)
            axis_to_channel[ax] = ch
            arr = channels[ch]
            stat = channel_stats[ch]
            mean_val = float(arr.mean())
            min_val = float(arr.min())
            max_val = float(arr.max())
            pressure_peak = int(SCALE_MAX - min_val)
            flag = stat["flag"]
            line_color = T_BLUE
            spine_color = T_BORD
            title_color = T_TEXT
            if flag == "CHECK":
                line_color = T_RED
                spine_color = T_RED
                title_color = T_RED
            elif flag == "FLAT":
                line_color = T_ORNG
                spine_color = T_ORNG
                title_color = T_ORNG

            ax.set_facecolor(plot_bg)
            ax.grid(True, color=grid_col, linewidth=0.5, alpha=0.45)
            ax.plot(x_plot, arr, color=line_color, linewidth=1.05 if flag else 0.9)
            ax.axhline(mean_val, color=T_RED, linewidth=0.85, linestyle="--", alpha=0.9)
            ax.set_title(
                f"ch{ch:02d}  avg {mean_val:.0f}  peak {pressure_peak}",
                color=title_color, fontsize=7.2, pad=3)
            ax.tick_params(colors=T_DIM, labelsize=5.5, length=0)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
                spine.set_linewidth(1.6 if flag else 0.8)

            if flag:
                ax.text(
                    0.98, 0.92, flag, transform=ax.transAxes,
                    ha="right", va="top", color="#ffffff",
                    fontsize=6.0, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor=T_RED if flag == "CHECK" else T_ORNG,
                              edgecolor="none", alpha=0.92)
                )

            margin = max(60.0, (max_val - min_val) * 0.18)
            ax.set_ylim(max(SCALE_MIN, min_val - margin),
                        min(SCALE_MAX, max_val + margin))
            if ch < 12:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x_label, color=T_DIM, fontsize=6)

        fig.tight_layout(pad=1.1, h_pad=1.0, w_pad=0.6)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        canvas.draw()
        canvas.mpl_connect(
            "button_press_event",
            lambda event: self._on_csv_graph_click(
                event, axis_to_channel, name, x_plot, x_label, channels, channel_stats)
        )

        tk.Button(
            popup, text="Close", font=("맑은 고딕", 9),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, command=popup.destroy
        ).pack(fill=tk.X, padx=14, pady=(0, 12))

    def _on_csv_graph_click(self, event, axis_to_channel, source_name,
                            x_plot, x_label, channels, channel_stats):
        if event.inaxes not in axis_to_channel:
            return
        ch = axis_to_channel[event.inaxes]
        self._open_csv_single_channel_graph(
            source_name, ch, x_plot, x_label, channels[ch], channel_stats[ch])

    def _open_csv_single_channel_graph(self, source_name, ch, x_plot, x_label, arr, stat):
        if hasattr(self, "_csv_single_graph_win") and self._csv_single_graph_win.winfo_exists():
            self._csv_single_graph_win.destroy()

        popup = tk.Toplevel(self.root)
        popup.title(f"CSV Channel Detail - ch{ch:02d}")
        popup.configure(bg=T_BG)
        popup.geometry("980x620")
        popup.minsize(760, 500)
        self._csv_single_graph_win = popup

        flag = stat.get("flag", "")
        reason = stat.get("reason", "")
        flag_fg = T_RED if flag == "CHECK" else T_ORNG if flag == "FLAT" else T_GREEN
        status_text = "정상"
        if flag:
            status_text = f"{flag} ({reason})" if reason else flag

        header = tk.Frame(popup, bg=T_BG)
        header.pack(fill=tk.X, padx=16, pady=(14, 6))
        tk.Label(
            header, text=f"ch{ch:02d} 상세 그래프", bg=T_BG, fg=T_TEXT,
            font=("맑은 고딕", 14, "bold")
        ).pack(side=tk.LEFT)
        tk.Label(
            header, text=status_text, bg=T_BG, fg=flag_fg,
            font=("맑은 고딕", 12, "bold")
        ).pack(side=tk.RIGHT)

        info = (
            f"{source_name}   |   "
            f"avg {stat['mean']:.1f}   min {stat['min']:.0f}   max {stat['max']:.0f}   "
            f"range {stat['range']:.0f}   std {stat['std']:.1f}   "
            f"pressure peak {stat['peak']:.0f}"
        )
        tk.Label(
            popup, text=info, bg=T_BG, fg=T_DIM,
            font=("Consolas", 9), anchor="w", wraplength=940
        ).pack(fill=tk.X, padx=16, pady=(0, 8))

        fig = Figure(figsize=(9.4, 4.9), dpi=100, facecolor=T_FIG)
        ax_raw = fig.add_subplot(2, 1, 1)
        ax_pressure = fig.add_subplot(2, 1, 2, sharex=ax_raw)
        plot_bg = "#101820" if self._is_dark else "#ffffff"
        grid_col = "#2f3b45" if self._is_dark else "#d8dee4"
        line_color = T_RED if flag == "CHECK" else T_ORNG if flag == "FLAT" else T_BLUE

        for ax in (ax_raw, ax_pressure):
            ax.set_facecolor(plot_bg)
            ax.grid(True, color=grid_col, linewidth=0.55, alpha=0.5)
            ax.tick_params(colors=T_DIM, labelsize=8, length=0)
            for spine in ax.spines.values():
                spine.set_color(flag_fg if flag else T_BORD)
                spine.set_linewidth(1.4 if flag else 0.8)

        ax_raw.plot(x_plot, arr, color=line_color, linewidth=1.2)
        ax_raw.axhline(stat["mean"], color=T_RED, linewidth=1.0, linestyle="--", alpha=0.9)
        ax_raw.set_title("Raw ADC value", color=T_TEXT, fontsize=10, pad=5)
        margin = max(60.0, stat["range"] * 0.18)
        ax_raw.set_ylim(max(SCALE_MIN, stat["min"] - margin),
                        min(SCALE_MAX, stat["max"] + margin))

        pressure_arr = SCALE_MAX - arr
        ax_pressure.plot(x_plot, pressure_arr, color=line_color, linewidth=1.2)
        ax_pressure.axhline(stat["peak"], color=T_RED, linewidth=1.0, linestyle="--", alpha=0.9)
        ax_pressure.set_title("Pressure view (4095 - raw)", color=T_TEXT, fontsize=10, pad=5)
        p_min = float(pressure_arr.min())
        p_max = float(pressure_arr.max())
        p_margin = max(60.0, (p_max - p_min) * 0.18)
        ax_pressure.set_ylim(max(SCALE_MIN, p_min - p_margin),
                             min(SCALE_MAX, p_max + p_margin))
        ax_pressure.set_xlabel(x_label, color=T_DIM, fontsize=9)

        fig.tight_layout(pad=1.1, h_pad=1.4)
        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        canvas.draw()

        tk.Button(
            popup, text="닫기", font=("맑은 고딕", 9),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, command=popup.destroy
        ).pack(fill=tk.X, padx=16, pady=(0, 12))

    def _pb_load(self):
        filepath = filedialog.askopenfilename(
            title="재생할 CSV 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except Exception as e:
            messagebox.showerror("파일 오류", str(e))
            return
        if not rows:
            messagebox.showwarning("빈 파일", "데이터가 없습니다.")
            return
        frames = []
        for row in rows:
            try:
                frames.append([float(row[f"ch{i}"]) for i in range(NUM_CHANNELS)])
            except (KeyError, ValueError):
                continue
        if not frames:
            messagebox.showwarning("파싱 실패", "ch0~ch15 컬럼을 찾을 수 없습니다.")
            return
        self._pb_stop()
        self._pb_frames = frames
        self._pb_idx = 0
        total = len(frames)
        self._pb_slider.config(to=max(0, total - 1), state=tk.NORMAL)
        self._pb_seeking = True
        self._pb_slider.set(0)
        self._pb_seeking = False
        self._pb_play_btn.config(state=tk.NORMAL)
        self._pb_stop_btn.config(state=tk.NORMAL)
        self._pb_file_label.config(
            text=os.path.basename(filepath), fg=T_BLUE)
        self._pb_frame_label.config(text=f"프레임  1 / {total}")
        self._pb_show_frame(0)
        self._open_csv_channel_graph(filepath, rows)

    def _pb_toggle_play(self):
        if not self._pb_frames:
            return
        if self._pb_playing:
            self._pb_playing = False
            self._pb_play_btn.config(text="▶")
        else:
            if self._pb_idx >= len(self._pb_frames) - 1:
                self._pb_idx = 0
            self._pb_playing = True
            self._pb_play_btn.config(text="⏸")
            self._pb_tick()

    def _pb_stop(self):
        self._pb_playing = False
        if self._pb_after_id:
            self.root.after_cancel(self._pb_after_id)
            self._pb_after_id = None
        if hasattr(self, "_pb_play_btn"):
            self._pb_play_btn.config(text="▶")
        if self._pb_frames:
            self._pb_idx = 0
            self._pb_seeking = True
            self._pb_slider.set(0)
            self._pb_seeking = False
            total = len(self._pb_frames)
            self._pb_frame_label.config(text=f"프레임  1 / {total}")
            self._pb_show_frame(0)

    def _pb_tick(self):
        if not self._pb_playing:
            return
        if self._pb_idx >= len(self._pb_frames) - 1:
            self._pb_playing = False
            self._pb_play_btn.config(text="▶")
            return
        self._pb_idx += 1
        self._pb_seeking = True
        self._pb_slider.set(self._pb_idx)
        self._pb_seeking = False
        total = len(self._pb_frames)
        self._pb_frame_label.config(text=f"프레임  {self._pb_idx + 1} / {total}")
        self._pb_show_frame(self._pb_idx)
        delay = max(5, int(10 / self._pb_speed))
        self._pb_after_id = self.root.after(delay, self._pb_tick)

    def _pb_seek(self, val):
        if self._pb_seeking or not self._pb_frames:
            return
        idx = int(float(val))
        self._pb_idx = idx
        total = len(self._pb_frames)
        self._pb_frame_label.config(text=f"프레임  {idx + 1} / {total}")
        self._pb_show_frame(idx)

    def _pb_show_frame(self, idx):
        if not self._pb_frames:
            return
        vals = self._pb_frames[idx]
        self.current_values = list(vals)
        self._draw_contour(vals, force=True)

    def _on_pb_speed(self, val):
        mapping = {"0.5×": 0.5, "1×": 1.0, "2×": 2.0, "5×": 5.0, "10×": 10.0}
        self._pb_speed = mapping.get(val, 1.0)

    # ---------------- 컬러맵 미리보기 팝업 ----------------
    _CMAPS = ["jet", "hot", "cool", "viridis", "plasma", "inferno", "RdYlBu_r"]

    def _make_cmap_photo(self, cmap_name, width, height):
        import tempfile
        try:
            cmap = matplotlib.colormaps[cmap_name]
        except AttributeError:
            cmap = matplotlib.cm.get_cmap(cmap_name)
        arr = np.linspace(0, 1, width)
        rgb = (cmap(arr)[:, :3] * 255).astype(np.uint8)
        ppm = f"P6\n{width} {height}\n255\n".encode() + rgb.tobytes() * height
        tmp = tempfile.NamedTemporaryFile(suffix=".ppm", delete=False)
        try:
            tmp.write(ppm)
            tmp.close()
            photo = tk.PhotoImage(file=tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
        return photo

    def _open_cmap_popup(self):
        if hasattr(self, "_cmap_popup") and self._cmap_popup.winfo_exists():
            self._cmap_popup.lift()
            return

        popup = tk.Toplevel(self.root)
        popup.title("컬러맵 선택")
        popup.configure(bg=T_BG)
        popup.resizable(False, False)
        self._cmap_popup = popup

        IMG_W, IMG_H = 180, 18

        tk.Label(popup, text="컬러맵 미리보기",
                 bg=T_BG, fg=T_TEXT, font=("맑은 고딕", 11, "bold")
                 ).pack(pady=(14, 10), padx=24)

        photos = []
        for cmap_name in self._CMAPS:
            photo = self._make_cmap_photo(cmap_name, IMG_W, IMG_H)
            photos.append(photo)
            is_sel = (cmap_name == self.cmap_name)
            row_bg = T_CARD if is_sel else T_BG

            row = tk.Frame(popup, bg=row_bg, cursor="hand2")
            row.pack(fill=tk.X, padx=16, pady=2)

            name_lbl = tk.Label(row, text=cmap_name, bg=row_bg, fg=T_TEXT,
                                font=("Consolas", 9), width=11, anchor="w")
            name_lbl.pack(side=tk.LEFT, padx=(10, 6), pady=6)

            cnv = tk.Canvas(row, width=IMG_W, height=IMG_H,
                            bg=row_bg, highlightthickness=2 if is_sel else 0,
                            highlightbackground=T_BLUE)
            cnv.pack(side=tk.LEFT, pady=6, padx=(0, 10))
            cnv.create_image(0, 0, anchor="nw", image=photo)

            def on_select(e, name=cmap_name):
                self._on_cmap_change(name)
                self._cmap_btn.config(text=f"{name}  ▾", fg=T_BLUE)
                popup.destroy()

            def on_enter(e, r=row, nl=name_lbl, c=cnv):
                r.config(bg=T_PANEL); nl.config(bg=T_PANEL); c.config(bg=T_PANEL)

            def on_leave(e, r=row, nl=name_lbl, c=cnv, sel=(cmap_name == self.cmap_name)):
                bg = T_CARD if sel else T_BG
                r.config(bg=bg); nl.config(bg=bg); c.config(bg=bg)

            for w in (row, name_lbl, cnv):
                w.bind("<Button-1>", on_select)
                w.bind("<Enter>", on_enter)
                w.bind("<Leave>", on_leave)

        popup.photos = photos  # GC 방지

        tk.Button(popup, text="닫기", font=("맑은 고딕", 9),
                  bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
                  relief=tk.FLAT, pady=5,
                  command=popup.destroy).pack(pady=(10, 14), padx=16, fill=tk.X)

        popup.update_idletasks()
        px = self.root.winfo_x() + (self.root.winfo_width() - popup.winfo_width()) // 2
        py = self.root.winfo_y() + (self.root.winfo_height() - popup.winfo_height()) // 2
        popup.geometry(f"+{px}+{py}")

    def _on_cmap_change(self, cmap_name):
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        self.cmap_name = cmap_name
        for cax, attr in ((self.cax_raw, "cb_raw"), (self.cax_cal, "cb_cal")):
            cax.cla()
            sm = ScalarMappable(norm=Normalize(vmin=SCALE_MIN, vmax=SCALE_MAX), cmap=cmap_name)
            sm.set_array([])
            cb = self.fig.colorbar(sm, cax=cax)
            cb.ax.yaxis.set_tick_params(color=T_TEXT)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color(T_TEXT)
            setattr(self, attr, cb)
        self._draw_contour(self.current_values, force=True)

    # ---------------- 동작 로직 ----------------
    def start_stream(self):
        port = self.port_entry.get().strip()
        self.reader = SerialReader(port, DEFAULT_BAUD, self.data_queue)
        self.reader.start()

        # 새 세션 시작 -> 이전 기록 초기화
        self.recorded_rows = []
        self.record_start_time = None
        self.live_min = None
        self.live_max = None
        self.record_count_label.config(text="기록  0 줄")
        self.save_btn.config(state=tk.DISABLED, fg=T_DIM)

        self.status_label.config(text=f"●  {port} 연결 중…", fg=T_ORNG)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL, bg=T_RED, fg=T_TEXT)
        self.port_entry.config(state=tk.DISABLED)

    def stop_stream(self):
        self._flush_active_clip()

        if self.reader:
            self.reader.stop()
            self.reader = None

        self.status_label.config(text="●  정지됨", fg=T_DIM)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED, bg=T_CARD, fg=T_DIM)
        self.port_entry.config(state=tk.NORMAL)
        self.record_count_label.config(text=f"기록  {len(self.recorded_rows)} 줄")

        if self.recorded_rows:
            self.save_btn.config(state=tk.NORMAL, fg=T_TEXT)
            # records/ 폴더에 자동으로도 한 부 보관해둔다 (관리자 탭에서 조회용).
            auto_name = datetime.datetime.now().strftime("pressure_%Y%m%d_%H%M%S.csv")
            auto_path = os.path.join(RECORDS_DIR, auto_name)
            try:
                self._write_csv_file(auto_path, self.recorded_rows)
            except OSError:
                pass  # 자동 보관은 실패해도 조용히 넘어감 (수동 저장은 별개로 가능)

    @staticmethod
    def _write_csv_file(filepath, rows):
        header = ["index", "elapsed_ms", "raw_timestamp_ms"] + [f"ch{i}" for i in range(NUM_CHANNELS)]
        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def save_csv(self):
        if not self.recorded_rows:
            messagebox.showinfo("저장", "저장할 데이터가 없습니다.")
            return

        default_name = datetime.datetime.now().strftime("pressure_%Y%m%d_%H%M%S.csv")
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        if not filepath:
            return

        try:
            self._write_csv_file(filepath, self.recorded_rows)
        except OSError as e:
            messagebox.showerror("저장 실패", str(e))
            return

        messagebox.showinfo("저장 완료", f"{len(self.recorded_rows)}줄 저장됨\n{filepath}")

    def reset_live_range(self):
        self.live_min = None
        self.live_max = None
        self.range_label.config(text="실시간: -")

    def _on_interp_change(self, value_str):
        self.smoothness = int(value_str)
        self.interp_value_label.config(text=str(self.smoothness))

        # 슬라이더를 움직이는 즉시 현재 값으로 다시 그려서 바로 체감되게 함
        if hasattr(self, "x_fine"):
            try:
                self._draw_contour(self.current_values, force=True)
            except Exception:
                pass  # 아직 데이터 수신 전이면 무시

    def poll_queue(self):
        latest_values = None   # 이번 주기에 들어온 것 중 가장 최신 프레임
        latest_index = None
        latest_timestamp = None

        try:
            while True:
                item = self.data_queue.get_nowait()

                if item[0] == 'error':
                    self.status_label.config(text="●  오류", fg=T_RED)
                    self.info_label.config(text=str(item[1])[:30])
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED, bg=T_CARD, fg=T_DIM)
                    self.port_entry.config(state=tk.NORMAL)

                elif item[0] == 'data':
                    _, index, timestamp, adc_values = item

                    # CSV 기록은 밀린 것까지 전부 누락 없이 저장한다.
                    if self.record_start_time is None:
                        self.record_start_time = timestamp
                    elapsed_ms = timestamp - self.record_start_time
                    self.recorded_rows.append(
                        [index, elapsed_ms, timestamp] + adc_values
                    )

                    # 화면 표시는 최신 프레임만 남긴다 (아래에서 1회만 그림).
                    latest_values = adc_values
                    latest_index = index
                    latest_timestamp = timestamp

        except queue.Empty:
            pass

        # 렌더링은 이번 주기에 딱 한 번. 밀린 과거 프레임은 그리지 않고 버린다.
        # (전부 그리면 렌더링이 수신 속도를 못 따라가 지연이 무한히 누적된다.)
        if latest_values is not None:
            self.status_label.config(text="●  수신 중", fg=T_GREEN)
            self.info_label.config(
                text=f"idx={latest_index}  ts={latest_timestamp}ms"
            )
            self.current_values = latest_values
            self._has_live_data = True
            self._update_live_range(latest_values)
            self.record_count_label.config(
                text=f"기록  {len(self.recorded_rows)} 줄"
            )
            try:
                self._draw_contour(latest_values)
            except Exception as e:
                self.status_label.config(text="●  렌더링 오류", fg=T_RED)
                self.info_label.config(text=str(e)[:30])

        self.root.after(10, self.poll_queue)

    def _update_live_range(self, values):
        v_min = min(values)
        v_max = max(values)

        if self.live_min is None or v_min < self.live_min:
            self.live_min = v_min
        if self.live_max is None or v_max > self.live_max:
            self.live_max = v_max

        self.range_label.config(text=f"실시간(raw): {self.live_min} ~ {self.live_max}")

    def _compute_display_row(self, clipped_values):
        """각 채널은 자기 칸(폭 1) 안에서 대부분 자기 값 그대로 평평하게
        유지되고, 칸과 칸의 경계 쪽 좁은 구간에서만 이웃과 부드럽게
        이어진다. 세게 눌러도 그 채널 칸의 대부분은 항상 원본값이다.

        예전 방식(채널 사이 전체 폭을 걸쳐 직선/S자로 잇는 방식)은,
        예를 들어 어떤 채널만 안 눌리고 양옆이 둘 다 눌린 경우 그 칸
        전체가 옆의 눌린 색으로 뭉개져 보이는 문제가 있었다. 이 방식은
        칸의 중심부(core)는 절대 안 건드리고, 경계에 가까운 좁은 구간
        (edge)만 이웃과 섞이므로 그 문제가 생기지 않는다.

        0   : 경계 폭 0 (완전한 계단식, 칸 전체가 자기 값)
        100 : 경계 폭이 칸 절반까지 넓어짐 (칸 경계에서 정확히 두 채널의
              평균값과 만나는 부드러운 전환)
        """
        v = clipped_values
        c = self.nearest_idx
        d = self.cell_offset

        half_w = 0.5
        edge = (self.smoothness / 100.0) * half_w
        core = half_w - edge

        ad = np.abs(d)
        result = v[c].copy()

        edge_mask = ad > core
        if edge > 1e-9:
            neighbor = np.clip(c + np.sign(d).astype(int), 0, NUM_CHANNELS - 1)
            local_t = np.clip((ad - core) / edge, 0.0, 1.0)
            eased = 3 * local_t ** 2 - 2 * local_t ** 3
            blended = v[c] * (1 - eased * 0.5) + v[neighbor] * (eased * 0.5)
            result = np.where(edge_mask, blended, result)

        return result

    def _draw_contour(self, values, force=False):
        now = time.time()
        if not force and (now - self._last_contour_draw) < self.CONTOUR_REDRAW_INTERVAL:
            return
        self._last_contour_draw = now

        arr = np.asarray(values, dtype=float)

        if self._has_live_data:
            # 규칙 기반 최대 압력 추적 — 활성 채널 중 필터 ON인 것만
            active_enabled = [ch for ch in ML_ACTIVE_CH if self._ch_enabled[ch]]
            if active_enabled:
                rule_max = float(np.clip(4095 - arr[active_enabled], 0, 4095).max() / 4095)
            else:
                rule_max = 0.0
            self._rule_max_history.append(rule_max)

            # ML 버퍼 갱신 및 추론
            self._ml_buffer.append(arr.tolist())
            if self._ml_model and len(self._ml_buffer) == ML_SEQ_LEN:
                self._run_ml_inference()

        # ── 좌측: 원본 (비활성화 채널은 0 압력으로 표시) ─────────────
        display_arr = arr.copy()
        for _ch in range(NUM_CHANNELS):
            if not self._ch_enabled[_ch]:
                display_arr[_ch] = float(SCALE_MAX)  # ADC 최대 = 압력 0
        raw_display = np.clip(SCALE_MAX - display_arr, SCALE_MIN, SCALE_MAX)
        Z_raw = np.tile(self._compute_display_row(raw_display), (self.STRIP_ROWS, 1))

        self.ax_raw.clear()
        self._style_axes(self.ax_raw)
        self.ax_raw.set_title("BEFORE  /  원본", color=T_DIM, fontsize=8.5, pad=5)
        self.ax_raw.contourf(self.x_fine, self.y_axis, Z_raw,
                             levels=self.contour_levels, cmap=self.cmap_name)

        # ── 우측: 보정 적용 (offset - raw) 또는 안내 텍스트 ────────────
        self.ax_cal.clear()
        self._style_axes(self.ax_cal)
        self.ax_cal.set_title("AFTER  /  보정 적용", color=T_BLUE, fontsize=8.5, pad=5)
        if self.cal_offsets and self.cal_apply:
            offsets = np.array([self.cal_offsets.get(i, float(SCALE_MAX))
                                for i in range(NUM_CHANNELS)])
            cal_display = np.clip(offsets - display_arr, SCALE_MIN, SCALE_MAX)
            Z_cal = np.tile(self._compute_display_row(cal_display), (self.STRIP_ROWS, 1))
            self.ax_cal.contourf(self.x_fine, self.y_axis, Z_cal,
                                 levels=self.contour_levels, cmap=self.cmap_name)
        else:
            self.ax_cal.set_facecolor(T_FIG)
            if self.cal_offsets:
                msg = "보정값 로드됨\n왼쪽 CALIBRATION에서\n보정 적용을 ON 하세요"
            else:
                msg = "보정값 없음\n보정 탭에서 계산하거나\nJSON 파일을 불러오세요"
            self.ax_cal.text(
                0.5, 0.5,
                msg,
                transform=self.ax_cal.transAxes,
                ha="center", va="center", color=T_DIM,
                fontsize=9
            )

        if not self._has_live_data:
            for i in range(NUM_CHANNELS):
                if not self._ch_enabled[i]:
                    self.raw_val_labels[i].config(text="OFF", bg=T_BORD, fg=T_DIM)
                    self.cal_val_labels[i].config(text="OFF", bg=T_BORD, fg=T_DIM)
                else:
                    self.raw_val_labels[i].config(text="-", bg=T_CARD, fg=T_DIM)
                    self.cal_val_labels[i].config(text="-", bg=T_CARD, fg=T_DIM)
            self.ch_status_label.config(text="●  수신 대기", fg=T_DIM)
            self.canvas_widget.draw_idle()
            return

        # ── 수치 테이블 갱신 + 채널 이상 감지 ────────────────────────────
        warn_channels = []

        for i in range(NUM_CHANNELS):
            # 비활성화 채널은 회색으로 표시하고 이상 감지 건너뜀
            if not self._ch_enabled[i]:
                prev = self._prev_anomaly[i]
                if prev is not None:
                    self._log_anomaly(i, "복구", prev, int(arr[i]), None)
                    self._prev_anomaly[i] = None
                self._ch_history[i].clear()
                self.raw_val_labels[i].config(text="OFF", bg=T_BORD, fg=T_DIM)
                self.cal_val_labels[i].config(text="OFF", bg=T_BORD, fg=T_DIM)
                continue

            raw_v = int(arr[i])

            # 히스토리 갱신 (최근 _FROZEN_SAMPLES 개만 유지)
            hist = self._ch_history[i]
            hist.append(raw_v)
            if len(hist) > self._FROZEN_SAMPLES:
                hist.pop(0)

            # 이상 여부 판정
            reason = None
            if raw_v <= self._SAT_LOW or raw_v >= self._SAT_HIGH:
                reason = "단선"
            elif (len(hist) >= self._FROZEN_SAMPLES
                  and float(np.std(hist)) < self._FROZEN_STD):
                reason = "무변화"

            # 보정Δ 계산 및 이상 추가 판정
            delta = None
            if self.cal_offsets and self.cal_apply:
                delta = int(self.cal_offsets.get(i, SCALE_MAX) - arr[i])
                delta_clamped = max(SCALE_MIN, min(SCALE_MAX, delta))
                if reason is None and delta < self._CAL_UNDERFLOW:
                    reason = "보정이탈"

            # 셀 색상 결정
            # 상태 변화 시에만 로그 (매 프레임 쓰지 않음)
            prev = self._prev_anomaly[i]
            if reason != prev:
                if reason is not None:
                    self._log_anomaly(i, "감지", reason, raw_v, delta)
                    self._fire_alarm(reason=f"ch{i:02d} {reason} 감지 (raw={raw_v})")
                else:
                    self._log_anomaly(i, "복구", prev, raw_v, delta)
                self._prev_anomaly[i] = reason

            if reason:
                warn_channels.append((i, reason))
                raw_bg, raw_fg = self._warn_bg, self._warn_fg
                cal_bg, cal_fg = self._warn_bg, self._warn_fg
            else:
                raw_bg, raw_fg = T_CARD, T_TEXT
                if delta is not None:
                    cal_bg = T_CARD
                    cal_fg = self._cal_fg_press if delta > 300 else T_BLUE
                else:
                    cal_bg, cal_fg = T_CARD, T_BLUE

            # 원본 셀 갱신
            self.raw_val_labels[i].config(text=str(raw_v), bg=raw_bg, fg=raw_fg)

            # 보정Δ 셀 갱신
            if delta is not None:
                self.cal_val_labels[i].config(
                    text=str(delta_clamped), bg=cal_bg, fg=cal_fg
                )
            else:
                self.cal_val_labels[i].config(text="-", bg=cal_bg, fg=cal_fg)

        # 상태 바 갱신
        if warn_channels:
            parts = [f"ch{i}({r})" for i, r in warn_channels]
            self.ch_status_label.config(
                text="⚠  이상 감지: " + "  ".join(parts), fg=T_ORNG)
        else:
            self.ch_status_label.config(text="●  전 채널 정상", fg=T_GREEN)

        self.canvas_widget.draw_idle()

    def _log_anomaly(self, channel, event, reason, raw_v, delta):
        is_new = (not os.path.exists(ANOMALY_LOG_PATH)
                  or os.path.getsize(ANOMALY_LOG_PATH) == 0)
        try:
            with open(ANOMALY_LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                if is_new:
                    w.writerow(["datetime", "channel", "event", "reason",
                                "raw_adc", "cal_delta"])
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                w.writerow([now_str, f"ch{channel}", event, reason,
                            raw_v, str(delta) if delta is not None else "-"])
        except OSError:
            pass

    def _open_anomaly_log(self):
        path = os.path.abspath(ANOMALY_LOG_PATH)
        if not os.path.exists(path):
            messagebox.showinfo("이상 로그", "아직 기록된 이상 이력이 없습니다.")
            return
        self._open_path(path)

    # ─────────────────────────────────────────────────────────────────
    # 채널별 보정 오프셋 그래픽 편집기
    # ─────────────────────────────────────────────────────────────────

    def _open_cal_graphic(self):
        if hasattr(self, '_cal_graphic_win') and self._cal_graphic_win.winfo_exists():
            self._cal_graphic_win.lift()
            return

        DEAD = self._DEAD_CH_SET

        popup = tk.Toplevel(self.root)
        popup.title("채널별 보정 오프셋 편집기")
        popup.configure(bg=T_BG)
        popup.geometry("1040x400")
        popup.resizable(True, False)
        self._cal_graphic_win = popup

        # ── 초기화: 비어 있으면 4095로 채움 ─────────────────────────
        for i in range(NUM_CHANNELS):
            if i not in DEAD and i not in self.cal_offsets:
                self.cal_offsets[i] = 4095.0

        # ── 헤더 버튼 행 ─────────────────────────────────────────────
        btn_row = tk.Frame(popup, bg=T_BG)
        btn_row.pack(fill=tk.X, padx=16, pady=(12, 8))

        val_labels = {}
        cur_labels = {}
        slider_vars = {}

        def _set_to_current():
            vals = getattr(self, 'current_values', None)
            if not vals:
                return
            for i in range(NUM_CHANNELS):
                if i not in DEAD:
                    v = max(0, min(4095, int(round(float(vals[i])))))
                    slider_vars[i].set(v)
                    self.cal_offsets[i] = float(v)
                    val_labels[i].config(text=str(v))
            self._sync_cal_offsets_from_tab()

        def _reset_all():
            for i in range(NUM_CHANNELS):
                if i not in DEAD:
                    slider_vars[i].set(4095)
                    self.cal_offsets[i] = 4095.0
                    val_labels[i].config(text="4095")
            self._sync_cal_offsets_from_tab()

        def _save_json():
            p = filedialog.asksaveasfilename(
                parent=popup, title="보정값 JSON 저장",
                defaultextension=".json",
                filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")])
            if not p:
                return
            data = {"offsets": {str(k): v for k, v in self.cal_offsets.items()}}
            try:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("저장 완료", f"저장: {os.path.basename(p)}", parent=popup)
            except OSError as e:
                messagebox.showerror("저장 실패", str(e), parent=popup)

        for text, cmd, col in [
            ("현재값으로 설정", _set_to_current, T_BLUE),
            ("전체 리셋 (4095)", _reset_all,     T_CARD),
            ("JSON 저장",        _save_json,      T_CARD),
        ]:
            tk.Button(btn_row, text=text, font=("맑은 고딕", 9),
                      bg=col, fg="#ffffff" if col == T_BLUE else T_DIM,
                      activebackground=T_BORD, relief=tk.FLAT, pady=4,
                      command=cmd).pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(btn_row, text="▲ 슬라이더 ↑ 올릴수록 더 민감  |  ▶ = 현재 수신값",
                 bg=T_BG, fg=T_DIM, font=("Consolas", 7)).pack(side=tk.RIGHT)

        # ── 슬라이더 영역 ─────────────────────────────────────────────
        grid = tk.Frame(popup, bg=T_BG)
        grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        SLIDER_LEN = 200

        def _make_cb(ch):
            def cb(val):
                v = int(float(val))
                self.cal_offsets[ch] = float(v)
                val_labels[ch].config(text=str(v))
                self._sync_cal_offsets_from_tab()
            return cb

        for i in range(NUM_CHANNELS):
            is_dead = i in DEAD
            col_frame = tk.Frame(grid, bg=T_BG)
            col_frame.grid(row=0, column=i, padx=2, sticky='n')

            # 채널 레이블
            tk.Label(col_frame, text=f"ch{i:02d}",
                     bg=T_BG, fg=T_BORD if is_dead else T_BLUE,
                     font=("Consolas", 7, "bold")).pack(pady=(0, 2))

            if is_dead:
                # 사망 채널: 비활성 사각형 표시
                dead_bar = tk.Frame(col_frame, bg=T_BORD, width=14,
                                    height=SLIDER_LEN)
                dead_bar.pack()
                dead_bar.pack_propagate(False)
                tk.Label(dead_bar, text="✗", bg=T_BORD, fg=T_DIM,
                         font=("Consolas", 8)).pack(expand=True)
                tk.Label(col_frame, text="─", bg=T_BG, fg=T_DIM,
                         font=("Consolas", 7)).pack(pady=(2, 0))
                tk.Label(col_frame, text="─", bg=T_BG, fg=T_DIM,
                         font=("Consolas", 6)).pack()
            else:
                sv = tk.IntVar(value=int(self.cal_offsets.get(i, 4095)))
                slider_vars[i] = sv
                sc = tk.Scale(
                    col_frame, variable=sv,
                    from_=4095, to=0,
                    orient=tk.VERTICAL, length=SLIDER_LEN,
                    bg=T_BG, fg=T_TEXT, troughcolor=T_CARD,
                    highlightthickness=0, bd=0, showvalue=False,
                    sliderlength=12, width=14,
                    command=_make_cb(i))
                sc.pack()

                # 오프셋값 레이블 (주황)
                vl = tk.Label(col_frame,
                              text=str(int(self.cal_offsets.get(i, 4095))),
                              bg=T_BG, fg=T_ORNG, font=("Consolas", 7))
                vl.pack(pady=(2, 0))
                val_labels[i] = vl

                # 현재 수신값 레이블 (실시간 갱신)
                cl = tk.Label(col_frame, text="─",
                              bg=T_BG, fg=T_DIM, font=("Consolas", 6))
                cl.pack()
                cur_labels[i] = cl

        # ── 실시간 현재값 갱신 루프 ──────────────────────────────────
        def _tick():
            if not popup.winfo_exists():
                return
            vals = getattr(self, 'current_values', None)
            if vals:
                for i in range(NUM_CHANNELS):
                    if i not in DEAD:
                        raw = int(float(vals[i]))
                        offset = int(self.cal_offsets.get(i, 4095))
                        delta = offset - raw
                        cur_labels[i].config(
                            text=f"▶{raw}",
                            fg=T_GREEN if delta >= 0 else T_RED)
            popup.after(400, _tick)

        _tick()

    def _load_calibration_file(self):
        filepath = filedialog.askopenfilename(
            title="보정 파일(JSON) 선택",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_offsets = data.get("offsets", {})
            self.cal_offsets = {int(k): float(v) for k, v in raw_offsets.items()}
        except Exception as e:
            messagebox.showerror("불러오기 실패", str(e))
            return
        self.cal_loaded_label.config(
            text=f"{os.path.basename(filepath)}\n({len(self.cal_offsets)}채널)",
            fg=T_BLUE
        )
        self._update_cal_apply_button()
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    def _update_cal_apply_button(self):
        if not hasattr(self, 'cal_apply_btn'):
            return
        if not self.cal_offsets:
            self.cal_apply = False
            self.cal_apply_btn.config(
                text="보정 적용: OFF", bg=T_CARD, fg=T_DIM, state=tk.DISABLED)
            return
        self.cal_apply_btn.config(state=tk.NORMAL)
        if self.cal_apply:
            self.cal_apply_btn.config(text="보정 적용: ON", bg=T_ORNG, fg="white")
        else:
            self.cal_apply_btn.config(text="보정 적용: OFF", bg=T_CARD, fg=T_DIM)

    def _toggle_cal_apply(self):
        if not self.cal_offsets:
            return
        self.cal_apply = not self.cal_apply
        self._update_cal_apply_button()
        # 즉시 화면 갱신
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    def _sync_cal_offsets_from_tab(self):
        if not self.cal_offsets:
            return
        self.cal_loaded_label.config(
            text=f"보정 탭에서 계산됨\n({len(self.cal_offsets)}채널)",
            fg=T_BLUE
        )
        self._update_cal_apply_button()
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    def _apply_calibration_from_tab(self):
        if not self.cal_offsets:
            messagebox.showwarning("보정 적용", "먼저 보정 실행으로 오프셋을 계산하세요.")
            return
        self.cal_apply = True
        self._update_cal_apply_button()
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    # ---------------- 보정 탭 ----------------
    def _build_calibration_tab(self, parent):
        self.cal_data = []
        self.cal_filepath = ""
        self.cal_offsets = {}

        left = tk.Frame(parent, bg=T_PANEL, width=272)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="압력 보정", bg=T_PANEL, fg=T_TEXT,
                 font=("맑은 고딕", 14, "bold")).pack(pady=(20, 6))
        tk.Label(left, text="무압력 상태에서 기록한 CSV로\n채널별 영점 오프셋을 계산합니다.",
                 bg=T_PANEL, fg=T_DIM, font=("맑은 고딕", 8),
                 justify="center").pack(pady=(0, 10))
        tk.Frame(left, bg=T_BORD, height=1).pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Label(left, text="① CSV 업로드", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20)
        tk.Button(
            left, text="CSV 업로드", font=("맑은 고딕", 11, "bold"),
            bg=T_BLUE, fg=T_TEXT, activebackground=T_GRNH,
            relief=tk.FLAT, height=2, command=self._upload_cal_csv
        ).pack(padx=20, pady=(4, 4), fill=tk.X)

        self.cal_file_label = tk.Label(
            left, text="파일이 선택되지 않음", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 8), wraplength=232, justify="left"
        )
        self.cal_file_label.pack(anchor="w", padx=20, pady=(0, 6))

        tk.Button(
            left, text="샘플 CSV 생성", font=("맑은 고딕", 9),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD, relief=tk.FLAT,
            command=self._generate_sample_csv
        ).pack(padx=20, pady=(0, 14), fill=tk.X)

        tk.Frame(left, bg=T_BORD, height=1).pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Label(left, text="② 보정 실행", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20)
        self.cal_run_btn = tk.Button(
            left, text="보  정", font=("맑은 고딕", 13, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, height=2, state=tk.DISABLED,
            command=self._run_calibration
        )
        self.cal_run_btn.pack(padx=20, pady=(4, 14), fill=tk.X)

        tk.Frame(left, bg=T_BORD, height=1).pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Label(left, text="채널별 평균값 (영점 기준)", bg=T_PANEL, fg=T_DIM,
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20, pady=(0, 4))
        self.cal_result_text = tk.Text(
            left, font=("Consolas", 8), bg=T_BG, fg=T_TEXT,
            relief=tk.FLAT, height=17, state=tk.DISABLED
        )
        self.cal_result_text.pack(padx=20, pady=(0, 8), fill=tk.X)

        self.cal_save_btn = tk.Button(
            left, text="보정값 저장 (JSON)", font=("맑은 고딕", 10, "bold"),
            bg=T_CARD, fg=T_DIM, relief=tk.FLAT,
            state=tk.DISABLED, command=self._save_calibration
        )
        self.cal_save_btn.pack(padx=20, pady=(0, 6), fill=tk.X)

        self.cal_apply_now_btn = tk.Button(
            left, text="계산값 바로 적용", font=("맑은 고딕", 10, "bold"),
            bg=T_CARD, fg=T_DIM, relief=tk.FLAT,
            state=tk.DISABLED, command=self._apply_calibration_from_tab
        )
        self.cal_apply_now_btn.pack(padx=20, pady=(0, 20), fill=tk.X)

        right = tk.Frame(parent, bg=T_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="채널별 데이터  ━  파랑: 원본값  /  빨강 점선: 평균(오프셋)",
                 bg=T_BG, fg=T_DIM,
                 font=("맑은 고딕", 9)).pack(pady=(12, 4))

        self.cal_fig = Figure(figsize=(7.2, 5.6), dpi=90, facecolor=T_FIG)
        self.cal_axes = []
        for i in range(NUM_CHANNELS):
            ax = self.cal_fig.add_subplot(4, 4, i + 1)
            ax.set_facecolor(T_CARD)
            ax.set_title(f"ch{i}", color=T_DIM, fontsize=7, pad=2)
            ax.tick_params(colors=T_DIM, labelsize=5)
            for spine in ax.spines.values():
                spine.set_color(T_BORD)
            self.cal_axes.append(ax)

        self.cal_fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.5)
        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=right)
        self.cal_canvas.get_tk_widget().pack(pady=4, padx=10, fill=tk.BOTH, expand=True)
        self.cal_canvas.draw()

    def _draw_cal_csv_preview(self):
        if not self.cal_data:
            return
        try:
            xs, channels = self._csv_rows_to_channel_arrays(self.cal_data)
        except Exception:
            return

        x_plot = xs - xs[0]
        x_label = "ms"
        if len(x_plot) > 1 and x_plot[-1] >= 5000:
            x_plot = x_plot / 1000.0
            x_label = "s"

        plot_bg = "#101820" if self._is_dark else "#ffffff"
        grid_col = "#2f3b45" if self._is_dark else "#d8dee4"

        for i, ax in enumerate(self.cal_axes):
            ax.clear()
            ax.set_facecolor(plot_bg)
            arr = channels[i]
            mean_val = float(arr.mean())
            min_val = float(arr.min())
            max_val = float(arr.max())
            ax.grid(True, color=grid_col, linewidth=0.45, alpha=0.45)
            ax.plot(x_plot, arr, color=T_BLUE, linewidth=0.8, alpha=0.95)
            ax.axhline(mean_val, color=T_RED, linewidth=0.85, linestyle="--", alpha=0.9)
            ax.set_title(f"ch{i:02d} avg {mean_val:.0f}", color=T_TEXT, fontsize=6.5, pad=2)
            ax.tick_params(colors=T_DIM, labelsize=5, length=0)
            for spine in ax.spines.values():
                spine.set_color(T_BORD)
            margin = max(60.0, (max_val - min_val) * 0.18)
            ax.set_ylim(max(SCALE_MIN, min_val - margin),
                        min(SCALE_MAX, max_val + margin))
            if i >= 12:
                ax.set_xlabel(x_label, color=T_DIM, fontsize=5)

        self.cal_fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.5)
        self.cal_canvas.draw_idle()

    def _upload_cal_csv(self):
        filepath = filedialog.askopenfilename(
            title="보정용 CSV 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                self.cal_data = list(csv.DictReader(f))
        except Exception as e:
            messagebox.showerror("파일 오류", str(e))
            return
        if not self.cal_data:
            messagebox.showwarning("빈 파일", "데이터가 없는 CSV입니다.")
            return
        self.cal_filepath = filepath
        self.cal_file_label.config(
            text=f"{os.path.basename(filepath)}\n({len(self.cal_data)}줄)",
            fg=T_BLUE
        )
        self.cal_run_btn.config(state=tk.NORMAL, bg=T_ORNG, fg=T_TEXT)
        self._draw_cal_csv_preview()

    def _generate_sample_csv(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile="calibration_sample.csv",
            filetypes=[("CSV 파일", "*.csv")]
        )
        if not save_path:
            return
        rng = np.random.default_rng(42)
        # 채널마다 다른 베이스라인으로 실제 센서 특성 모사
        baselines = rng.integers(2800, 3800, size=NUM_CHANNELS)
        n_rows = 200
        header = ["index", "elapsed_ms", "raw_timestamp_ms"] + [f"ch{i}" for i in range(NUM_CHANNELS)]
        rows = []
        for idx in range(n_rows):
            elapsed = idx * 10
            values = [int(np.clip(baselines[ch] + rng.normal(0, 40), 0, 4095))
                      for ch in range(NUM_CHANNELS)]
            rows.append([idx, elapsed, 1000 + elapsed] + values)
        try:
            with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            messagebox.showinfo("생성 완료", f"샘플 CSV 생성됨:\n{save_path}")
        except OSError as e:
            messagebox.showerror("저장 실패", str(e))

    def _run_calibration(self):
        if not self.cal_data:
            return

        try:
            elapsed, channels = self._csv_rows_to_channel_arrays(self.cal_data)
        except ValueError as e:
            messagebox.showwarning("파싱 오류", f"데이터를 읽을 수 없습니다. CSV 헤더를 확인하세요.\n{e}")
            return

        # 채널별 평균 계산 → 영점 오프셋
        self.cal_offsets = {}
        for i in range(NUM_CHANNELS):
            self.cal_offsets[i] = float(np.mean(channels[i]))

        # 16채널 그래프 그리기
        for i, ax in enumerate(self.cal_axes):
            ax.clear()
            ax.set_facecolor(T_CARD)
            arr = channels[i]
            mean_val = self.cal_offsets[i]

            ax.plot(elapsed, arr, color=T_BLUE, linewidth=0.8, alpha=0.9)
            ax.axhline(mean_val, color=T_RED, linewidth=1.2, linestyle="--")
            ax.set_title(f"ch{i}  {mean_val:.0f}", color=T_TEXT, fontsize=6.5, pad=2)
            ax.tick_params(colors=T_DIM, labelsize=5)
            for spine in ax.spines.values():
                spine.set_color(T_BORD)
            margin = max(200, float(np.std(arr)) * 4)
            ax.set_ylim(
                max(0, mean_val - margin),
                min(4095, mean_val + margin)
            )

        self.cal_fig.tight_layout(pad=0.8, h_pad=1.0, w_pad=0.5)
        self.cal_canvas.draw()

        # 결과 텍스트
        self.cal_result_text.config(state=tk.NORMAL)
        self.cal_result_text.delete("1.0", tk.END)
        lines = [f"ch{i:2d}: {self.cal_offsets[i]:8.1f}" for i in range(NUM_CHANNELS)]
        self.cal_result_text.insert("1.0", "\n".join(lines))
        self.cal_result_text.config(state=tk.DISABLED)

        self.cal_save_btn.config(state=tk.NORMAL, bg=T_GREEN, fg=T_TEXT)
        self.cal_apply_now_btn.config(state=tk.NORMAL, bg=T_ORNG, fg=T_TEXT)
        self._sync_cal_offsets_from_tab()
        messagebox.showinfo(
            "보정 완료",
            f"16채널 보정 완료\n\n"
            f"그래프: 파랑=원본 / 빨강점선=평균(오프셋)\n\n"
            f"측정 탭 → '보정 적용' 버튼으로 실시간 적용 ON/OFF\n"
            f"'보정값 저장' 버튼으로 JSON 저장 가능"
        )

    def _save_calibration(self):
        if not self.cal_offsets:
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="calibration.json",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        if not save_path:
            return
        cal_data = {
            "offsets": {str(i): self.cal_offsets[i] for i in range(NUM_CHANNELS)},
            "source_file": os.path.basename(self.cal_filepath),
            "created_at": datetime.datetime.now().isoformat()
        }
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cal_data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("저장 완료", f"보정값 저장됨:\n{save_path}")
        except OSError as e:
            messagebox.showerror("저장 실패", str(e))

    # ─────────────────────────────────────────────────────────────────
    # 사용자 설정 저장 / 복원
    # ─────────────────────────────────────────────────────────────────

    def _save_settings(self):
        try:
            data = {
                'port':           self.port_entry.get().strip(),
                'sigma_k':        round(self._ml_sigma_k, 2),
                'rule_thresh':    round(self._rule_thresh, 2),
                'is_dark':        self._is_dark,
                'cmap_name':      self.cmap_name,
                'pb_speed':       self._pb_speed,
                'alarm_enabled':  self._alarm_enabled,
                'alarm_mode':     self._alarm_mode,
                'alarm_cooldown': int(self._alarm_cooldown),
                'cal_apply':      self.cal_apply,
                'ch_enabled':     self._ch_enabled,
            }
            with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            return

        # ── 포트 ─────────────────────────────────────────────────────
        port = str(data.get('port', DEFAULT_PORT))
        self.port_entry.delete(0, tk.END)
        self.port_entry.insert(0, port)

        # ── ML σ 배수 ─────────────────────────────────────────────────
        try:
            self._ml_sigma_k = float(data['sigma_k'])
        except (KeyError, ValueError):
            pass

        # ── 규칙 임계값 ───────────────────────────────────────────────
        try:
            rt = float(data['rule_thresh'])
            self._rule_thresh = max(0.05, min(0.95, rt))
            self._rule_thresh_var.set(self._rule_thresh)
            self._rule_thresh_val_lbl.config(text=f"{self._rule_thresh:.2f}")
        except (KeyError, ValueError):
            pass

        # 밝은 테마를 기본값으로 유지한다. 테마 버튼으로 수동 전환은 가능하다.

        # ── 컬러맵 ───────────────────────────────────────────────────
        try:
            cmap = str(data['cmap_name'])
            if cmap != self.cmap_name:
                self._on_cmap_change(cmap)
        except (KeyError, ValueError):
            pass

        # ── 재생 속도 ─────────────────────────────────────────────────
        try:
            self._pb_speed = float(data.get('pb_speed', 1.0))
        except ValueError:
            pass

        # ── 알람 ─────────────────────────────────────────────────────
        try:
            enabled = bool(data.get('alarm_enabled', False))
            if enabled != self._alarm_enabled:
                self._toggle_alarm()
        except Exception:
            pass
        try:
            mode = str(data['alarm_mode'])
            if mode in ("소리", "토스트", "소리+토스트"):
                self._alarm_mode = mode
                self._alarm_mode_var.set(mode)
        except (KeyError, AttributeError):
            pass
        try:
            cd = float(data['alarm_cooldown'])
            self._alarm_cooldown = cd
            self._alarm_cooldown_var.set(cd)
            self._alarm_cooldown_lbl.config(text=f"{int(cd)}s")
        except (KeyError, AttributeError, ValueError):
            pass

        try:
            self.cal_apply = bool(data.get('cal_apply', False))
            self._update_cal_apply_button()
        except Exception:
            pass

        # ── 채널 필터 ─────────────────────────────────────────────────
        try:
            saved_enabled = list(data['ch_enabled'])
            if len(saved_enabled) == NUM_CHANNELS:
                for ch in range(NUM_CHANNELS):
                    if ch in self._DEAD_CH_SET:
                        continue
                    want = bool(saved_enabled[ch])
                    if want != self._ch_enabled[ch]:
                        self._toggle_ch_filter(ch)
        except (KeyError, TypeError):
            pass

    def on_close(self):
        self._flush_active_clip()
        self._save_settings()
        self._pb_stop()
        if self.reader:
            self.reader.stop()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
