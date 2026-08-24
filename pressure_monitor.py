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
import json
import hashlib
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

NUM_CHANNELS = 16
DEFAULT_PORT = 'COM3'
DEFAULT_BAUD = 115200
SCALE_MIN = 0
SCALE_MAX = 4095  # 히트맵 색상 범위 고정. 음수는 나오지 않게 클리핑한다.

ADMIN_CONFIG_PATH = 'admin_config.json'
RECORDS_DIR = 'records'
DEFAULT_ADMIN_ID = 'admin'
DEFAULT_ADMIN_PW = '1234'
ANOMALY_LOG_PATH = 'anomaly_log.csv'

# ── UI 테마 ──────────────────────────────────────────────────────────
T_BG    = "#0d1117"   # 창 배경
T_PANEL = "#161b22"   # 좌측 패널
T_CARD  = "#21262d"   # 카드/입력 배경
T_BORD  = "#30363d"   # 구분선
T_TEXT  = "#e6edf3"   # 기본 텍스트
T_DIM   = "#8b949e"   # 보조 텍스트
T_GREEN = "#238636"   # 시작/정상
T_GRNH  = "#2ea043"   # 시작 hover
T_BLUE  = "#388bfd"   # 강조
T_ORNG  = "#d29922"   # 경고
T_RED   = "#da3633"   # 정지/오류
T_FIG   = "#0d1117"   # matplotlib figure 배경


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

        # 색상 범위는 SCALE_MIN~SCALE_MAX 고정. 아래는 참고 표시용 실시간 min/max.
        self.live_min = None
        self.live_max = None
        self.current_values = [0] * NUM_CHANNELS

        # Start~Stop 구간 동안 쌓이는 기록 (index, elapsed_ms, raw_timestamp, ch0..ch15)
        self.recorded_rows = []
        self.record_start_time = None

        self._last_contour_draw = 0.0
        self.smoothness = 50  # 0=계단식, 100=경계 넓게 부드러움 (칸 중심은 항상 원본값)

        self.cal_offsets = {}
        self.cal_apply = False
        self._ch_history   = [[] for _ in range(NUM_CHANNELS)]
        self._prev_anomaly = [None] * NUM_CHANNELS  # 직전 이상 상태 (변화 시에만 로그)

        # --- 관리자 계정 / 기록 보관 폴더 준비 ---
        self.admin_config, is_new_config = load_or_create_admin_config()
        self.admin_authenticated = False
        os.makedirs(RECORDS_DIR, exist_ok=True)

        # --- 탭(Notebook) 구성 ---
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure('TNotebook', background=T_BG, borderwidth=0)
        style.configure('TNotebook.Tab', background=T_PANEL, foreground=T_DIM,
                        padding=(22, 9), font=("맑은 고딕", 10))
        style.map('TNotebook.Tab',
                 background=[('selected', T_CARD)],
                 foreground=[('selected', T_TEXT)])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        measuring_tab = tk.Frame(self.notebook, bg=T_BG)
        admin_tab = tk.Frame(self.notebook, bg=T_BG)
        calibration_tab = tk.Frame(self.notebook, bg=T_BG)
        self.notebook.add(measuring_tab, text="측정")
        self.notebook.add(admin_tab, text="관리자")
        self.notebook.add(calibration_tab, text="보정")

        self._build_left_panel(measuring_tab)
        self._build_right_panel(measuring_tab)
        self._build_admin_tab(admin_tab)
        self._build_calibration_tab(calibration_tab)

        self.root.after(10, self.poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if is_new_config:
            messagebox.showinfo(
                "관리자 계정 생성됨",
                f"관리자 설정파일({ADMIN_CONFIG_PATH})이 없어서 기본 계정으로 새로 만들었습니다.\n\n"
                f"아이디: {DEFAULT_ADMIN_ID}\n비밀번호: {DEFAULT_ADMIN_PW}\n\n"
                "관리자 탭에 로그인한 뒤 비밀번호를 바꾸는 걸 권장합니다."
            )

    # ---------------- 왼쪽: 제어판 ----------------
    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=T_PANEL, width=248)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        # ── 앱 헤더 ──────────────────────────────────────────────────
        hdr = tk.Frame(left, bg="#0a0f16", height=72)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="PRESSURE MONITOR", bg="#0a0f16", fg=T_TEXT,
                 font=("Consolas", 10, "bold")).place(relx=0.5, rely=0.38, anchor="center")
        tk.Label(hdr, text="1×16 ch  ·  STM32 + AD7175",
                 bg="#0a0f16", fg=T_DIM, font=("Consolas", 7)).place(
            relx=0.5, rely=0.72, anchor="center")

        def _sec(text):
            """섹션 헤더: 파란 레이블 + 가로 구분선"""
            f = tk.Frame(left, bg=T_PANEL)
            f.pack(fill=tk.X, padx=14, pady=(13, 5))
            tk.Label(f, text=text, bg=T_PANEL, fg=T_BLUE,
                     font=("Consolas", 7, "bold")).pack(side=tk.LEFT)
            tk.Frame(f, bg=T_BORD, height=1).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0), pady=4)

        # ── CONNECTION ───────────────────────────────────────────────
        _sec("CONNECTION")

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

        # ── DISPLAY ──────────────────────────────────────────────────
        _sec("DISPLAY")

        icard = tk.Frame(left, bg=T_CARD,
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
                 bg=T_CARD, fg=T_DIM, font=("Consolas", 6)).pack(
            anchor="w", padx=10, pady=(0, 7))

        self.reset_btn = tk.Button(
            left, text="실시간 범위 초기화", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=4, command=self.reset_live_range)
        self.reset_btn.pack(padx=14, pady=(0, 2), fill=tk.X)

        # ── DATA ─────────────────────────────────────────────────────
        _sec("DATA")

        self.save_btn = tk.Button(
            left, text="CSV 저장", font=("맑은 고딕", 9, "bold"),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, state=tk.DISABLED, command=self.save_csv)
        self.save_btn.pack(padx=14, pady=(0, 5), fill=tk.X)
        self.record_count_label = tk.Label(
            left, text="기록  0 줄", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 8))
        self.record_count_label.pack(anchor="w", padx=14)

        # ── STATUS ───────────────────────────────────────────────────
        _sec("STATUS")

        self.status_label = tk.Label(
            left, text="●  연결 안 됨", bg=T_PANEL, fg=T_DIM,
            font=("맑은 고딕", 9))
        self.status_label.pack(anchor="w", padx=14, pady=(0, 2))
        self.info_label = tk.Label(
            left, text="", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7), justify="left")
        self.info_label.pack(anchor="w", padx=14)
        self.scale_label = tk.Label(
            left, text=f"범위  {SCALE_MIN} – {SCALE_MAX}", bg=T_PANEL,
            fg=T_DIM, font=("Consolas", 7))
        self.scale_label.pack(anchor="w", padx=14, pady=(4, 0))
        self.range_label = tk.Label(
            left, text="실시간  –", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7))
        self.range_label.pack(anchor="w", padx=14, pady=(1, 0))
        tk.Button(
            left, text="이상 로그 열기", font=("Consolas", 7),
            bg=T_PANEL, fg=T_DIM, activebackground=T_CARD,
            relief=tk.FLAT, pady=2, command=self._open_anomaly_log
        ).pack(anchor="w", padx=14, pady=(5, 0))

        # ── CALIBRATION ──────────────────────────────────────────────
        _sec("CALIBRATION")

        tk.Button(
            left, text="보정 파일 불러오기", font=("맑은 고딕", 8),
            bg=T_CARD, fg=T_DIM, activebackground=T_BORD,
            relief=tk.FLAT, pady=5, command=self._load_calibration_file
        ).pack(padx=14, pady=(0, 4), fill=tk.X)
        self.cal_loaded_label = tk.Label(
            left, text="파일 없음", bg=T_PANEL, fg=T_DIM,
            font=("Consolas", 7), wraplength=215, justify="left")
        self.cal_loaded_label.pack(anchor="w", padx=14)
        tk.Label(left, text="우측 컨투어에 자동 반영",
                 bg=T_PANEL, fg=T_DIM, font=("Consolas", 6)
                 ).pack(anchor="w", padx=14, pady=(2, 0))

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
        sm = ScalarMappable(norm=Normalize(vmin=SCALE_MIN, vmax=SCALE_MAX), cmap="jet")
        sm.set_array([])
        for cax in (self.cax_raw, self.cax_cal):
            cb = self.fig.colorbar(sm, cax=cax)
            cb.ax.yaxis.set_tick_params(color="white")
            for lbl in cb.ax.get_yticklabels():
                lbl.set_color("white")

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

        # ── 채널 상태 바 ──────────────────────────────────────────────
        self.ch_status_label = tk.Label(
            right, text="●  전 채널 정상",
            bg=T_BG, fg=T_GREEN,
            font=("Consolas", 8), anchor="w"
        )
        self.ch_status_label.pack(fill=tk.X, padx=12, pady=(0, 5))

        self._draw_contour([0] * NUM_CHANNELS, force=True)

    # ---------------- 관리자 탭 ----------------
    def _build_admin_tab(self, parent):
        # 로그인 화면과 기록관리 화면, 두 개를 같은 자리에 만들어두고
        # 인증 여부에 따라 하나만 보이도록 전환한다.
        self.admin_login_frame = tk.Frame(parent, bg="#1e1e1e")
        self.admin_content_frame = tk.Frame(parent, bg="#1e1e1e")

        self._build_admin_login_view(self.admin_login_frame)
        self._build_admin_content_view(self.admin_content_frame)

        self.admin_login_frame.pack(fill=tk.BOTH, expand=True)

    def _build_admin_login_view(self, parent):
        box = tk.Frame(parent, bg="#252526")
        box.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(box, text="🔒  관리자 로그인", bg="#252526", fg="white",
                 font=("맑은 고딕", 15, "bold")).pack(pady=(30, 20), padx=60)

        tk.Label(box, text="아이디", bg="#252526", fg="#aaaaaa",
                 font=("맑은 고딕", 10)).pack(anchor="w", padx=30)
        self.admin_id_entry = tk.Entry(box, font=("Consolas", 12), width=22)
        self.admin_id_entry.pack(padx=30, pady=(2, 12))

        tk.Label(box, text="비밀번호", bg="#252526", fg="#aaaaaa",
                 font=("맑은 고딕", 10)).pack(anchor="w", padx=30)
        self.admin_pw_entry = tk.Entry(box, font=("Consolas", 12), width=22, show="●")
        self.admin_pw_entry.pack(padx=30, pady=(2, 6))
        self.admin_pw_entry.bind("<Return>", lambda e: self._try_admin_login())

        self.admin_login_error = tk.Label(
            box, text="", bg="#252526", fg="#f85149", font=("맑은 고딕", 9)
        )
        self.admin_login_error.pack(pady=(0, 6))

        tk.Button(
            box, text="로그인", font=("맑은 고딕", 11, "bold"),
            bg="#2ea043", fg="white", activebackground="#3fb950",
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
        top = tk.Frame(parent, bg="#1e1e1e")
        top.pack(fill=tk.X, padx=20, pady=(16, 8))

        tk.Label(top, text="CSV 기록 관리", bg="#1e1e1e", fg="white",
                 font=("맑은 고딕", 14, "bold")).pack(side=tk.LEFT)
        tk.Button(top, text="로그아웃", font=("맑은 고딕", 9),
                 bg="#3a3a3a", fg="white", relief=tk.FLAT,
                 command=self._admin_logout).pack(side=tk.RIGHT)

        tk.Label(top, text=f"({os.path.abspath(RECORDS_DIR)})", bg="#1e1e1e",
                fg="#666666", font=("Consolas", 8)).pack(side=tk.RIGHT, padx=10)

        # --- 파일 목록 ---
        list_frame = tk.Frame(parent, bg="#1e1e1e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 8))

        columns = ("name", "size", "mtime")
        style = ttk.Style()
        style.configure("Admin.Treeview", background="#252526", fieldbackground="#252526",
                        foreground="white", rowheight=24)
        style.configure("Admin.Treeview.Heading", background="#333333", foreground="white")

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
        bottom = tk.Frame(parent, bg="#1e1e1e")
        bottom.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.record_preview_label = tk.Label(
            bottom, text="파일을 선택하면 정보가 여기 표시됩니다.",
            bg="#1e1e1e", fg="#888888", font=("Consolas", 9), justify="left"
        )
        self.record_preview_label.pack(anchor="w", pady=(0, 10))

        btn_row = tk.Frame(bottom, bg="#1e1e1e")
        btn_row.pack(fill=tk.X)

        tk.Button(btn_row, text="새로고침", font=("맑은 고딕", 10),
                 bg="#3a3a3a", fg="white", relief=tk.FLAT,
                 command=self._refresh_records_list).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btn_row, text="폴더 열기", font=("맑은 고딕", 10),
                 bg="#3a3a3a", fg="white", relief=tk.FLAT,
                 command=self._open_records_folder).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="선택 파일 열기", font=("맑은 고딕", 10),
                 bg="#3a3a3a", fg="white", relief=tk.FLAT,
                 command=self._open_selected_record).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="선택 파일 삭제", font=("맑은 고딕", 10),
                 bg="#7a2222", fg="white", relief=tk.FLAT,
                 command=self._delete_selected_record).pack(side=tk.LEFT, padx=6)

        # --- 비밀번호 변경 ---
        pw_frame = tk.LabelFrame(parent, text="관리자 비밀번호 변경", bg="#1e1e1e",
                                 fg="#aaaaaa", font=("맑은 고딕", 9))
        pw_frame.pack(fill=tk.X, padx=20, pady=(0, 20))

        row = tk.Frame(pw_frame, bg="#1e1e1e")
        row.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(row, text="새 비밀번호", bg="#1e1e1e", fg="#aaaaaa",
                font=("맑은 고딕", 9)).pack(side=tk.LEFT)
        self.new_pw_entry = tk.Entry(row, font=("Consolas", 10), show="●", width=16)
        self.new_pw_entry.pack(side=tk.LEFT, padx=(6, 16))

        tk.Label(row, text="확인", bg="#1e1e1e", fg="#aaaaaa",
                font=("맑은 고딕", 9)).pack(side=tk.LEFT)
        self.new_pw_confirm_entry = tk.Entry(row, font=("Consolas", 10), show="●", width=16)
        self.new_pw_confirm_entry.pack(side=tk.LEFT, padx=(6, 16))

        tk.Button(row, text="변경", font=("맑은 고딕", 9, "bold"),
                 bg="#2ea043", fg="white", relief=tk.FLAT,
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

        # ── 좌측: 원본 (4095 - raw) ──────────────────────────────────
        raw_display = np.clip(SCALE_MAX - arr, SCALE_MIN, SCALE_MAX)
        Z_raw = np.tile(self._compute_display_row(raw_display), (self.STRIP_ROWS, 1))

        self.ax_raw.clear()
        self._style_axes(self.ax_raw)
        self.ax_raw.contourf(self.x_fine, self.y_axis, Z_raw,
                             levels=self.contour_levels, cmap="jet")

        # ── 우측: 보정 적용 (offset - raw) 또는 안내 텍스트 ────────────
        self.ax_cal.clear()
        self._style_axes(self.ax_cal)
        if self.cal_offsets:
            offsets = np.array([self.cal_offsets.get(i, float(SCALE_MAX))
                                for i in range(NUM_CHANNELS)])
            cal_display = np.clip(offsets - arr, SCALE_MIN, SCALE_MAX)
            Z_cal = np.tile(self._compute_display_row(cal_display), (self.STRIP_ROWS, 1))
            self.ax_cal.contourf(self.x_fine, self.y_axis, Z_cal,
                                 levels=self.contour_levels, cmap="jet")
        else:
            self.ax_cal.set_facecolor("#1a1a1a")
            self.ax_cal.text(
                0.5, 0.5,
                "보정값 없음\n보정 탭에서 계산하거나\nJSON 파일을 불러오세요",
                transform=self.ax_cal.transAxes,
                ha="center", va="center", color="#555555",
                fontsize=9
            )

        # ── 수치 테이블 갱신 + 채널 이상 감지 ────────────────────────────
        warn_channels = []

        for i in range(NUM_CHANNELS):
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
            if self.cal_offsets:
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
                else:
                    self._log_anomaly(i, "복구", prev, raw_v, delta)
                self._prev_anomaly[i] = reason

            if reason:
                warn_channels.append((i, reason))
                raw_bg, raw_fg = "#332200", "#ffcc00"
                cal_bg, cal_fg = "#332200", "#ffcc00"
            else:
                raw_bg, raw_fg = "#252526", "#cccccc"
                if delta is not None:
                    cal_bg = "#252526"
                    cal_fg = "#ff6b6b" if delta > 300 else "#4da3ff"
                else:
                    cal_bg, cal_fg = "#252526", "#4da3ff"

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
            fg="#4da3ff"
        )
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    def _toggle_cal_apply(self):
        if not self.cal_offsets:
            return
        self.cal_apply = not self.cal_apply
        if self.cal_apply:
            self.cal_apply_btn.config(text="보정 적용: ON", bg="#c67a00", fg="white")
        else:
            self.cal_apply_btn.config(text="보정 적용: OFF", bg="#4a4a4a", fg="#888888")
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
            fg="#4da3ff"
        )
        try:
            self._draw_contour(self.current_values, force=True)
        except Exception:
            pass

    # ---------------- 보정 탭 ----------------
    def _build_calibration_tab(self, parent):
        self.cal_data = []
        self.cal_filepath = ""
        self.cal_offsets = {}

        left = tk.Frame(parent, bg="#252526", width=220)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="압력 보정", bg="#252526", fg="white",
                 font=("맑은 고딕", 14, "bold")).pack(pady=(20, 6))
        tk.Label(left, text="무압력 상태에서 기록한 CSV로\n채널별 영점 오프셋을 계산합니다.",
                 bg="#252526", fg="#888888", font=("맑은 고딕", 8),
                 justify="center").pack(pady=(0, 10))
        tk.Frame(left, bg="#3a3a3a", height=1).pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Label(left, text="① CSV 업로드", bg="#252526", fg="#aaaaaa",
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20)
        tk.Button(
            left, text="CSV 업로드", font=("맑은 고딕", 11, "bold"),
            bg="#2a6eaa", fg="white", activebackground="#3a7eba",
            relief=tk.FLAT, height=2, command=self._upload_cal_csv
        ).pack(padx=20, pady=(4, 4), fill=tk.X)

        self.cal_file_label = tk.Label(
            left, text="파일이 선택되지 않음", bg="#252526", fg="#666666",
            font=("Consolas", 8), wraplength=180, justify="left"
        )
        self.cal_file_label.pack(anchor="w", padx=20, pady=(0, 6))

        tk.Button(
            left, text="샘플 CSV 생성", font=("맑은 고딕", 9),
            bg="#3a3a3a", fg="#aaaaaa", relief=tk.FLAT,
            command=self._generate_sample_csv
        ).pack(padx=20, pady=(0, 14), fill=tk.X)

        tk.Frame(left, bg="#3a3a3a", height=1).pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Label(left, text="② 보정 실행", bg="#252526", fg="#aaaaaa",
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20)
        self.cal_run_btn = tk.Button(
            left, text="보  정", font=("맑은 고딕", 13, "bold"),
            bg="#4a4a4a", fg="#888888", activebackground="#5a5a5a",
            relief=tk.FLAT, height=2, state=tk.DISABLED,
            command=self._run_calibration
        )
        self.cal_run_btn.pack(padx=20, pady=(4, 14), fill=tk.X)

        tk.Frame(left, bg="#3a3a3a", height=1).pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Label(left, text="채널별 평균값 (영점 기준)", bg="#252526", fg="#aaaaaa",
                 font=("맑은 고딕", 9, "bold")).pack(anchor="w", padx=20, pady=(0, 4))
        self.cal_result_text = tk.Text(
            left, font=("Consolas", 8), bg="#1a1a1a", fg="#cccccc",
            relief=tk.FLAT, height=17, state=tk.DISABLED
        )
        self.cal_result_text.pack(padx=20, pady=(0, 8), fill=tk.X)

        self.cal_save_btn = tk.Button(
            left, text="보정값 저장 (JSON)", font=("맑은 고딕", 10, "bold"),
            bg="#4a4a4a", fg="#888888", relief=tk.FLAT,
            state=tk.DISABLED, command=self._save_calibration
        )
        self.cal_save_btn.pack(padx=20, pady=(0, 20), fill=tk.X)

        right = tk.Frame(parent, bg="#1e1e1e")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="채널별 데이터  ━  파랑: 원본값  /  빨강 점선: 평균(오프셋)",
                 bg="#1e1e1e", fg="#aaaaaa",
                 font=("맑은 고딕", 9)).pack(pady=(12, 4))

        self.cal_fig = Figure(figsize=(7.2, 5.6), dpi=90, facecolor="#1e1e1e")
        self.cal_axes = []
        for i in range(NUM_CHANNELS):
            ax = self.cal_fig.add_subplot(4, 4, i + 1)
            ax.set_facecolor("#252526")
            ax.set_title(f"ch{i}", color="#aaaaaa", fontsize=7, pad=2)
            ax.tick_params(colors="#777777", labelsize=5)
            for spine in ax.spines.values():
                spine.set_color("#444444")
            self.cal_axes.append(ax)

        self.cal_fig.tight_layout(pad=0.8, h_pad=1.2, w_pad=0.5)
        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=right)
        self.cal_canvas.get_tk_widget().pack(pady=4, padx=10, fill=tk.BOTH, expand=True)
        self.cal_canvas.draw()

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
            fg="#4da3ff"
        )
        self.cal_run_btn.config(state=tk.NORMAL, bg="#c67a00", fg="white")

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

        elapsed_list = []
        channels = {i: [] for i in range(NUM_CHANNELS)}
        for row in self.cal_data:
            try:
                elapsed_list.append(float(row["elapsed_ms"]))
                for i in range(NUM_CHANNELS):
                    channels[i].append(float(row[f"ch{i}"]))
            except (KeyError, ValueError):
                continue

        if not elapsed_list:
            messagebox.showwarning("파싱 오류", "데이터를 읽을 수 없습니다. CSV 헤더를 확인하세요.")
            return

        elapsed = np.array(elapsed_list)

        # 채널별 평균 계산 → 영점 오프셋
        self.cal_offsets = {}
        for i in range(NUM_CHANNELS):
            self.cal_offsets[i] = float(np.mean(channels[i]))

        # 16채널 그래프 그리기
        for i, ax in enumerate(self.cal_axes):
            ax.clear()
            ax.set_facecolor("#252526")
            arr = np.array(channels[i])
            mean_val = self.cal_offsets[i]

            ax.plot(elapsed, arr, color="#4da3ff", linewidth=0.8, alpha=0.9)
            ax.axhline(mean_val, color="#ff6b6b", linewidth=1.2, linestyle="--")
            ax.set_title(f"ch{i}  {mean_val:.0f}", color="#cccccc", fontsize=6.5, pad=2)
            ax.tick_params(colors="#777777", labelsize=5)
            for spine in ax.spines.values():
                spine.set_color("#444444")
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

        self.cal_save_btn.config(state=tk.NORMAL, bg="#2ea043", fg="white")
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

    def on_close(self):
        if self.reader:
            self.reader.stop()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()