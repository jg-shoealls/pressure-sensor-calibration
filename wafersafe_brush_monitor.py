"""
WaferSafe Brush Monitor 스타일의 1x16 압력센서 실시간 모니터
STM32F103 + AD7175 / Python Tkinter + Matplotlib

핵심 변경점
- 첨부 사진과 유사한 밝은 좌측 제어 패널 + 어두운 16채널 모니터 화면
- Measuring / Admin 탭
- Calibration Data / Serial Port / Settings / Control / Export 그룹
- S1~S16 구획, 각 채널 값 표시, jet 컬러바
- 기존 시리얼 수신 형식 유지: @index,timestamp,ch0,...,ch15
- CSV 저장 / 이미지 저장 / 압력 CSV 불러오기
- 선택형 Calibration CSV 지원

Calibration CSV 형식(선택 기능):
channel,slope,offset,unit
S1,0.001,0.0,kPa
S2,0.001,0.0,kPa
...
S16,0.001,0.0,kPa

Calibration CSV를 불러오지 않으면 원시 ADC 값을 그대로 표시합니다.

필요 라이브러리:
    pip install pyserial numpy matplotlib
"""

import csv
import datetime
import queue
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import numpy as np
import serial
from serial.tools import list_ports

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


NUM_CHANNELS = 16
DEFAULT_PORT = "COM3"
DEFAULT_BAUD = 115200


# -----------------------------------------------------------------------------
# Serial reader
# -----------------------------------------------------------------------------
class SerialReader(threading.Thread):
    """시리얼 포트를 별도 스레드에서 읽고 파싱 결과를 Queue에 넣는다."""

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
            self.out_queue.put(("error", str(e)))
            return

        # STM32/USB-UART 환경에서 연결 직후 안정화 시간
        time.sleep(1.0)
        self.running = True
        self.out_queue.put(("connected", self.port))
        self.send_command("STREAM_ON")

        while self.running:
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                self.out_queue.put(("error", str(e)))
                break

            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("@"):
                continue

            parts = line.split(",")
            if len(parts) != NUM_CHANNELS + 2:
                continue

            try:
                index = int(parts[0][1:])
                timestamp = int(parts[1])
                adc_values = [int(v) for v in parts[2:]]
            except ValueError:
                continue

            self.out_queue.put(("data", index, timestamp, adc_values))

        self._safe_close()

    def send_command(self, cmd):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((cmd + "\r\n").encode("utf-8"))
            except serial.SerialException:
                pass

    def stop(self):
        self.running = False
        if self.ser and self.ser.is_open:
            self.send_command("STREAM_OFF")
            time.sleep(0.05)
            self._safe_close()

    def _safe_close(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
class App:
    UI_POLL_MS = 10
    PLOT_REDRAW_INTERVAL = 0.04   # 약 25 FPS. 10ms 수신이어도 UI는 덜 버벅이게 함
    SCALE_WINDOW = 200            # 최근 약 2초(10ms 기준) 범위

    BG = "#f3f4f7"
    PANEL_BG = "#f7f7fa"
    BORDER = "#d7d8de"
    TEXT = "#2b2d33"
    MUTED = "#777b86"
    BLUE = "#0b84e5"
    GREEN = "#35c56d"
    RED = "#e25c5c"
    ORANGE = "#ff8a1f"
    DARK_PLOT = "#171c28"
    GRID = "#d7dce8"

    def __init__(self, root):
        self.root = root
        self.root.title("WaferSafe Brush Monitor")
        self.root.geometry("1080x650")
        self.root.minsize(920, 560)
        self.root.configure(bg=self.BG)

        self.data_queue = queue.Queue()
        self.reader = None
        self.measuring = False

        # 실시간 데이터
        self.raw_values = [0] * NUM_CHANNELS
        self.display_values = [0.0] * NUM_CHANNELS
        self.current_index = 0
        self.current_timestamp = 0

        # Calibration: pressure = raw * slope + offset
        self.cal_slopes = np.ones(NUM_CHANNELS, dtype=float)
        self.cal_offsets = np.zeros(NUM_CHANNELS, dtype=float)
        self.calibration_loaded = False
        self.unit = "raw"

        # 자동 스케일
        self.scale_window = deque(maxlen=self.SCALE_WINDOW)
        self.observed_min = None
        self.observed_max = None

        # 기록
        self.recorded_rows = []
        self.record_start_time = None

        self._last_plot_draw = 0.0

        self._setup_styles()
        self._build_ui()
        self.refresh_ports()

        self.root.after(self.UI_POLL_MS, self.poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "App.TNotebook",
            background=self.BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "App.TNotebook.Tab",
            font=("Segoe UI", 10),
            padding=(24, 8),
            background="#ececf1",
            foreground="#555963",
            borderwidth=1,
        )
        style.map(
            "App.TNotebook.Tab",
            background=[("selected", self.BLUE)],
            foreground=[("selected", "white")],
        )

        style.configure(
            "Port.TCombobox",
            padding=4,
            fieldbackground="white",
            background="white",
        )

        style.configure(
            "Small.TSpinbox",
            padding=3,
            fieldbackground="white",
        )

    def _build_ui(self):
        # Notebook
        self.notebook = ttk.Notebook(self.root, style="App.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        self.measuring_tab = tk.Frame(self.notebook, bg=self.BG)
        self.admin_tab = tk.Frame(self.notebook, bg=self.BG)
        self.notebook.add(self.measuring_tab, text="Measuring")
        self.notebook.add(self.admin_tab, text="Admin")

        self._build_measuring_tab()
        self._build_admin_tab()

        # bottom status bar
        status = tk.Frame(self.root, bg="#ececf1", height=27)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)

        self.status_left = tk.Label(
            status,
            text="WaferSafe Brush Monitor  /  Ready",
            bg="#ececf1",
            fg="#6d7079",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.status_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

        self.status_right = tk.Label(
            status,
            text="",
            bg="#ececf1",
            fg="#6d7079",
            font=("Consolas", 9),
            anchor="e",
        )
        self.status_right.pack(side=tk.RIGHT, padx=10)

    def _build_measuring_tab(self):
        self.measuring_tab.grid_rowconfigure(0, weight=1)
        self.measuring_tab.grid_columnconfigure(1, weight=1)

        # Left panel
        left = tk.Frame(
            self.measuring_tab,
            bg=self.PANEL_BG,
            width=245,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10), pady=0)
        left.grid_propagate(False)

        self._build_left_panel(left)

        # Right monitor
        right = tk.Frame(self.measuring_tab, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=0)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        title = tk.Label(
            right,
            text="WaferSafe Brush Monitor",
            bg=self.BG,
            fg=self.TEXT,
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="ew", padx=4, pady=(10, 8))

        plot_holder = tk.Frame(
            right,
            bg="white",
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        plot_holder.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 10))

        self._build_plot(plot_holder)

    def _build_admin_tab(self):
        box = tk.Frame(
            self.admin_tab,
            bg="white",
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        box.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        tk.Label(
            box,
            text="Admin",
            bg="white",
            fg=self.TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=24, pady=(24, 8))

        tk.Label(
            box,
            text="관리자용 설정 영역입니다. 향후 센서별 보정값, 통신 설정, 장치 정보 등을 배치할 수 있습니다.",
            bg="white",
            fg=self.MUTED,
            font=("맑은 고딕", 10),
            justify="left",
        ).pack(anchor="w", padx=24)

    def _group(self, parent, title):
        frame = tk.LabelFrame(
            parent,
            text=title,
            bg=self.PANEL_BG,
            fg="#696d76",
            font=("Segoe UI", 10),
            bd=1,
            relief=tk.GROOVE,
            padx=10,
            pady=8,
        )
        frame.pack(fill=tk.X, padx=10, pady=(8, 0))
        return frame

    def _build_left_panel(self, left):
        # Calibration Data
        cal = self._group(left, "Calibration Data")
        self.cal_btn = tk.Button(
            cal,
            text="Load Calibration CSV",
            command=self.load_calibration_csv,
            bg=self.BLUE,
            fg="white",
            activebackground="#0877ca",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.cal_btn.pack(fill=tk.X, pady=(0, 5))

        self.cal_status = tk.Label(
            cal,
            text="Not loaded",
            bg=self.PANEL_BG,
            fg="#8b8f98",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.cal_status.pack(fill=tk.X)

        self.unit_label = tk.Label(
            cal,
            text="Unit: raw",
            bg=self.PANEL_BG,
            fg="#8b8f98",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.unit_label.pack(fill=tk.X)

        # Serial Port
        serial_box = self._group(left, "Serial Port")
        port_row = tk.Frame(serial_box, bg=self.PANEL_BG)
        port_row.pack(fill=tk.X)

        self.port_var = tk.StringVar(value=DEFAULT_PORT)
        self.port_combo = ttk.Combobox(
            port_row,
            textvariable=self.port_var,
            style="Port.TCombobox",
            state="normal",
            font=("Segoe UI", 10),
        )
        self.port_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.refresh_btn = tk.Button(
            port_row,
            text="↻",
            command=self.refresh_ports,
            bg="#6f727c",
            fg="white",
            activebackground="#5e616a",
            activeforeground="white",
            relief=tk.FLAT,
            width=3,
            font=("Segoe UI Symbol", 11, "bold"),
            cursor="hand2",
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=(6, 0), ipady=1)

        # Settings
        settings = self._group(left, "Settings")

        self.decimal_var = tk.IntVar(value=0)
        self.interp_var = tk.IntVar(value=5)

        row1 = tk.Frame(settings, bg=self.PANEL_BG)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(
            row1, text="Decimal:", bg=self.PANEL_BG, fg=self.MUTED,
            font=("Segoe UI", 9), width=12, anchor="w"
        ).pack(side=tk.LEFT)
        self.decimal_spin = ttk.Spinbox(
            row1,
            from_=0,
            to=4,
            width=6,
            textvariable=self.decimal_var,
            style="Small.TSpinbox",
            command=self._refresh_value_labels,
        )
        self.decimal_spin.pack(side=tk.RIGHT)

        row2 = tk.Frame(settings, bg=self.PANEL_BG)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(
            row2, text="Interpolation:", bg=self.PANEL_BG, fg=self.MUTED,
            font=("Segoe UI", 9), width=12, anchor="w"
        ).pack(side=tk.LEFT)
        self.interp_spin = ttk.Spinbox(
            row2,
            from_=0,
            to=10,
            width=6,
            textvariable=self.interp_var,
            style="Small.TSpinbox",
            command=lambda: self._draw_monitor(force=True),
        )
        self.interp_spin.pack(side=tk.RIGHT)

        # Control
        control = self._group(left, "Control")
        self.measure_btn = tk.Button(
            control,
            text="▶  Start Measuring",
            command=self.toggle_measurement,
            bg=self.GREEN,
            fg="white",
            activebackground="#2caf60",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
            height=2,
        )
        self.measure_btn.pack(fill=tk.X)

        # Export
        export = self._group(left, "Export")
        self.save_image_btn = tk.Button(
            export,
            text="Save Contour Image",
            command=self.save_contour_image,
            bg="#676a73",
            fg="white",
            activebackground="#575a62",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.save_image_btn.pack(fill=tk.X, pady=(0, 5))

        self.save_csv_btn = tk.Button(
            export,
            text="Save Pressure CSV",
            command=self.save_pressure_csv,
            bg="#676a73",
            fg="white",
            activebackground="#575a62",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
            state=tk.DISABLED,
        )
        self.save_csv_btn.pack(fill=tk.X, pady=(0, 5))

        self.load_csv_btn = tk.Button(
            export,
            text="📁  Load Pressure CSV",
            command=self.load_pressure_csv,
            bg=self.ORANGE,
            fg="white",
            activebackground="#eb7813",
            activeforeground="white",
            relief=tk.FLAT,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        self.load_csv_btn.pack(fill=tk.X)

        # small live info
        self.live_info = tk.Label(
            left,
            text="Ready",
            bg=self.PANEL_BG,
            fg="#989ba3",
            font=("Consolas", 8),
            justify="left",
            anchor="w",
        )
        self.live_info.pack(fill=tk.X, padx=12, pady=(8, 6))

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    def _build_plot(self, holder):
        self.fig = Figure(figsize=(8.0, 5.0), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes([0.045, 0.08, 0.86, 0.84])
        self.cax = self.fig.add_axes([0.925, 0.10, 0.025, 0.80])

        self.ax.set_facecolor(self.DARK_PLOT)
        self.ax.set_xlim(-0.5, NUM_CHANNELS - 0.5)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        for spine in self.ax.spines.values():
            spine.set_edgecolor("#8f96a5")
            spine.set_linewidth(0.8)

        # 2D heat field (시각적 표현용). 실제 센서는 1 x 16.
        # 채널 사이를 정수 배(SEGMENTS_PER_CHANNEL)로 나눠서 격자점이 채널의
        # 정수 위치(0,1,...,15)에 항상 정확히 맞도록 한다. 이렇게 안 하면
        # 부드러움(Interpolation) 값을 바꿀 때 채널 위치의 실제 값(피크)이
        # 격자 반올림 오차로 미세하게 흔들려 보일 수 있다.
        SEGMENTS_PER_CHANNEL = 40
        core = np.linspace(0, NUM_CHANNELS - 1,
                           (NUM_CHANNELS - 1) * SEGMENTS_PER_CHANNEL + 1)
        # 좌우 반 칸 여백(-0.5 ~ 0, 15 ~ 15.5)은 끝 채널값으로 평평하게 이어짐
        pad = SEGMENTS_PER_CHANNEL // 2
        left_pad = np.linspace(-0.5, 0, pad, endpoint=False)
        right_pad = np.linspace(NUM_CHANNELS - 1, NUM_CHANNELS - 0.5, pad + 1)[1:]
        self.x_grid = np.concatenate([left_pad, core, right_pad])
        self.y_grid = np.linspace(0.0, 1.0, 180)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)

        empty = np.zeros_like(self.X)
        self.image = self.ax.imshow(
            empty,
            extent=[-0.5, NUM_CHANNELS - 0.5, 0.0, 1.0],
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=0,
            vmax=1,
            interpolation="bilinear",
            alpha=np.zeros_like(empty),
            zorder=1,
        )

        # Sensor separators
        for i in range(NUM_CHANNELS + 1):
            self.ax.axvline(i - 0.5, color=self.GRID, linewidth=0.7, alpha=0.85, zorder=3)

        # Labels
        self.channel_labels = []
        self.value_labels = []
        for i in range(NUM_CHANNELS):
            ch = self.ax.text(
                i,
                0.73,
                f"S{i + 1}",
                color="#f3f5fb",
                fontsize=9,
                ha="center",
                va="center",
                zorder=4,
            )
            self.channel_labels.append(ch)

            val = self.ax.text(
                i,
                0.055,
                "0",
                color="#f3f5fb",
                fontsize=8,
                ha="center",
                va="center",
                zorder=4,
            )
            self.value_labels.append(val)

        # Colorbar
        self.norm = Normalize(vmin=0, vmax=1)
        self.sm = ScalarMappable(norm=self.norm, cmap="jet")
        self.sm.set_array([])
        self.colorbar = self.fig.colorbar(self.sm, cax=self.cax)
        self.colorbar.ax.tick_params(labelsize=8, colors="#646873")
        self.colorbar.outline.set_edgecolor("#8f96a5")
        self.cax.set_facecolor("white")

        self.canvas = FigureCanvasTkAgg(self.fig, master=holder)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._draw_monitor(force=True)

    def _refresh_value_labels(self):
        self._draw_monitor(force=True)

    def _apply_calibration(self, raw_values):
        raw = np.asarray(raw_values, dtype=float)
        return raw * self.cal_slopes + self.cal_offsets

    def _update_scale(self, values):
        arr = np.asarray(values, dtype=float)
        v_min = float(np.nanmin(arr))
        v_max = float(np.nanmax(arr))
        self.scale_window.append((v_min, v_max))

        self.observed_min = min(v[0] for v in self.scale_window)
        self.observed_max = max(v[1] for v in self.scale_window)

    def _peak_preserving_row(self, values, xs, smoothness):
        """각 채널은 자기 칸(폭 1) 안에서 대부분 자기 값 그대로 평평하게
        유지되고, 칸과 칸의 경계 쪽 좁은 구간에서만 이웃과 부드럽게
        이어진다 (경계 부분의 부드러움만 smoothness 0~100으로 조절됨).

        예전 방식(채널 사이 전체 폭을 걸쳐 직선/S자로 잇는 방식)은, 예를
        들어 채널11만 안눌리고 양옆(10,12)이 둘 다 눌린 경우 채널11의
        칸 전체가 옆의 눌린 색으로 뭉개져 보이는 문제가 있었다. 이 방식은
        칸의 중심부(core)는 절대 안 건드리고, 경계에 가까운 좁은 구간
        (edge)만 이웃과 섞이므로 그 문제가 생기지 않는다.

        smoothness=0   : 경계 폭 0 (완전한 계단식, 칸 전체가 자기 값)
        smoothness=100 : 경계 폭이 칸 절반까지 넓어짐 (칸 경계에서 정확히
                         두 채널의 평균값과 만나는 부드러운 전환)
        """
        values = np.asarray(values, dtype=float)
        xs_clamped = np.clip(xs, 0, NUM_CHANNELS - 1)
        c = np.round(xs_clamped).astype(int)          # 이 격자점이 속한 칸(채널)
        d = xs_clamped - c                              # 칸 중심으로부터의 오프셋(-0.5~0.5)

        half_w = 0.5
        edge = (smoothness / 100.0) * half_w            # 경계 부드러움 폭
        core = half_w - edge                             # 칸 중심의 "무조건 원본값" 구간 폭

        ad = np.abs(d)
        result = values[c].copy()

        edge_mask = ad > core
        if edge > 1e-9 and np.any(edge_mask):
            neighbor = np.clip(c + np.sign(d).astype(int), 0, NUM_CHANNELS - 1)
            local_t = np.clip((ad - core) / edge, 0.0, 1.0)
            eased = 3 * local_t ** 2 - 2 * local_t ** 3   # smoothstep: 경계에서 부드럽게
            blended = values[c] * (1 - eased * 0.5) + values[neighbor] * (eased * 0.5)
            result = np.where(edge_mask, blended, result)

        return result

    def _make_heat_field(self, values, lo, hi):
        """
        pressure_monitor.py와 동일하게, X(채널) 방향으로만 값이 결정되고
        Y(세로) 방향은 위아래 어디든 완전히 동일한 색으로 채워진다.

        이전에는 Y방향에 가우시안 감쇠(브러시 형태)를 곱해서, 안 눌린
        채널까지도 위쪽은 어둡고 아래로 갈수록 밝아지는 세로 무지개
        그라데이션이 전체에 깔리는 문제가 있었다. pressure_monitor.py는
        이런 세로 효과가 없으므로 여기서도 없앤다.

        주의: 센서는 실제로 1행 16열이므로 Y방향은 시각화 목적일 뿐이고,
        지금은 그 목적이 "단순히 띠를 두껍게 보여주는 것" 뿐이다.
        """
        vals = np.asarray(values, dtype=float)
        span = max(hi - lo, 1e-12)
        normalized = np.clip((vals - lo) / span, 0.0, 1.0)

        try:
            interp = int(self.interp_var.get())
        except (ValueError, tk.TclError):
            interp = 5
        interp = np.clip(interp, 0, 10)
        smoothness = interp / 10.0 * 100.0  # 스핀박스 0~10 -> 부드러움 0~100

        # X방향: 채널 사이만 계단식~S자 곡선으로 잇는다 (채널 위치의 값은 불변)
        x_profile = self._peak_preserving_row(normalized, self.x_grid, smoothness)

        # Y방향으로 그대로 복제만 한다 (세로 효과 없음, pressure_monitor.py와 동일)
        heat = np.tile(x_profile, (len(self.y_grid), 1))
        heat = np.clip(heat, 0.0, 1.0)

        # 실제 colorbar 값 범위로 환산
        field = lo + heat * span

        # 항상 불투명 (예전처럼 낮은 영역만 투명해지는 효과 없음)
        alpha = np.full_like(heat, 0.95)
        return field, alpha

    def _draw_monitor(self, force=False):
        now = time.time()
        if not force and (now - self._last_plot_draw) < self.PLOT_REDRAW_INTERVAL:
            return
        self._last_plot_draw = now

        values = np.asarray(self.display_values, dtype=float)

        if self.observed_min is None or self.observed_max is None:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = float(self.observed_min), float(self.observed_max)
            if hi <= lo:
                # 값이 모두 같은 경우 화면/컬러바가 깨지지 않도록 작은 범위 생성
                delta = max(abs(lo) * 0.05, 1.0)
                lo -= delta * 0.5
                hi += delta * 0.5

        field, alpha = self._make_heat_field(values, lo, hi)
        self.image.set_data(field)
        self.image.set_alpha(alpha)
        self.image.set_clim(lo, hi)

        self.norm.vmin = lo
        self.norm.vmax = hi
        self.sm.set_norm(self.norm)
        self.colorbar.update_normal(self.sm)

        try:
            decimals = int(self.decimal_var.get())
        except (ValueError, tk.TclError):
            decimals = 0
        decimals = max(0, min(decimals, 4))

        fmt = f"{{:.{decimals}f}}"
        for i, text_artist in enumerate(self.value_labels):
            text_artist.set_text(fmt.format(values[i]))

        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def load_calibration_csv(self):
        path = filedialog.askopenfilename(
            title="Calibration CSV 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields = {str(x).strip().lower() for x in (reader.fieldnames or [])}

                if not {"channel", "slope", "offset"}.issubset(fields):
                    raise ValueError(
                        "Calibration CSV에는 channel, slope, offset 열이 필요합니다.\n"
                        "예: S1,0.001,0.0,kPa"
                    )

                slopes = np.ones(NUM_CHANNELS, dtype=float)
                offsets = np.zeros(NUM_CHANNELS, dtype=float)
                found = set()
                unit = None

                for row in reader:
                    # key의 대소문자 차이를 흡수
                    norm_row = {str(k).strip().lower(): v for k, v in row.items()}
                    ch_text = str(norm_row.get("channel", "")).strip().upper()
                    idx = self._channel_to_index(ch_text)
                    if idx is None:
                        continue

                    slopes[idx] = float(norm_row["slope"])
                    offsets[idx] = float(norm_row["offset"])
                    found.add(idx)

                    u = str(norm_row.get("unit", "")).strip()
                    if u:
                        unit = u

                if len(found) != NUM_CHANNELS:
                    missing = [f"S{i + 1}" for i in range(NUM_CHANNELS) if i not in found]
                    raise ValueError("16개 채널 보정값이 모두 필요합니다. 누락: " + ", ".join(missing))

            self.cal_slopes = slopes
            self.cal_offsets = offsets
            self.unit = unit or "cal"
            self.calibration_loaded = True
            self.cal_status.config(text="Loaded", fg="#3b9b61")
            self.unit_label.config(text=f"Unit: {self.unit}")

            self.display_values = self._apply_calibration(self.raw_values).tolist()
            self.reset_scale(redraw=False)
            self._update_scale(self.display_values)
            self._draw_monitor(force=True)
            self._set_status(f"Calibration loaded: {path}")

        except Exception as e:
            messagebox.showerror("Calibration CSV 오류", str(e))

    @staticmethod
    def _channel_to_index(ch_text):
        t = ch_text.strip().upper()
        if t.startswith("S"):
            t = t[1:]
            try:
                n = int(t)
                return n - 1 if 1 <= n <= NUM_CHANNELS else None
            except ValueError:
                return None

        if t.startswith("CH"):
            t = t[2:]
            try:
                n = int(t)
                return n if 0 <= n < NUM_CHANNELS else None
            except ValueError:
                return None

        try:
            n = int(t)
            if 1 <= n <= NUM_CHANNELS:
                return n - 1
            if 0 <= n < NUM_CHANNELS:
                return n
        except ValueError:
            pass
        return None

    # ------------------------------------------------------------------
    # Serial control
    # ------------------------------------------------------------------
    def refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.port_combo["values"] = ports

        current = self.port_var.get().strip()
        if not current:
            if ports:
                self.port_var.set(ports[0])
            else:
                self.port_var.set(DEFAULT_PORT)

        self._set_status("Serial ports refreshed")

    def toggle_measurement(self):
        if self.measuring:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("Serial Port", "COM 포트를 선택하거나 입력하세요.")
            return

        # 이전 reader가 남아있으면 정리
        if self.reader:
            self.reader.stop()
            self.reader = None

        # 새 세션 기록 초기화
        self.recorded_rows = []
        self.record_start_time = None
        self.scale_window.clear()
        self.observed_min = None
        self.observed_max = None

        self.reader = SerialReader(port, DEFAULT_BAUD, self.data_queue)
        self.reader.start()
        self.measuring = True

        self.measure_btn.config(
            text="■  Stop Measuring",
            bg=self.RED,
            activebackground="#cc4f4f",
        )
        self.port_combo.config(state=tk.DISABLED)
        self.refresh_btn.config(state=tk.DISABLED)
        self.save_csv_btn.config(state=tk.DISABLED)
        self._set_status(f"Connecting to {port} ...")

    def stop_stream(self):
        if self.reader:
            self.reader.stop()
            self.reader = None

        self.measuring = False
        self.measure_btn.config(
            text="▶  Start Measuring",
            bg=self.GREEN,
            activebackground="#2caf60",
        )
        self.port_combo.config(state="normal")
        self.refresh_btn.config(state=tk.NORMAL)

        if self.recorded_rows:
            self.save_csv_btn.config(state=tk.NORMAL)

        self._set_status("Stopped")

    # ------------------------------------------------------------------
    # Queue / data
    # ------------------------------------------------------------------
    def poll_queue(self):
        latest_data = None

        try:
            while True:
                item = self.data_queue.get_nowait()
                kind = item[0]

                if kind == "error":
                    self._handle_serial_error(item[1])

                elif kind == "connected":
                    self._set_status(f"Connected: {item[1]}")

                elif kind == "data":
                    # 10ms마다 모든 프레임을 Matplotlib로 그리면 느려질 수 있으므로
                    # 기록은 모두 남기고, 화면은 마지막 프레임 중심으로 갱신한다.
                    latest_data = item
                    self._record_data(item)

        except queue.Empty:
            pass

        if latest_data is not None:
            _, index, timestamp, adc_values = latest_data
            self.current_index = index
            self.current_timestamp = timestamp
            self.raw_values = adc_values
            self.display_values = self._apply_calibration(adc_values).tolist()
            self._update_scale(self.display_values)
            self._draw_monitor()

            self.live_info.config(
                text=(
                    f"Index: {index}\n"
                    f"Timestamp: {timestamp} ms\n"
                    f"Rows: {len(self.recorded_rows)}\n"
                    f"Range: {self.observed_min:.2f} ~ {self.observed_max:.2f}"
                )
            )
            self.status_right.config(text=f"{index}  |  {timestamp} ms")

        self.root.after(self.UI_POLL_MS, self.poll_queue)

    def _record_data(self, item):
        _, index, timestamp, adc_values = item
        if self.record_start_time is None:
            self.record_start_time = timestamp

        elapsed_ms = timestamp - self.record_start_time
        calibrated = self._apply_calibration(adc_values).tolist()

        # 원시 ADC + 표시값을 모두 보관
        self.recorded_rows.append(
            [index, elapsed_ms, timestamp] + adc_values + calibrated
        )

        if len(self.recorded_rows) == 1:
            self.save_csv_btn.config(state=tk.NORMAL)

    def _handle_serial_error(self, error_text):
        if self.reader:
            try:
                self.reader.stop()
            except Exception:
                pass
        self.reader = None
        self.measuring = False

        self.measure_btn.config(
            text="▶  Start Measuring",
            bg=self.GREEN,
            activebackground="#2caf60",
        )
        self.port_combo.config(state="normal")
        self.refresh_btn.config(state=tk.NORMAL)

        self._set_status("Serial error")
        messagebox.showerror(
            "Serial 연결 오류",
            f"{error_text}\n\n"
            "확인 사항:\n"
            "1) COM 포트 번호가 맞는지\n"
            "2) TeraTerm/Putty 등 다른 프로그램이 포트를 사용 중인지\n"
            "3) USB 케이블/보드 연결 상태가 정상인지",
        )

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------
    def save_contour_image(self):
        default_name = datetime.datetime.now().strftime("contour_%Y%m%d_%H%M%S.png")
        path = filedialog.asksaveasfilename(
            title="Contour Image 저장",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
            self._set_status(f"Image saved: {path}")
        except OSError as e:
            messagebox.showerror("저장 실패", str(e))

    def save_pressure_csv(self):
        if not self.recorded_rows:
            messagebox.showinfo("Save Pressure CSV", "저장할 데이터가 없습니다.")
            return

        default_name = datetime.datetime.now().strftime("pressure_%Y%m%d_%H%M%S.csv")
        path = filedialog.asksaveasfilename(
            title="Pressure CSV 저장",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        raw_header = [f"ch{i}" for i in range(NUM_CHANNELS)]
        disp_header = [f"S{i + 1}_{self.unit}" for i in range(NUM_CHANNELS)]
        header = ["index", "elapsed_ms", "raw_timestamp_ms"] + raw_header + disp_header

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.recorded_rows)
            self._set_status(f"CSV saved: {len(self.recorded_rows)} rows")
        except OSError as e:
            messagebox.showerror("저장 실패", str(e))

    def load_pressure_csv(self):
        path = filedialog.askopenfilename(
            title="Pressure CSV 불러오기",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 2:
                raise ValueError("CSV에 데이터 행이 없습니다.")

            header = rows[0]
            data_rows = rows[1:]
            last = data_rows[-1]

            values = self._extract_16_values_from_row(header, last)
            self.display_values = values
            self.raw_values = [int(round(v)) for v in values]

            self.scale_window.clear()
            self.observed_min = None
            self.observed_max = None
            self._update_scale(values)
            self._draw_monitor(force=True)

            self.live_info.config(
                text=f"Loaded CSV\nRows: {len(data_rows)}\nFile: {path.split('/')[-1]}"
            )
            self._set_status(f"Pressure CSV loaded: {len(data_rows)} rows")

        except Exception as e:
            messagebox.showerror("CSV 불러오기 오류", str(e))

    def _extract_16_values_from_row(self, header, row):
        """S1_* 우선, 없으면 ch0~ch15, 그래도 없으면 마지막 16개 숫자를 사용."""
        h_lower = [h.strip().lower() for h in header]

        # 1) calibrated/display columns: s1_xxx ... s16_xxx
        idxs = []
        for i in range(NUM_CHANNELS):
            prefix = f"s{i + 1}_"
            candidates = [j for j, h in enumerate(h_lower) if h.startswith(prefix)]
            if not candidates:
                idxs = []
                break
            idxs.append(candidates[0])
        if len(idxs) == NUM_CHANNELS:
            return [float(row[j]) for j in idxs]

        # 2) raw columns ch0 ... ch15
        idxs = []
        for i in range(NUM_CHANNELS):
            name = f"ch{i}"
            if name not in h_lower:
                idxs = []
                break
            idxs.append(h_lower.index(name))
        if len(idxs) == NUM_CHANNELS:
            raw = [float(row[j]) for j in idxs]
            return self._apply_calibration(raw).tolist()

        # 3) fallback: last 16 numeric cells
        nums = []
        for cell in row:
            try:
                nums.append(float(cell))
            except ValueError:
                pass
        if len(nums) < NUM_CHANNELS:
            raise ValueError("16개 채널 값을 찾을 수 없습니다.")
        return nums[-NUM_CHANNELS:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def reset_scale(self, redraw=True):
        self.scale_window.clear()
        self.observed_min = None
        self.observed_max = None
        if redraw:
            self._draw_monitor(force=True)

    def _set_status(self, text):
        self.status_left.config(text=f"WaferSafe Brush Monitor  /  {text}")

    def on_close(self):
        if self.reader:
            try:
                self.reader.stop()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
