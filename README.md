Purpose & context

JG is a developer building embedded hardware + Python UI systems, currently focused on a real-time pressure sensor monitoring application for a foot pressure scanner. The hardware stack involves an STM32F103R board with an external AD7175 (24-bit SPI ADC) and a 1×16 pressure sensor array communicating over USB CDC serial. JG has self-described as a beginner and prefers step-by-step explanations with reasoning provided for each action.

A secondary interest involves writing resume/portfolio descriptions of this project in both Korean and English, with an emphasis on replacing outsourced vendor dependency with independently developed in-house technology.

Current state

The core application (pressure_monitor.py) is largely feature-complete, including:

USB CDC serial connection with STREAM_ON/STREAM_OFF commands
Parsing of @index,timestamp,ch0..ch15 formatted data at 10ms intervals
Jet colormap contourf heatmap embedded in tkinter (TkAgg), with fixed 0–4095 color scale and 4095 - raw inversion (unpressed=blue, pressed=red)
Smoothness slider using cell-based, peak-preserving interpolation (not Gaussian blending)
Queue-based architecture separating serial thread from UI render thread, with frame-skipping to prevent latency accumulation
CSV auto-save to records/ folder on Stop, plus manual export
Admin tab (ttk.Notebook) with SHA-256 login, CSV record management, and in-app password change

A companion file (wafersafe_brush_monitor.py) was updated to use the same peak-preserving interpolation method.

Pending task at last session end: Calibration CSV template generation and an upload/apply button for per-channel calibration.

On the horizon

Completing the calibration CSV template + upload/apply UI feature
Potential PyInstaller packaging into a standalone executable for distribution

Key learnings & principles

Colorbar lifecycle: colorbar.remove() + recreation causes AttributeError and eventually RecursionError after many frames — solution is to create the colorbar once via ScalarMappable and never recreate it.
contourf levels must be explicit: Passing an integer to levels= ignores vmin/vmax and recomputes from data each frame, causing near-zero scale artifacts when all values are identical. Always pass np.linspace(0, 4095, 26).
Peak-preserving interpolation: Gaussian/neighbor-blending reduces peak values. The cell-based method ensures each channel's center region always equals its raw ADC value; only boundary zones blend.
AD7175 note for resume use: Datasheet spec is 24-bit, but actual implementation operates in 0–4095 range — this discrepancy should be flagged before any public claims.
Resume writing caveat: ADC Device ID issue should be described only as "root cause identified," not "resolved," unless confirmed otherwise.

Approach & patterns

JG prefers iterative development with correction loops — bugs are identified through testing and Claude corrects each issue explicitly.
For resume/writing tasks, JG prefers copy-paste-ready plain prose over bullet lists or tables, in both Korean and English.
Python environment: Python 3.11 (downgraded from newer version due to DLL compatibility issues for PyInstaller packaging), using a virtual environment.

Tools & resources

Hardware: STM32F103R, AD7175 SPI ADC, 1×16 pressure sensor array
Communication: USB CDC serial, Tera Term for verification
Python stack: pyserial, numpy, matplotlib (TkAgg backend), tkinter, ttk, PyInstaller
Python version: 3.11 (pinned for packaging compatibility)

Other instructions

Python UI 작업 시: 파일을 제공하기 전에 항상 레이아웃 목업(창 전체 구성)을 먼저 보여주고, 확인 후 파일을 생성한다.
(When working on Python UI tasks: always show a full layout mockup of the entire window first, then generate the file only after JG confirms.)
