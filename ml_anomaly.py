"""
LSTM Autoencoder 기반 압력 이상 감지 모델 훈련 스크립트

Usage:
    python ml_anomaly.py pressure_data.csv

출력:
    anomaly_model.pt   - 학습된 모델 가중치
    anomaly_stats.npz  - 복원 오차 통계 (임계값 포함)
"""
import sys
import os
import csv
import numpy as np

ACTIVE_CH  = [0, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13]  # ch01,05,14,15 사망
SEQ_LEN    = 30    # 슬라이딩 윈도우 (300ms)
HIDDEN     = 32
EPOCHS     = 80
LR         = 1e-3
BATCH      = 64
MODEL_PATH = 'anomaly_model.pt'
STATS_PATH = 'anomaly_stats.npz'


def load_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    frames = []
    for row in rows:
        try:
            frames.append([float(row[f'ch{i}']) for i in range(16)])
        except (KeyError, ValueError):
            continue
    if not frames:
        raise ValueError("ch0~ch15 컬럼을 찾을 수 없습니다.")
    return np.array(frames, dtype=np.float32)


def preprocess(arr):
    """ADC raw → 정규화된 압력 (0=무압력, 1=최대압력)"""
    active = arr[:, ACTIVE_CH]
    return (4095 - active) / 4095.0


def make_windows(data, seq_len):
    n = len(data) - seq_len
    if n <= 0:
        raise ValueError(f"데이터가 너무 짧습니다 (필요: >{seq_len} 프레임)")
    return np.stack([data[i:i + seq_len] for i in range(n)])


def build_model():
    import torch.nn as nn
    n = len(ACTIVE_CH)

    class LSTMAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(n, HIDDEN, batch_first=True)
            self.decoder = nn.LSTM(HIDDEN, n, batch_first=True)

        def forward(self, x):
            _, (h, _) = self.encoder(x)
            rep = h[-1].unsqueeze(1).expand(-1, x.size(1), -1)
            out, _ = self.decoder(rep)
            return out

    return LSTMAutoencoder()


def train(csv_path, save_dir=None, print_fn=print):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(csv_path))

    print_fn(f"[ML] 파일 로드: {os.path.basename(csv_path)}")
    raw  = load_csv(csv_path)
    data = preprocess(raw)
    X    = make_windows(data, SEQ_LEN)
    print_fn(f"[ML] 프레임={len(raw)}, 윈도우={len(X)}, 활성채널={len(ACTIVE_CH)}")

    tensor = torch.from_numpy(X)
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH, shuffle=True)

    model   = build_model()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for (batch,) in loader:
            pred = model(batch)
            loss = loss_fn(pred, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if (epoch + 1) % 20 == 0:
            print_fn(f"[ML] Epoch {epoch+1}/{EPOCHS}  loss={total/len(X):.6f}")

    # 훈련 복원 오차 통계 → 임계값
    model.eval()
    with torch.no_grad():
        errors = ((model(tensor) - tensor) ** 2).mean(dim=(1, 2)).numpy()

    mean_e = float(errors.mean())
    std_e  = float(errors.std())
    thresh = mean_e + 3.0 * std_e

    # σ배수별 임계값 및 훈련 데이터 기준 실측 이상률
    sigma_ks      = np.arange(1.0, 6.5, 0.5)
    thresholds_arr = mean_e + sigma_ks * std_e
    fp_rates      = np.array([(errors > t).mean() for t in thresholds_arr])

    mp = os.path.join(save_dir, MODEL_PATH)
    sp = os.path.join(save_dir, STATS_PATH)
    torch.save(model.state_dict(), mp)
    np.savez(sp,
             mean_err=mean_e, std_err=std_e, threshold=thresh,
             n_windows=len(X),
             sigma_ks=sigma_ks, thresholds=thresholds_arr, fp_rates=fp_rates)

    print_fn(f"[ML] 복원오차: mean={mean_e:.6f}  std={std_e:.6f}")
    print_fn(f"[ML] 이상 임계값 (mean+3σ): {thresh:.6f}")
    print_fn(f"[ML] 저장: {mp}")
    return mean_e, std_e, thresh


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'pressure_20260820_143332.csv'
    if not os.path.exists(path):
        print(f"파일 없음: {path}")
        sys.exit(1)
    train(path)
    print("완료.")
